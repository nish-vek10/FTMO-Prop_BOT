"""
Multi-Asset Chandelier Exit (Heikin-Ashi) — FundedNext-Style 2-Step (Demo)
Runs all configured assets concurrently on 5-minute bars.

- One MT5 login, multiple assets
- OANDA REST (requests.get) for candles per asset
- FundedNext prop rules (P1=8%, P2=5%, MaxDaily=5%, MaxOverall=10%) on $100k
- Risk per trade (per asset): 0.20% of INITIAL account
- Per-asset: MT5 symbol, OANDA ticker, SL/TP distances, sessions, partials, BE buffer, magic #
- Prints last 10 OANDA candles + HA table each time a new bar closes
- Manages partial TP and SL->BE when first TP is touched

!! IMPORTANT !!
- Fill/verify each asset's MT5 symbol and OANDA ticker for YOUR broker (placeholders included).
- SL/TP distances are in ABSOLUTE PRICE UNITS (e.g., XAU: $1.00, FX: 0.0010, NAS100: 1.0 index point).
- Some brokers use different "cash" symbols for indices; adjust accordingly.
- This demo uses market orders + post-fill SL (and then manages partial/BE). Adjust if you prefer.
"""

import os
import sys
import time
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

import pandas as pd
from pytz import timezone, UTC
import requests
import MetaTrader5 as mt5
from email.message import EmailMessage
from pathlib import Path

# =====================================================================================
# GLOBAL SETTINGS (TIMEZONE, LOGGING, IO)
# =====================================================================================

# Local/logging timezone (affects printing & session windows only)
local_tz = timezone("Europe/London")

def log(msg: str):
    ts = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {msg}", flush=True)

# =====================================================================================
# ACCOUNT / PROP RULES (FUNDEDNEXT-STYLE)
# =====================================================================================

ACCOUNT_SIZE_USD   = 100_000.00
PHASE1_TARGET_PCT  = 8.0   # FundedNext-style
PHASE2_TARGET_PCT  = 5.0
MAX_DAILY_LOSS_PCT = 5.0
MAX_OVERALL_LOSS_PCT = 10.0
CURRENT_PHASE      = 1            # 1 or 2
AUTO_CLOSE_ON_BREACH = True

# FTMO/FundedNext-style daily reset (use Prague as in your prior code)
PROP_RESET_TZ      = timezone("Europe/Prague")
PROP_RESET_HOUR    = 0
PROP_RESET_MIN     = 0

# =====================================================================================
# DATA / EMAIL / PATHS
# =====================================================================================

# OANDA (Practice) — fill your token
OANDA_TOKEN    = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
OANDA_BASE_URL = "https://api-fxpractice.oanda.com/v3"

# Email (optional)
ALERTS_ENABLED = True
EMAIL_ENABLED  = True
REPORT_DIR     = r"C:\Users\anish\OneDrive\Desktop\Anish\A - EAs Reports"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "anishv2610@gmail.com"
SMTP_PASS = "lpignmmkhgymwlpi"
EMAIL_TO  = ["anishmvekaria@gmail.com"]

FORCE_DAILY_REPORT_ON_START = False  # optionally send yesterday→now report on boot

# =====================================================================================
# MT5 LOGIN
# =====================================================================================

MT5_LOGIN          = 52521539
MT5_PASSWORD       = "$rYU7zfqzmka0!"
MT5_SERVER         = "ICMarketsSC-Demo"
MT5_TERMINAL_PATH  = r"C:\MT5\EA-MultiAsset\terminal64.exe"

# =====================================================================================
# ENGINE / STRATEGY DEFAULTS
# =====================================================================================

# Candle settings
NUM_CANDLES       = 500
USE_HEIKIN_ASHI   = True
ATR_PERIOD        = 1
ATR_MULT          = 1.85
SLIPPAGE          = 10  # MT5 slippage points (broker-specific)
REQUIRE_POST_START_CANDLE = True  # enforce "first bar after session open"

# =====================================================================================
# ASSET CONFIGURATION (EDIT ONLY THIS LIST TO ADD/REMOVE ASSETS)
# - risk_pct_initial: per-trade risk as % of INITIAL account (prop-style) → 0.20%
# - sl_usd_distance / tp_usd_distance in absolute price units
# - sessions: local London time windows
# - magic_number: distinct per asset
# - be buffer: small cushion for SL->BE (will also use spread if enabled)
# =====================================================================================

@dataclass
class AssetConfig:
    name: str
    mt5_symbol: str
    oanda_symbol: str
    risk_pct_initial: float
    sl_usd_distance: float
    tp1_usd_distance: float
    tp1_fraction: float
    tp2_usd_distance: float
    tp2_fraction: float
    breakeven_buffer_usd: float
    use_spread_for_be_buffer: bool
    session_windows: List[Tuple[str, str]]
    magic_number: int


# --- PRE-FILLED EXAMPLES (YOU CAN TWEAK/DELETE/ADD) ---
ASSETS: List[AssetConfig] = [
    # Metals
    AssetConfig(
        name="XAUUSD",
        mt5_symbol="XAUUSD",
        oanda_symbol="XAU_USD",
        risk_pct_initial=0.15,
        sl_usd_distance=5.0,        # Price in $
        tp1_usd_distance=6.0,       # TP1
        tp1_fraction=0.50,          # 50%
        tp2_usd_distance=12.0,      # TP2
        tp2_fraction=0.25,          # 25%
        breakeven_buffer_usd=0.20,
        use_spread_for_be_buffer=True,
        session_windows=[("06:00", "12:00"), ("13:00", "18:00")],
        magic_number=114470,
    ),
    # AssetConfig(
    #     name="XAGUSD",
    #     mt5_symbol="XAGUSD",          # verify
    #     oanda_symbol="XAG_USD",       # verify with OANDA list
    #     risk_pct_initial=0.20,
    #     sl_usd_distance=0.25,         # ~25c SL
    #     tp_usd_distance=0.50,         # ~50c TP
    #     partial_tp_fraction=0.50,
    #     breakeven_buffer_usd=0.02,    # 2c (plus spread if enabled)
    #     use_spread_for_be_buffer=True,
    #     session_windows=[("06:00","12:00"), ("13:00","18:00")],  # similar to XAU
    #     magic_number=114471,
    # ),

    # FX (pips expressed as price units: e.g. 15 pips in GBPUSD ≈ 0.0015)
    AssetConfig(
        name="GBPUSD",
        mt5_symbol="GBPUSD",
        oanda_symbol="GBP_USD",
        risk_pct_initial=0.15,
        sl_usd_distance=0.0011,         # Price in Pips
        tp1_usd_distance=0.0012,        # TP1
        tp1_fraction=0.50,              # 50%
        tp2_usd_distance=0.0024,        # TP2
        tp2_fraction=0.25,              # 25%
        breakeven_buffer_usd=0.0002,    # 2 pips min + spread
        use_spread_for_be_buffer=True,
        session_windows=[("07:00","18:00")],  # London + overlap
        magic_number=114472,
    ),
    AssetConfig(
        name="EURUSD",
        mt5_symbol="EURUSD",
        oanda_symbol="EUR_USD",
        risk_pct_initial=0.15,
        sl_usd_distance=0.0008,         # Price in Pips
        tp1_usd_distance=0.0010,        # TP1
        tp1_fraction=0.50,              # 50%
        tp2_usd_distance=0.0022,        # TP2
        tp2_fraction=0.25,              # 25%
        breakeven_buffer_usd=0.0002,    # 2 pips + spread
        use_spread_for_be_buffer=True,
        session_windows=[("07:00","18:00")],
        magic_number=114473,
    ),

    # Indices (points in price units; confirm your broker's point size)
    AssetConfig(
        name="NAS100",
        mt5_symbol="USTEC",
        oanda_symbol="NAS100_USD",
        risk_pct_initial=0.15,
        sl_usd_distance=25.0,           # Price in $
        tp1_usd_distance=35.0,          # TP1
        tp1_fraction=0.50,              # 50%
        tp2_usd_distance=70.0,          # TP2
        tp2_fraction=0.25,              # 25%
        breakeven_buffer_usd=3.5,       # small cushion + spread
        use_spread_for_be_buffer=True,
        session_windows=[("14:00","20:30")],  # US pre-open/early + power hour (UK local)
        magic_number=114474,
    ),
    AssetConfig(
        name="SPX500",
        mt5_symbol="US500",
        oanda_symbol="SPX500_USD",
        risk_pct_initial=0.15,
        sl_usd_distance=8.0,            # Price in $
        tp1_usd_distance=8.0,           # TP1
        tp1_fraction=0.50,              # 50%
        tp2_usd_distance=16.0,          # TP2
        tp2_fraction=0.25,              # 25%
        breakeven_buffer_usd=0.5,
        use_spread_for_be_buffer=True,
        session_windows=[("14:00","21:00")],
        magic_number=114475,
    ),
    AssetConfig(
        name="US30",
        mt5_symbol="US30",
        oanda_symbol="US30_USD",
        risk_pct_initial=0.15,
        sl_usd_distance=120.0,          # 120 pts SL (Price in $)
        tp1_usd_distance=120.0,         # TP1
        tp1_fraction=0.50,              # 50%
        tp2_usd_distance=240.0,         # TP2
        tp2_fraction=0.25,              # 25%
        breakeven_buffer_usd=5.0,
        use_spread_for_be_buffer=True,
        session_windows=[("14:00","20:30")],
        magic_number=114476,
    ),
    AssetConfig(
        name="GER40",
        mt5_symbol="DE40",
        oanda_symbol="DE30_EUR",
        risk_pct_initial=0.15,
        sl_usd_distance=40.0,           # 40 pts SL (Price in EUR)
        tp1_usd_distance=45.0,          # TP1
        tp1_fraction=0.50,              # 50%
        tp2_usd_distance=100.0,         # TP2
        tp2_fraction=0.25,              # 25%
        breakeven_buffer_usd=2.0,
        use_spread_for_be_buffer=True,
        session_windows=[("06:30","18:30")],  # Frankfurt + US overlap
        magic_number=114477,
    ),
]

AUTO_CLOSE_AT_SESSION_END = True

# =====================================================================================
# EMAIL / ALERTS / REPORTS
# =====================================================================================

Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

def _email_smoke_test():
    msg = EmailMessage()
    msg["Subject"] = "SMTP test — PropEA - FundedNext (Multi-Asset)"
    msg["From"] = SMTP_USER or "bot@localhost"
    msg["To"] = ", ".join(EMAIL_TO) if EMAIL_TO else (SMTP_USER or "me@localhost")
    msg.set_content("If you can read this, SMTP is working. :)")
    import smtplib
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
        s.starttls()
        if SMTP_USER:
            s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)
    print("[SMTP] Test email sent successfully.")

try:
    if EMAIL_ENABLED:
        _email_smoke_test()
except Exception as e:
    print(f"[SMTP] Test failed: {e}")

# Deal/Entry maps for pretty logs
DEAL_TYPE_MAP = {
    getattr(mt5, "DEAL_TYPE_BUY", 0): "BUY",
    getattr(mt5, "DEAL_TYPE_SELL", 1): "SELL",
    getattr(mt5, "DEAL_TYPE_BALANCE", 2): "BALANCE",
    getattr(mt5, "DEAL_TYPE_CREDIT", 3): "CREDIT",
    getattr(mt5, "DEAL_TYPE_CHARGE", 4): "CHARGE",
    getattr(mt5, "DEAL_TYPE_CORRECTION", 5): "CORRECTION",
    getattr(mt5, "DEAL_TYPE_BONUS", 6): "BONUS",
    getattr(mt5, "DEAL_TYPE_COMMISSION", 7): "COMMISSION",
    getattr(mt5, "DEAL_TYPE_COMMISSION_DAILY", 8): "COMMISSION_DAILY",
    getattr(mt5, "DEAL_TYPE_COMMISSION_MONTHLY", 9): "COMMISSION_MONTHLY",
    getattr(mt5, "DEAL_TYPE_COMMISSION_AGENT_DAILY", 10): "COMMISSION_AGENT_DAILY",
    getattr(mt5, "DEAL_TYPE_COMMISSION_AGENT_MONTHLY", 11): "COMMISSION_AGENT_MONTHLY",
    getattr(mt5, "DEAL_TYPE_INTEREST", 12): "INTEREST",
    getattr(mt5, "DEAL_TYPE_BUY_CANCELED", 13): "BUY_CANCELED",
    getattr(mt5, "DEAL_TYPE_SELL_CANCELED", 14): "SELL_CANCELED",
    getattr(mt5, "DEAL_DIVIDEND", 15): "DIVIDEND",
    getattr(mt5, "DEAL_DIVIDEND_FRANKED", 16): "DIVIDEND_FRANKED",
    getattr(mt5, "DEAL_TAX", 17): "TAX",
}

ENTRY_MAP = {
    getattr(mt5, "DEAL_ENTRY_IN", 0): "OPEN",
    getattr(mt5, "DEAL_ENTRY_OUT", 1): "CLOSE",
    getattr(mt5, "DEAL_ENTRY_INOUT", 2): "REVERSE",
    getattr(mt5, "DEAL_ENTRY_OUT_BY", 3): "CLOSE_BY",
}

def _fmt_dt_utc_to_tz(ts_seconds: int, tz):
    try:
        return datetime.fromtimestamp(int(ts_seconds), tz=UTC).astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return ""

class Alerter:
    def __init__(self):
        self._daily_breach_sent_key = None
        self._overall_breach_sent = False
        self._phase_sent = set()

    def _send_email(self, subject: str, body: str, attachments: list = None):
        print(f"[ALERT] {subject} | {body.replace(chr(10), ' | ')}")
        if not EMAIL_ENABLED:
            return
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = SMTP_USER or "bot@localhost"
            msg["To"] = ", ".join(EMAIL_TO) if EMAIL_TO else (SMTP_USER or "me@localhost")
            msg.set_content(body)
            for fp in attachments or []:
                try:
                    with open(fp, "rb") as f:
                        data = f.read()
                    msg.add_attachment(data, maintype="application", subtype="octet-stream",
                                       filename=os.path.basename(fp))
                except Exception as e:
                    print(f"[ALERT] Attachment failed: {fp} ({e})")
            import smtplib
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
                s.starttls()
                if SMTP_USER:
                    s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
            print("[ALERT] Email sent.")
        except Exception as e:
            print(f"[ALERT] Email error: {e}")

    def reset_day_flags(self):
        self._daily_breach_sent_key = None

    def breach_daily(self, day_key: str, closed: float, openp: float, limit: float, tzname: str):
        if self._daily_breach_sent_key == day_key:
            return
        self._daily_breach_sent_key = day_key
        subject = "FundedNext BREACH — Max Daily Loss"
        body = (f"*** DAILY BREACH ***\n"
                f"Closed today: ${closed:,.2f}\nOpen PnL: ${openp:,.2f}\n"
                f"Daily sum (C+O): ${closed+openp:,.2f} <= -${limit:,.2f}\n"
                f"Reset TZ: {tzname}\n")
        self._send_email(subject, body)

    def breach_overall(self, equity: float, min_equity: float):
        if self._overall_breach_sent:
            return
        self._overall_breach_sent = True
        subject = "FundedNext BREACH — Max Overall Loss"
        body = (f"*** OVERALL BREACH ***\n"
                f"Equity: ${equity:,.2f}\nRequired minimum equity: ${min_equity:,.2f}\n")
        self._send_email(subject, body)

    def phase_passed(self, phase: int, gain: float, target: float):
        if phase in self._phase_sent:
            return
        self._phase_sent.add(phase)
        subject = f"FundedNext Phase {phase} PASSED"
        body = (f"Congratulations — Phase {phase} target reached!\n"
                f"Gain: ${gain:,.2f} >= Target: ${target:,.2f}\n")
        self._send_email(subject, body)

    def daily_report(self, csv_path: str, net_pnl: float, date_label: str, summary: str = None):
        subject = f"FundedNext Daily Report — {date_label}"
        body = (summary or "") + f"\nNet PnL (Closed+Open at send time): ${net_pnl:,.2f}\nFile: {csv_path}\n"
        self._send_email(subject, body, attachments=[csv_path] if csv_path else None)

# =====================================================================================
# PROP RULES (Shared across all assets)
# =====================================================================================

class PropRules:
    def __init__(self, initial_account_size: float, daily_loss_pct: float, overall_loss_pct: float,
                 phase1_target_pct: float, phase2_target_pct: float, reset_tz: timezone,
                 reset_hour: int, reset_minute: int, phase: int = 1,
                 auto_close_on_breach: bool = True, alerter: 'Alerter' = None):
        self.initial = float(initial_account_size)
        self.daily_limit = self.initial * (daily_loss_pct / 100.0)
        self.overall_limit = self.initial * (overall_loss_pct / 100.0)
        self.phase1_target = self.initial * (phase1_target_pct / 100.0)
        self.phase2_target = self.initial * (phase2_target_pct / 100.0)
        self.phase = 1 if phase == 1 else 2
        self.reset_tz = reset_tz
        self.reset_hour = reset_hour
        self.reset_minute = reset_minute
        self.auto_close = auto_close_on_breach
        self.alerter = alerter
        self.today_anchor_equity = None
        self.last_reset_at = None

    def _now_reset_tz(self):
        return datetime.now(self.reset_tz)

    def _next_reset_time(self, now_tz):
        candidate = now_tz.replace(hour=self.reset_hour, minute=self.reset_minute, second=0, microsecond=0)
        if now_tz >= candidate:
            candidate += timedelta(days=1)
        return candidate

    def ensure_daily_anchor(self):
        now = self._now_reset_tz()
        if self.last_reset_at is None:
            eq = self.get_mt5_equity()
            self.today_anchor_equity = eq
            self.last_reset_at = now.replace(hour=self.reset_hour, minute=self.reset_minute, second=0, microsecond=0)
            print(f"[PROP] Daily anchor initialized at {self.today_anchor_equity:.2f} {self.reset_tz.zone}")
            return

        next_reset = self._next_reset_time(self.last_reset_at)
        if self._now_reset_tz() >= next_reset:
            try:
                self._send_daily_report_for_window(self.last_reset_at, next_reset)
            except Exception as e:
                print(f"[REPORT] Error while sending daily report: {e}")
            self.today_anchor_equity = self.get_mt5_equity()
            self.last_reset_at = next_reset
            if self.alerter:
                self.alerter.reset_day_flags()
            print(f"[PROP] Daily anchor RESET → {self.today_anchor_equity:.2f} at {self.last_reset_at.isoformat()}")

    @staticmethod
    def get_mt5_equity():
        acc = mt5.account_info()
        if acc is None:
            raise RuntimeError("[PROP] Account info unavailable")
        return float(acc.equity)

    @staticmethod
    def get_open_pnl():
        pos_list = mt5.positions_get()
        if not pos_list:
            return 0.0
        return sum(float(p.profit) for p in pos_list)

    def today_closed_pnl(self):
        if self.last_reset_at is None:
            return 0.0
        start_utc = self.last_reset_at.astimezone(UTC).replace(tzinfo=None)
        end_utc   = datetime.now(UTC).replace(tzinfo=None)
        deals = mt5.history_deals_get(start_utc, end_utc)
        if deals is None:
            return 0.0
        pnl = 0.0
        for d in deals:
            pnl += float(d.profit)
        return pnl

    def _send_daily_report_for_window(self, start_tz_dt, end_tz_dt):
        try:
            start_utc = start_tz_dt.astimezone(UTC).replace(tzinfo=None)
            end_utc   = end_tz_dt.astimezone(UTC).replace(tzinfo=None)
            deals = mt5.history_deals_get(start_utc, end_utc)

            cols = ['time','ticket','position_id','order','symbol','side','entry','volume','price',
                    'profit','commission','swap','magic','comment']
            rows = []
            closed_realized = 0.0
            if deals:
                for d in deals:
                    side  = DEAL_TYPE_MAP.get(getattr(d,'type',None), str(getattr(d,'type','')))
                    entry = ENTRY_MAP.get(getattr(d,'entry',None), str(getattr(d,'entry','')))
                    tstr  = _fmt_dt_utc_to_tz(getattr(d,'time',0), self.reset_tz)
                    profit= float(getattr(d,'profit',0.0))
                    if getattr(d,'entry',None) in (getattr(mt5,"DEAL_ENTRY_OUT",1), getattr(mt5,"DEAL_ENTRY_OUT_BY",3)):
                        closed_realized += profit
                    rows.append({
                        'time': tstr, 'ticket': getattr(d,'ticket',''), 'position_id': getattr(d,'position_id',''),
                        'order': getattr(d,'order',''), 'symbol': getattr(d,'symbol',''), 'side': side,
                        'entry': entry, 'volume': getattr(d,'volume',0.0), 'price': getattr(d,'price',0.0),
                        'profit': profit, 'commission': getattr(d,'commission',0.0), 'swap': getattr(d,'swap',0.0),
                        'magic': getattr(d,'magic',0), 'comment': getattr(d,'comment',''),
                    })

            df = pd.DataFrame(rows, columns=cols)

            open_pnl_now = self.get_open_pnl()
            daily_sum    = closed_realized + open_pnl_now
            daily_loss_amount = -daily_sum
            limit        = self.daily_limit
            remaining    = max(0.0, limit - max(0.0, daily_loss_amount))

            start_equity = None
            if self.last_reset_at and abs((start_tz_dt - self.last_reset_at).total_seconds()) < 120:
                start_equity = float(self.today_anchor_equity) if self.today_anchor_equity is not None else None

            end_equity = self.get_mt5_equity()
            eq_change  = (end_equity - start_equity) if start_equity is not None else None

            day_str  = start_tz_dt.strftime('%Y%m%d')
            out_path = os.path.join(REPORT_DIR, f'FundedNext_daily_{day_str}.csv')
            df.to_csv(out_path, index=False)
            print(f"[REPORT] Saved daily report → {out_path} "
                  f"(closed_realized=${closed_realized:,.2f}, open=${open_pnl_now:,.2f}, daily_sum=${daily_sum:,.2f})")

            if alerts_enabled := True:
                summary_lines = [
                    f"Window: {start_tz_dt.strftime('%Y-%m-%d %H:%M %Z')} → {end_tz_dt.strftime('%Y-%m-%d %H:%M %Z')}",
                    f"Closed PnL today:  ${closed_realized:,.2f}",
                    f"Open PnL now:      ${open_pnl_now:,.2f}",
                    f"Daily sum (C+O):   ${daily_sum:,.2f}",
                    f"Max Daily Loss:    ${limit:,.2f}",
                    f"Remaining today:   ${remaining:,.2f}",
                    (f"Start equity:       ${start_equity:,.2f}" if start_equity is not None else "Start equity:       n/a"),
                    f"End equity now:    ${end_equity:,.2f}",
                ]
                if eq_change is not None:
                    summary_lines.append(f"Change (end-start): ${eq_change:,.2f}")
                summary = "\n".join(summary_lines) + "\n"
                alerter.daily_report(out_path, daily_sum, start_tz_dt.strftime('%Y-%m-%d'), summary=summary)
        except Exception as e:
            print(f"[REPORT] Failed to build/send daily report: {e}")

    def current_daily_loss(self):
        self.ensure_daily_anchor()
        closed = self.today_closed_pnl()
        openp  = self.get_open_pnl()
        return -(closed + openp), closed, openp

    def remaining_daily_risk(self):
        loss, closed, openp = self.current_daily_loss()
        remaining = self.daily_limit - max(0.0, loss)
        return max(0.0, remaining), loss, closed, openp

    def breached_daily(self):
        remaining, loss, _, _ = self.remaining_daily_risk()
        return remaining <= 0.0, loss

    def breached_overall(self):
        eq = self.get_mt5_equity()
        min_equity = self.initial - self.overall_limit
        return eq < min_equity, eq, min_equity

    def profit_target_hit(self):
        eq = self.get_mt5_equity()
        gain = eq - self.initial
        target = self.phase1_target if self.phase == 1 else self.phase2_target
        return gain >= target, gain, target

    def would_breach_with_order(self, stop_loss_risk_usd: float) -> bool:
        remaining_daily, _, _, _ = self.remaining_daily_risk()
        if stop_loss_risk_usd > remaining_daily:
            print(f"[PROP] Order veto: SL risk ${stop_loss_risk_usd:.2f} > remaining daily ${remaining_daily:.2f}")
            return True
        eq = self.get_mt5_equity()
        if (eq - stop_loss_risk_usd) < (self.initial - self.overall_limit):
            print("[PROP] Order veto: worst-case SL would breach overall max loss")
            return True
        return False

    def enforce_breaches(self):
        self.ensure_daily_anchor()
        daily_breached, loss = self.breached_daily()
        _, closed, openp     = self.current_daily_loss()
        overall_breached, eq, min_eq = self.breached_overall()

        breached = False
        if daily_breached:
            print(f"[BREACH] Max Daily Loss hit. Current daily loss ≈ ${loss:.2f}. No new trades.")
            breached = True
            if alerter and ALERTS_ENABLED and self.last_reset_at:
                day_key = self.last_reset_at.strftime('%Y-%m-%d')
                alerter.breach_daily(day_key, closed, openp, self.daily_limit, self.reset_tz.zone)

        if overall_breached:
            print(f"[BREACH] Max Overall Loss hit. Equity ${eq:.2f} < minimum ${min_eq:.2f}. No new trades.")
            breached = True
            if alerter and ALERTS_ENABLED:
                alerter.breach_overall(eq, min_eq)

        if breached and self.auto_close:
            pos = mt5.positions_get()
            if pos:
                print("[ACTION] Closing all open positions due to rule breach…")
                for p in pos:
                    _close_position_ticket_generic(p)
        return breached

alerter = Alerter()
prop = PropRules(
    initial_account_size=ACCOUNT_SIZE_USD,
    daily_loss_pct=MAX_DAILY_LOSS_PCT,
    overall_loss_pct=MAX_OVERALL_LOSS_PCT,
    phase1_target_pct=PHASE1_TARGET_PCT,
    phase2_target_pct=PHASE2_TARGET_PCT,
    reset_tz=PROP_RESET_TZ,
    reset_hour=PROP_RESET_HOUR,
    reset_minute=PROP_RESET_MIN,
    phase=CURRENT_PHASE,
    auto_close_on_breach=AUTO_CLOSE_ON_BREACH,
    alerter=alerter,
)

# =====================================================================================
# MT5 CONNECT
# =====================================================================================

print("MT5 Path Exists?", os.path.exists(MT5_TERMINAL_PATH))
if not mt5.initialize(login=int(MT5_LOGIN) if str(MT5_LOGIN).isdigit() else None,
                      password=MT5_PASSWORD,
                      server=MT5_SERVER,
                      path=MT5_TERMINAL_PATH):
    print("[ERROR] MT5 initialization failed:", mt5.last_error())
    sys.exit(1)

account = mt5.account_info()
if account is None:
    raise RuntimeError(f"Failed to retrieve account info: {mt5.last_error()}\n")

print(f"\nMT5 ACCOUNT CONNECTED!\nACCOUNT: {account.login}\nBALANCE: ${account.balance:.2f}\n")

# =====================================================================================
# OANDA FETCH
# =====================================================================================

def fetch_oanda_candles(oanda_symbol: str, granularity="M5", count=NUM_CANDLES) -> Optional[pd.DataFrame]:
    """Fetch candles from OANDA REST API. Returns DataFrame indexed by local time."""
    url = f"{OANDA_BASE_URL}/instruments/{oanda_symbol}/candles"
    headers = {"Authorization": f"Bearer {OANDA_TOKEN}"}
    params = {"granularity": granularity, "count": count, "price": "M"}  # Mid prices
    try:
        r = requests.get(url, headers=headers, params=params, timeout=(5, 15))
    except requests.RequestException as e:
        print(f"[ERROR] OANDA network error: {e.__class__.__name__}: {e}")
        return None
    if r.status_code != 200:
        print("[ERROR] OANDA fetch failed:", r.status_code, r.text[:300])
        return None

    raw = r.json().get("candles", [])
    data = {"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
    for c in raw:
        if c.get("complete", False):
            utc_time = pd.to_datetime(c["time"], utc=True)
            local_time = utc_time.tz_convert(local_tz)
            data["time"].append(local_time)
            data["open"].append(float(c["mid"]["o"]))
            data["high"].append(float(c["mid"]["h"]))
            data["low"].append(float(c["mid"]["l"]))
            data["close"].append(float(c["mid"]["c"]))
            data["volume"].append(int(c["volume"]))
    if not data["time"]:
        return None
    return pd.DataFrame(data).set_index("time")

# =====================================================================================
# STRATEGY: HEIKIN-ASHI + CHANDELIER
# =====================================================================================

def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_df = pd.DataFrame(index=df.index)
    ha_df["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_df["ha_close"].iloc[i-1]) / 2)
    ha_df["ha_open"] = ha_open
    ha_df["ha_high"] = pd.concat([df["high"], ha_df["ha_open"], ha_df["ha_close"]], axis=1).max(axis=1)
    ha_df["ha_low"]  = pd.concat([df["low"],  ha_df["ha_open"], ha_df["ha_close"]], axis=1).min(axis=1)
    return ha_df

def calculate_indicators(df: pd.DataFrame, useHA=True, atrPeriod=ATR_PERIOD, atrMult=ATR_MULT) -> pd.DataFrame:
    if useHA:
        ha = calculate_heikin_ashi(df)
        o = ha["ha_open"]; h = ha["ha_high"]; l = ha["ha_low"]; c = ha["ha_close"]
    else:
        o = df["open"]; h = df["high"]; l = df["low"]; c = df["close"]

    tr = pd.DataFrame(index=df.index)
    tr["o"] = o; tr["h"] = h; tr["l"] = l; tr["c"] = c
    tr["c_prev"] = tr["c"].shift(1)

    def _tr(row):
        if pd.isna(row["c_prev"]):
            return row["h"] - row["l"]
        return max(row["h"] - row["l"], abs(row["h"] - row["c_prev"]), abs(row["l"] - row["c_prev"]))
    tr["true_range"] = tr.apply(_tr, axis=1)

    n = int(max(1, atrPeriod))
    if n == 1:
        tr["atr"] = tr["true_range"]
    else:
        vals = tr["true_range"].to_numpy()
        rma = [None] * len(vals)
        if len(vals) >= n:
            sma_seed = float(pd.Series(vals[:n]).mean())
            rma[n-1] = sma_seed
            alpha = 1.0 / n
            for i in range(n, len(vals)):
                rma[i] = rma[i-1] + alpha * (vals[i] - rma[i-1])
        tr["atr"] = pd.Series(rma, index=tr.index).ffill().bfill()

    atr_val = atrMult * tr["atr"]
    hh = tr["h"].rolling(window=n, min_periods=n).max()
    ll = tr["l"].rolling(window=n, min_periods=n).min()
    long_stop  = hh - atr_val
    short_stop = ll + atr_val

    lss = long_stop.copy()
    sss = short_stop.copy()
    for i in range(len(tr)):
        if i == 0: continue
        long_prev  = lss.iloc[i-1] if pd.notna(lss.iloc[i-1]) else long_stop.iloc[i]
        short_prev = sss.iloc[i-1] if pd.notna(sss.iloc[i-1]) else short_stop.iloc[i]
        if tr["c"].iloc[i-1] > long_prev:
            lss.iloc[i] = max(long_stop.iloc[i], long_prev)
        else:
            lss.iloc[i] = long_stop.iloc[i]
        if tr["c"].iloc[i-1] < short_prev:
            sss.iloc[i] = min(short_stop.iloc[i], short_prev)
        else:
            sss.iloc[i] = short_stop.iloc[i]

    dir_vals = [1]
    for i in range(1, len(tr)):
        if tr["c"].iloc[i] > sss.iloc[i-1]:
            dir_vals.append(1)
        elif tr["c"].iloc[i] < lss.iloc[i-1]:
            dir_vals.append(-1)
        else:
            dir_vals.append(dir_vals[-1])

    tr["dir"] = dir_vals
    tr["dir_prev"] = tr["dir"].shift(1)
    tr["buy_signal"]  = (tr["dir"] ==  1) & (tr["dir_prev"] == -1)
    tr["sell_signal"] = (tr["dir"] == -1) & (tr["dir_prev"] ==  1)

    tr["long_stop_smooth"]  = lss
    tr["short_stop_smooth"] = sss
    tr["ha_open"] = o
    tr["ha_high"] = h
    tr["ha_low"]  = l
    tr["ha_c"]    = c
    return tr

# =====================================================================================
# ASSET STATE + HELPERS
# =====================================================================================

@dataclass
class AssetState:
    last_candle_time: Optional[pd.Timestamp] = None
    current_session_start: Optional[datetime] = None
    current_session_end: Optional[datetime] = None
    saw_candle_after_session_start: bool = False
    # pending execution
    pending_signal: Optional[str] = None
    pending_since: Optional[datetime] = None
    last_retry_at: Optional[datetime] = None
    # per-position management: ticket -> flags
    pos_state: Dict[int, Dict[str, bool]] = field(default_factory=dict)
    # next time to check for a bar close
    next_bar_close: Optional[datetime] = None

RETRY_EVERY_SECS = 15

def _round_to_step(value, step):
    steps = math.floor(value / step)
    return max(steps * step, 0.0)

def _make_dt(base_dt, hhmm, tz):
    h, m = map(int, hhmm.split(":"))
    return base_dt.astimezone(tz).replace(hour=h, minute=m, second=0, microsecond=0)

def in_session(now_local: datetime, session_windows: List[Tuple[str,str]]):
    for start, end in session_windows:
        start_dt = _make_dt(now_local, start, local_tz)
        end_dt   = _make_dt(now_local, end,   local_tz)
        if start_dt <= now_local < end_dt:
            return True
    return False

def current_session_bounds(now_local: datetime, session_windows: List[Tuple[str,str]]):
    for start, end in session_windows:
        start_dt = _make_dt(now_local, start, local_tz)
        end_dt   = _make_dt(now_local, end,   local_tz)
        if start_dt <= now_local < end_dt:
            return start_dt, end_dt
    return None, None

def next_5m_close(now: datetime):
    base = now.replace(second=0, microsecond=0)
    mins_to_add = (5 - (base.minute % 5)) % 5
    target = base + timedelta(minutes=mins_to_add)
    if target <= now:
        target += timedelta(minutes=5)
    return target

# =====================================================================================
# POSITION / ORDER HELPERS (per-asset)
# =====================================================================================

def _get_position(cfg: AssetConfig):
    positions = mt5.positions_get(symbol=cfg.mt5_symbol)
    if not positions:
        return None
    for p in positions:
        if p.magic == cfg.magic_number:
            return p
    return None

def _close_position_ticket_generic(position):
    """Close an MT5 position (generic — used for breach auto-close)."""
    symbol = position.symbol
    action_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
    if tick is None: return
    price = tick.bid if action_type == mt5.ORDER_TYPE_SELL else tick.ask
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": action_type,
        "position": position.ticket,
        "price": price,
        "deviation": SLIPPAGE,
        "magic": getattr(position, "magic", 0),
        "comment": "PropRules Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Failed to close position: {getattr(res, 'retcode', None)}, {getattr(res, 'comment', None)}")
    else:
        print(f"[OK] POSITION CLOSED: {res}")

def _close_position_ticket(cfg: AssetConfig, position):
    symbol = cfg.mt5_symbol
    action_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
    if tick is None: return
    price = tick.bid if action_type == mt5.ORDER_TYPE_SELL else tick.ask
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": action_type,
        "position": position.ticket,
        "price": price,
        "deviation": SLIPPAGE,
        "magic": cfg.magic_number,
        "comment": f"PropRules Close {cfg.name}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[{cfg.name}] [ERROR] Failed to close position: {getattr(res, 'retcode', None)}, {getattr(res, 'comment', None)}")
    else:
        print(f"[{cfg.name}] [OK] POSITION CLOSED: {res}")

def compute_lot_for_risk_static_initial(cfg: AssetConfig, sl_distance_price_units: float) -> Tuple[Optional[float], float]:
    si = mt5.symbol_info(cfg.mt5_symbol)
    if si is None:
        print(f"[{cfg.name}] [ERROR] Symbol info not found")
        return None, 0.0

    risk_amount = float(ACCOUNT_SIZE_USD) * (float(cfg.risk_pct_initial) / 100.0)
    if risk_amount <= 0:
        return None, 0.0

    value_per_1unit_per_lot = None
    tv = getattr(si, "trade_tick_value", None)
    ts = getattr(si, "trade_tick_size", None)
    if tv not in (None, 0) and ts not in (None, 0):
        value_per_1unit_per_lot = float(tv) / float(ts)
    if value_per_1unit_per_lot is None:
        tv2 = getattr(si, "tick_value", None)
        ts2 = getattr(si, "tick_size", None)
        if tv2 not in (None, 0) and ts2 not in (None, 0):
            value_per_1unit_per_lot = float(tv2) / float(ts2)
    if value_per_1unit_per_lot is None:
        tv3 = getattr(si, "tick_value", None)
        pt  = getattr(si, "point", None)
        if tv3 not in (None, 0) and pt not in (None, 0):
            value_per_1unit_per_lot = float(tv3) / float(pt)
    if value_per_1unit_per_lot is None:
        cs = getattr(si, "trade_contract_size", None)
        if cs not in (None, 0):
            value_per_1unit_per_lot = float(cs)

    if value_per_1unit_per_lot in (None, 0):
        print(f"[{cfg.name}] [ERROR] Could not derive per-unit value for 1 lot.")
        return None, 0.0

    risk_per_lot = value_per_1unit_per_lot * float(sl_distance_price_units)
    if risk_per_lot <= 0:
        return None, 0.0

    raw_lot = risk_amount / risk_per_lot
    lot = _round_to_step(raw_lot, si.volume_step)
    lot = max(si.volume_min, min(lot, si.volume_max))
    return lot, risk_amount

def compute_sl_price(cfg: AssetConfig, action_type, entry_price, sl_usd):
    si = mt5.symbol_info(cfg.mt5_symbol)
    if si is None:
        return None
    digits = si.digits
    point  = si.point
    if action_type == mt5.ORDER_TYPE_BUY:
        sl = entry_price - float(sl_usd)
    else:
        sl = entry_price + float(sl_usd)

    stops_pts = getattr(si, "trade_stops_level", 0)
    if stops_pts and stops_pts > 0:
        min_dist = stops_pts * point
        diff = abs(entry_price - sl)
        if diff < min_dist:
            sl = entry_price - min_dist if action_type == mt5.ORDER_TYPE_BUY else entry_price + min_dist
    return round(sl, digits)

def compute_tp_price(cfg: AssetConfig, action_type, entry_price, tp_usd):
    si = mt5.symbol_info(cfg.mt5_symbol)
    if si is None:
        return None
    digits = si.digits
    if action_type == mt5.ORDER_TYPE_BUY:
        tp = entry_price + float(tp_usd)
    else:
        tp = entry_price - float(tp_usd)
    return round(tp, digits)

def _round_volume_to_step(si, vol):
    vol = math.floor(vol / si.volume_step) * si.volume_step
    return max(si.volume_min, min(vol, si.volume_max))

def _partial_close(cfg: AssetConfig, position, fraction: float) -> bool:
    si = mt5.symbol_info(cfg.mt5_symbol)
    if si is None:
        return False
    part_vol_raw = position.volume * float(fraction)
    part_vol = _round_volume_to_step(si, part_vol_raw)

    remain_vol = round(position.volume - part_vol, 8)
    if remain_vol > 0 and remain_vol < si.volume_min:
        part_vol = _round_volume_to_step(si, position.volume - si.volume_min)
        remain_vol = round(position.volume - part_vol, 8)

    if part_vol <= 0:
        return False

    opp_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(cfg.mt5_symbol)
    if tick is None:
        return False
    close_price = tick.bid if opp_type == mt5.ORDER_TYPE_SELL else tick.ask

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": cfg.mt5_symbol,
        "volume": part_vol,
        "type": opp_type,
        "position": position.ticket,
        "price": close_price,
        "deviation": SLIPPAGE,
        "magic": cfg.magic_number,
        "comment": f"PartialTP {cfg.name}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[{cfg.name}] [OK] Partial closed {part_vol} from ticket {position.ticket}")
        return True
    print(f"[{cfg.name}] [WARN] Partial close failed: {getattr(res,'retcode',None)} {getattr(res,'comment',None)}")
    return False

def _move_sl_to_breakeven(cfg: AssetConfig, position, buffer_usd=0.0) -> bool:
    si = mt5.symbol_info(cfg.mt5_symbol)
    tick = mt5.symbol_info_tick(cfg.mt5_symbol)
    if si is None or tick is None:
        return False

    entry = float(position.price_open)
    point = si.point
    stops_pts = getattr(si, "trade_stops_level", 0) or 0
    min_dist = stops_pts * point

    spread = (tick.ask - tick.bid) if cfg.use_spread_for_be_buffer else 0.0
    be_buf = max(float(buffer_usd), spread)

    if position.type == mt5.ORDER_TYPE_BUY:
        desired_sl = round(entry + be_buf, si.digits)
        max_allowed_sl = tick.bid - min_dist if min_dist > 0 else tick.bid
        if desired_sl <= max_allowed_sl and desired_sl < tick.bid:
            mod = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": cfg.mt5_symbol,
                "position": position.ticket,
                "sl": desired_sl,
                "magic": cfg.magic_number,
                "comment": f"SL->BE {cfg.name}",
            }
            res = mt5.order_send(mod)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[{cfg.name}] [OK] SL moved to BE+buffer at {desired_sl} (buy).")
                return True
            print(f"[{cfg.name}] [WARN] BE SL set failed: {getattr(res,'retcode',None)} {getattr(res,'comment',None)}")
            return False
        return False
    else:
        desired_sl = round(entry - be_buf, si.digits)
        min_allowed_sl = tick.ask + min_dist if min_dist > 0 else tick.ask
        if desired_sl >= min_allowed_sl and desired_sl > tick.ask:
            mod = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": cfg.mt5_symbol,
                "position": position.ticket,
                "sl": desired_sl,
                "magic": cfg.magic_number,
                "comment": f"SL->BE {cfg.name}",
            }
            res = mt5.order_send(mod)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[{cfg.name}] [OK] SL moved to BE+buffer at {desired_sl} (sell).")
                return True
            print(f"[{cfg.name}] [WARN] BE SL set failed: {getattr(res,'retcode',None)} {getattr(res,'comment',None)}")
            return False
        return False

def manage_open_position(cfg: AssetConfig, state: AssetState):
    """
    Per-asset management (multi-asset backbone):
      1) Take tp1_fraction at tp1_usd_distance (e.g., 50% at $X)
      2) AFTER (and only after) tp1 is booked, move SL to BE+buffer (respect broker min distance)
      3) Take tp2_fraction at tp2_usd_distance (e.g., 25% at $Y)
      4) Leave the remainder; opposite signal elsewhere handles exit/flip
    """
    if prop.enforce_breaches():
        return

    position = _get_position(cfg)
    if not position:
        if state.pos_state:
            state.pos_state.clear()
        return

    tick = mt5.symbol_info_tick(cfg.mt5_symbol)
    si   = mt5.symbol_info(cfg.mt5_symbol)
    if tick is None or si is None:
        return

    ticket = position.ticket
    flags = state.pos_state.get(ticket, {"partial50_done": False, "partial25_done": False, "moved_to_be": False})

    # Pre-compute absolute TP prices for this entry
    tp1_price = compute_tp_price(cfg, position.type, position.price_open, cfg.tp1_usd_distance)
    tp2_price = compute_tp_price(cfg, position.type, position.price_open, cfg.tp2_usd_distance)
    if tp1_price is None or tp2_price is None:
        return

    # Helper: "has price reached target?" using executable-side price
    def _touched(target_price, pos_type):
        if pos_type == mt5.ORDER_TYPE_BUY:
            return tick.bid >= target_price  # close buy at bid
        else:
            return tick.ask <= target_price  # close sell at ask

    tp1_hit = _touched(tp1_price, position.type)
    tp2_hit = _touched(tp2_price, position.type)

    # --- 1) TP1 (e.g., 50%) ---
    if not flags["partial50_done"] and tp1_hit:
        ok = _partial_close(cfg, position, cfg.tp1_fraction)
        if ok:
            flags["partial50_done"] = True
            state.pos_state[ticket] = flags
            # refresh handles changed volume
            time.sleep(0.3)
            position = _get_position(cfg)
            if not position:
                return
            tick = mt5.symbol_info_tick(cfg.mt5_symbol)
            si   = mt5.symbol_info(cfg.mt5_symbol)
            if tick is None or si is None:
                return

    # --- 2) Move SL to BE ONLY AFTER TP1 booked (retry until allowed) ---
    if flags["partial50_done"] and not flags["moved_to_be"]:
        if _move_sl_to_breakeven(cfg, position, cfg.breakeven_buffer_usd):
            flags["moved_to_be"] = True
            state.pos_state[ticket] = flags
        # If broker min distance blocks it now, we try again next loop.

    # --- 3) TP2 (e.g., 25%) ---
    # If market jumped to TP2 before TP1, we still did TP1 above first (now flags reflect that),
    # and we can proceed to TP2 if it's touched.
    if tp2_hit and not flags["partial25_done"]:
        # refresh again to be safe before partial2
        position = _get_position(cfg)
        if not position:
            return
        tick = mt5.symbol_info_tick(cfg.mt5_symbol)
        si   = mt5.symbol_info(cfg.mt5_symbol)
        if tick is None or si is None:
            return

        ok2 = _partial_close(cfg, position, cfg.tp2_fraction)
        if ok2:
            flags["partial25_done"] = True
            state.pos_state[ticket] = flags

    # Runner remains; the next opposite signal logic elsewhere will flip/close it.


def _filling_modes():
    modes = []
    if hasattr(mt5, "ORDER_FILLING_RETURN"): modes.append(mt5.ORDER_FILLING_RETURN)
    if hasattr(mt5, "ORDER_FILLING_FOK"):    modes.append(mt5.ORDER_FILLING_FOK)
    if hasattr(mt5, "ORDER_FILLING_IOC"):    modes.append(mt5.ORDER_FILLING_IOC)
    return modes or [mt5.ORDER_FILLING_IOC]

def send_order(cfg: AssetConfig, action_type) -> bool:
    if prop.enforce_breaches():
        print(f"[{cfg.name}] [BLOCK] Prop rule breached — order blocked.")
        return False

    if not mt5.symbol_select(cfg.mt5_symbol, True):
        print(f"[{cfg.name}] [ERROR] Failed to select symbol")
        return False
    si = mt5.symbol_info(cfg.mt5_symbol)
    if si is None:
        print(f"[{cfg.name}] [ERROR] Symbol info not found")
        return False
    if hasattr(si, "trade_allowed") and not si.trade_allowed:
        print(f"[{cfg.name}] [ERROR] Trading not allowed")
        return False

    tick = mt5.symbol_info_tick(cfg.mt5_symbol)
    if tick is None:
        print(f"[{cfg.name}] [ERROR] No tick data")
        return False
    entry_price = tick.ask if action_type == mt5.ORDER_TYPE_BUY else tick.bid

    sl_snapshot = compute_sl_price(cfg, action_type, entry_price, cfg.sl_usd_distance)
    if sl_snapshot is None:
        print(f"[{cfg.name}] [ERROR] Could not compute SL.")
        return False
    sl_dist = abs(entry_price - sl_snapshot)

    lot, risk_amount = compute_lot_for_risk_static_initial(cfg, sl_dist)
    if lot is None or lot <= 0:
        print(f"[{cfg.name}] [ERROR] Invalid lot; abort.")
        return False

    # Pre-trade oversight against prop breaches (approximate)
    v_per_1unit = None
    if getattr(si, 'trade_tick_value', 0) and getattr(si, 'trade_tick_size', 0):
        v_per_1unit = float(si.trade_tick_value) / float(si.trade_tick_size)
    elif getattr(si, 'tick_value', 0) and getattr(si, 'tick_size', 0):
        v_per_1unit = float(si.tick_value) / float(si.tick_size)
    elif getattr(si, 'tick_value', 0) and getattr(si, 'point', 0):
        v_per_1unit = float(si.tick_value) / float(si.point)
    elif getattr(si, 'trade_contract_size', 0):
        v_per_1unit = float(si.trade_contract_size)

    if v_per_1unit:
        worst_case_loss = v_per_1unit * sl_dist * float(lot)
        if prop.would_breach_with_order(worst_case_loss):
            print(f"[{cfg.name}] [BLOCK] Order would breach rules — aborted.")
            return False
    else:
        print(f"[{cfg.name}] [WARN] Could not derive per-unit value; skipping pre-trade check.")

    base_req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": cfg.mt5_symbol,
        "volume": lot,
        "type": action_type,
        "price": entry_price,
        "deviation": SLIPPAGE,
        "magic": cfg.magic_number,
        "comment": f"CEBot-Prop {cfg.name}",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    last_res = None
    for fm in _filling_modes():
        req = dict(base_req, type_filling=fm)
        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[{cfg.name}] [OK] ORDER PLACED (fill_mode={fm}): ticket={res.order}, price={entry_price}, lot={lot:.2f}")
            pos = _get_position(cfg)
            if pos:
                filled = pos.price_open
                desired_sl = compute_sl_price(cfg, action_type, filled, cfg.sl_usd_distance)
                if desired_sl:
                    mod = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": cfg.mt5_symbol,
                        "position": pos.ticket,
                        "sl": desired_sl,
                        "magic": cfg.magic_number,
                        "comment": f"Set SL {cfg.name}",
                    }
                    mod_res = mt5.order_send(mod)
                    if mod_res is not None and mod_res.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"[{cfg.name}] [OK] SL SET to {desired_sl}")
                    else:
                        print(f"[{cfg.name}] [WARN] SL set failed: {getattr(mod_res,'retcode',None)} {getattr(mod_res,'comment',None)}")
            else:
                print(f"[{cfg.name}] [WARN] Position not found after fill; cannot set SL.")
            return True
        last_res = res

    print(f"[{cfg.name}] [ERROR] ORDER FAILED across all filling modes.")
    if last_res:
        print(f"          retcode={last_res.retcode}, comment={last_res.comment}")
    else:
        print(f"          order_send returned None. last_error={mt5.last_error()}")
    return False

def attempt_execution_for_signal(cfg: AssetConfig, state: AssetState, desired_side: str) -> bool:
    if prop.enforce_breaches():
        return False
    hit, _, _ = prop.profit_target_hit()
    if hit:
        return False

    position = _get_position(cfg)
    open_side = None
    if position:
        open_side = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"

    if open_side == desired_side:
        return True

    if open_side and open_side != desired_side:
        _close_position_ticket(cfg, position)
        time.sleep(0.5)

    order_type = mt5.ORDER_TYPE_BUY if desired_side == "BUY" else mt5.ORDER_TYPE_SELL
    return bool(send_order(cfg, order_type))

def maybe_retry_pending(cfg: AssetConfig, state: AssetState):
    if not state.pending_signal:
        return
    hit, _, _ = prop.profit_target_hit()
    if hit:
        return
    now_local = datetime.now(local_tz)
    if not in_session(now_local, cfg.session_windows):
        return
    if state.last_retry_at and (now_local - state.last_retry_at).total_seconds() < RETRY_EVERY_SECS:
        return
    if prop.enforce_breaches():
        return

    print(f"[{cfg.name}] [RETRY] Pending {state.pending_signal} — attempting execution…")
    ok = attempt_execution_for_signal(cfg, state, state.pending_signal)
    state.last_retry_at = now_local
    if ok:
        print(f"[{cfg.name}] [OK] Pending {state.pending_signal} executed.")
        state.pending_signal = None
        state.pending_since  = None
        state.last_retry_at  = None

# =====================================================================================
# ONE-SHOT REPORT (optional)
# =====================================================================================

def force_daily_report_now():
    try:
        now_tz  = datetime.now(PROP_RESET_TZ)
        start_tz = now_tz - timedelta(days=1)
        print(f"[REPORT] Forcing daily report for: {start_tz.strftime('%Y-%m-%d %H:%M %Z')} → {now_tz.strftime('%Y-%m-%d %H:%M %Z')}")
        prop._send_daily_report_for_window(start_tz, now_tz)
    except Exception as e:
        print(f"[REPORT] Startup forced daily report failed: {e}")

# =====================================================================================
# MAIN SCHEDULER (single-threaded cooperative loop trading ALL assets)
# =====================================================================================

def print_daily_risk_diag():
    try:
        rem, loss, closed, openp = prop.remaining_daily_risk()
        log(f"[DIAG] Remaining daily risk: ${rem:,.2f} | Daily loss: ${loss:,.2f} (closed=${closed:,.2f}, open=${openp:,.2f})")
    except Exception as e:
        log(f"[DIAG] Failed to compute daily risk: {e}")

def process_new_bar_for_asset(cfg: AssetConfig, state: AssetState, df: pd.DataFrame):
    # 1) Print raw last 10
    try:
        raw_to_show = df[["open","high","low","close","volume"]].copy()
        raw_to_show.index = raw_to_show.index.strftime("%Y-%m-%d %H:%M")
        # print(f"\n= = = = =   RAW OANDA ({cfg.name}) LAST 10 CANDLES   = = = = =")
        # print(raw_to_show.tail(10))
    except Exception as e:
        log(f"[{cfg.name}] [DEBUG] Raw print failed: {e}")

    # 2) HA + Chandelier
    tr = calculate_indicators(df, useHA=USE_HEIKIN_ASHI, atrPeriod=ATR_PERIOD, atrMult=ATR_MULT)
    latest = tr.iloc[-1]
    try:
        debug_df = tr[["ha_c","ha_open","ha_high","ha_low","dir","buy_signal","sell_signal"]].copy()
        debug_df["signal"] = debug_df.apply(lambda r: "BUY" if r["buy_signal"] else ("SELL" if r["sell_signal"] else ""), axis=1)
        debug_df.index = debug_df.index.strftime("%Y-%m-%d %H:%M")
        # print(f"\n= = = = =   ({cfg.name}) HA CANDLES + SIGNALS (LAST 10)  = = = = =")
        # print(debug_df[["ha_c","ha_open","ha_high","ha_low","dir","signal"]].tail(10))
    except Exception as e:
        log(f"[{cfg.name}] [DEBUG] HA table print failed: {e}")

    # 3) Signals
    signal = "BUY" if bool(latest["buy_signal"]) else ("SELL" if bool(latest["sell_signal"]) else None)

    # 4) Position state
    position = _get_position(cfg)
    open_side = None

    if position:
        open_side = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
        print(f"[{cfg.name}] [INFO] OPEN POSITION: {open_side}, vol: {position.volume}, entry: {position.price_open}")
        if position.ticket not in state.pos_state:
            tp1_preview = compute_tp_price(cfg, position.type, position.price_open, cfg.tp1_usd_distance)
            tp2_preview = compute_tp_price(cfg, position.type, position.price_open, cfg.tp2_usd_distance)
            print(
                f"[{cfg.name}] [INFO] Managing ticket {position.ticket}: entry={position.price_open}, "
                f"TP1({int(cfg.tp1_fraction * 100)}%)={tp1_preview}, TP2({int(cfg.tp2_fraction * 100)}%)={tp2_preview}")
            state.pos_state[position.ticket] = {"partial50_done": False, "partial25_done": False, "moved_to_be": False}

    else:
        print(f"[{cfg.name}] [INFO] No open position currently.")

    # 5) Profit target gate
    hit, gain, target = prop.profit_target_hit()
    if hit:
        print(f"[TARGET] Phase {prop.phase} reached: Gain ${gain:.2f} ≥ ${target:.2f}. Halt new trades.")
        if ALERTS_ENABLED and alerter:
            alerter.phase_passed(prop.phase, gain, target)
        state.pending_signal = None
        state.pending_since  = None
        state.last_retry_at  = None
        return

    # 6) Session end guard (handled outside too)
    now_local = datetime.now(local_tz)
    if state.current_session_end and now_local >= state.current_session_end:
        print(f"[{cfg.name}] [INFO] SESSION ENDED POST-CALC.")
        if AUTO_CLOSE_AT_SESSION_END and position:
            _close_position_ticket(cfg, position)
        state.pending_signal = None
        state.pending_since  = None
        state.last_retry_at  = None
        return

    # 7) Trade execution / backfill
    if signal:
        if open_side != signal:
            print_daily_risk_diag()
            print(f"[{cfg.name}] [TRADE] New signal={signal} | open={open_side or 'NONE'}")
            ok = attempt_execution_for_signal(cfg, state, signal)
            if ok:
                state.pending_signal = None
                state.pending_since  = None
                state.last_retry_at  = None
            else:
                if not state.pending_signal:
                    state.pending_signal = signal
                    state.pending_since  = now_local
                    print(f"[{cfg.name}] [PENDING] Queued {signal} (will retry).")
    else:
        # Backfill previous bar's signal if exists
        if len(tr) >= 2:
            prev = tr.iloc[-2]
            prev_signal = "BUY" if prev["buy_signal"] else ("SELL" if prev["sell_signal"] else None)
            if prev_signal and open_side != prev_signal and not state.pending_signal:
                print(f"[{cfg.name}] [BACKFILL] Previous bar had {prev_signal} — try now.")
                ok = attempt_execution_for_signal(cfg, state, prev_signal)
                if ok:
                    state.pending_signal = None
                    state.pending_since  = None
                    state.last_retry_at  = None
                else:
                    state.pending_signal = prev_signal
                    state.pending_since  = now_local
                    print(f"[{cfg.name}] [PENDING] Queued {prev_signal} from previous bar (will retry).")

def main():
    # boot banner
    print("[BOOT] PropRules engine active.",
          "Phase:", prop.phase,
          "| Targets → P1:", f"${prop.phase1_target:.0f}",
          "P2:", f"${prop.phase2_target:.0f}",
          "| Limits → Daily:", f"${prop.daily_limit:.0f}",
          "Overall:", f"${prop.overall_limit:.0f}")

    # force daily report if wanted
    if FORCE_DAILY_REPORT_ON_START:
        force_daily_report_now()

    # Per-asset runtime state
    states: Dict[str, AssetState] = {cfg.name: AssetState() for cfg in ASSETS}

    try:
        while True:
            if prop.enforce_breaches():
                # Clear all pendings when breached
                for st in states.values():
                    st.pending_signal = None
                    st.pending_since  = None
                    st.last_retry_at  = None
                time.sleep(1)
                continue

            now_local = datetime.now(local_tz)

            for cfg in ASSETS:
                st = states[cfg.name]

                # Update session status
                in_sess = in_session(now_local, cfg.session_windows)
                if in_sess and (st.current_session_start is None or now_local >= st.current_session_end if st.current_session_end else True):
                    # (Re)enter session - compute current bounds
                    st.current_session_start, st.current_session_end = current_session_bounds(now_local, cfg.session_windows)
                    st.saw_candle_after_session_start = False
                    print(f"[{cfg.name}] [+] IN SESSION: "
                          f"{st.current_session_start.strftime('%H:%M:%S %Z')}–{st.current_session_end.strftime('%H:%M:%S %Z')}")

                # If out of session:
                if not in_sess:
                    # session just ended? ensure close if requested
                    if st.current_session_end and now_local >= st.current_session_end:
                        pos = _get_position(cfg)
                        if pos and AUTO_CLOSE_AT_SESSION_END:
                            print(f"[{cfg.name}] [ACTION] Closing open position at session end…")
                            _close_position_ticket(cfg, pos)
                    # reset session bounds when out
                    st.current_session_start = None
                    st.current_session_end   = None
                    st.next_bar_close        = None
                    # still try partial/BE management if any (no new trades)
                    manage_open_position(cfg, st)
                    continue

                # Manage partials/BE intrabar while waiting
                manage_open_position(cfg, st)
                maybe_retry_pending(cfg, st)

                # Determine next 5m close target for this asset
                if st.next_bar_close is None:
                    st.next_bar_close = next_5m_close(now_local)

                # If session ends before next close, clamp to session end
                if st.current_session_end and st.next_bar_close > st.current_session_end:
                    st.next_bar_close = st.current_session_end

                # Wait until close time per asset (cooperative: we tick every 0.25s)
                if now_local < st.next_bar_close:
                    continue  # will revisit next loop tick

                # Tiny grace for OANDA to finalize the bar
                time.sleep(2)

                # If session ended right at the bar close
                now_local = datetime.now(local_tz)
                if st.current_session_end and now_local >= st.current_session_end:
                    # final manage and optional close
                    manage_open_position(cfg, st)
                    pos = _get_position(cfg)
                    if pos and AUTO_CLOSE_AT_SESSION_END:
                        _close_position_ticket(cfg, pos)
                    # prepare for next cycle out of session
                    st.current_session_start = None
                    st.current_session_end   = None
                    st.next_bar_close        = None
                    continue

                # Fetch and process new OANDA bar
                df = fetch_oanda_candles(cfg.oanda_symbol, granularity="M5", count=NUM_CANDLES)
                if df is None or df.empty:
                    print(f"[{cfg.name}] [TIMEOUT] No candle data. Will retry next loop.")
                    # schedule next check at next 5m
                    st.next_bar_close = next_5m_close(datetime.now(local_tz))
                    continue

                latest_candle_time = df.index[-1]

                # Avoid double-processing same bar
                if st.last_candle_time is not None and latest_candle_time <= st.last_candle_time:
                    # already fetched; schedule next bar
                    st.next_bar_close = next_5m_close(datetime.now(local_tz))
                    continue

                st.last_candle_time = latest_candle_time
                print(f"[{cfg.name}] [OK] New 5-min candle: {latest_candle_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

                # Enforce first post-session-start bar if requested
                if REQUIRE_POST_START_CANDLE and not st.saw_candle_after_session_start:
                    if st.current_session_start and st.last_candle_time <= st.current_session_start:
                        print(f"[{cfg.name}] [WAIT] First bar after session open not closed yet; skipping.")
                        st.next_bar_close = next_5m_close(datetime.now(local_tz))
                        continue
                    st.saw_candle_after_session_start = True

                # Process bar (signals + trades)
                process_new_bar_for_asset(cfg, st, df)

                # Schedule next bar close for this asset
                st.next_bar_close = next_5m_close(datetime.now(local_tz))

            # cooperative sleep (engine ticks ~4x per second)
            time.sleep(0.25)

    except KeyboardInterrupt:
        log("[INFO] Stopped by user (CTRL-C).")
    finally:
        try:
            mt5.shutdown()
            log("[INFO] MT5 shutdown complete.")
        except Exception as e:
            log(f"[WARN] MT5 shutdown error: {e}")

# =====================================================================================
# ENTRYPOINT
# =====================================================================================

if __name__ == "__main__":
    main()
