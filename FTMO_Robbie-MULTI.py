"""
Chandelier Exit Strategy — FTMO-Style 2-Step Prop Rules (Profiles: XAU M5 + XAU M15)
Using Heikin-Ashi Candles | Multiple Timeframes | Per-Profile Sessions, Risk, SL/TP/Partials, BE

Key changes:
- Two separate strategy profiles (XAU M5, XAU M15) with distinct sessions, risk, SL/TP/partials, BE buffers, ATR params.
- Risk is % of CURRENT account equity (not fixed initial) — per profile.
- SL attach retry cadence reduced to 5s and profile-scoped (by magic number), keeps retrying until set.
- Move SL to BE at TP1 hit even if no partials are taken (tp1_fraction=None).
- Clean, profile-tagged logging for tidy terminal output.
- Non-overlapping management by using unique magic numbers and profile-aware functions.
"""

import os
import sys
import time
import math
from datetime import datetime, timedelta, time as dtime
import pandas as pd
from pytz import timezone, UTC

import requests
import MetaTrader5 as mt5

from email.message import EmailMessage
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ===================== USER/ENV CONFIG ===================== #
local_tz = timezone('Europe/London')

def log(msg: str):
    ts = datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts} | {msg}", flush=True)

# --- SL retry cadence --- #
SL_RETRY_EVERY_SECS = 5
_last_sl_retry_at = None
_need_sl_seed_scan = False  # when an order is filled but position isn't visible yet (rare)

# legacy pending signal (kept for compatibility; used lightly)
pending_signal = None
pending_since  = None
last_retry_at  = None
RETRY_EVERY_SECS = 15

# per-position state: ticket -> {...}
pos_state = {}
# queued SL attempts: ticket -> {"desired_sl": float, "last_try": datetime, "magic": int}
pending_sl = {}

# === PROP RULES CONFIG === #
phase1_target_pct        = 10.0
phase2_target_pct        = 5.0
max_daily_loss_pct       = 5.0
max_overall_loss_pct     = 10.0

prop_reset_tz            = timezone('Europe/Prague')
prop_reset_hour          = 0
prop_reset_minute        = 0
current_phase            = 1
auto_close_on_breach     = True

# === EMAIL / REPORTS === #
alerts_enabled           = True
email_enabled            = True

report_dir               = r"C:\Users\anish\OneDrive\Desktop\Anish\A - EAs Reports\FTMO-Robbie"

smtp_host = "smtp.gmail.com"
smtp_port = 587
smtp_user = "anishv2610@gmail.com"
smtp_pass = "lpignmmkhgymwlpi"
email_to  = ["ea.prop0925@gmail.com"]

FORCE_DAILY_REPORT_ON_START = False

# === ACCOUNT LOGIN CONFIG === #
mt5_login = 511012068
mt5_password = "H9qb9n8l!4Gzz8"
mt5_server = "FTMO-Server"
mt5_terminal_path = r"C:\MT5\EA-Prop_FTMO\FTMO-Robbie\terminal64.exe"

# === OANDA CONFIG === #
oanda_api_base    = "https://api-fxpractice.oanda.com/v3"
oanda_token       = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"

# === EXECUTION TUNABLES === #
slippage = 10
num_candles = 500
require_post_start_candle = True
auto_close_at_session_end = True

# ===================== DEAL/ENTRY MAPPINGS ===================== #
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
        return datetime.fromtimestamp(int(ts_seconds), tz=UTC).astimezone(tz).strftime('%Y-%m-%d %H:%M:%S %Z')
    except Exception:
        return ""

# ===================== ALERTS / REPORTS ===================== #
class Alerter:
    def __init__(self):
        self._daily_breach_sent_key = None
        self._overall_breach_sent   = False
        self._phase_sent            = set()

    def _send_email(self, subject: str, body: str, attachments: list = None):
        print(f"[ALERT] {subject} | {body.replace(chr(10), ' | ')}")
        if not email_enabled:
            return
        try:
            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From']    = smtp_user or 'bot@localhost'
            msg['To']      = ', '.join(email_to) if email_to else (smtp_user or 'me@localhost')
            msg.set_content(body)
            for fp in attachments or []:
                try:
                    with open(fp, 'rb') as f:
                        data = f.read()
                    msg.add_attachment(data, maintype='application', subtype='octet-stream', filename=os.path.basename(fp))
                except Exception as e:
                    print(f"[ALERT] Attachment failed: {fp} ({e})")
            import smtplib
            with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as s:
                s.starttls()
                if smtp_user:
                    s.login(smtp_user, smtp_pass)
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
        subject = "FTMO BREACH — Max Daily Loss"
        body = (f"*** DAILY BREACH ***\n"
                f"Closed today: ${closed:,.2f}\nOpen PnL: ${openp:,.2f}\n"
                f"Daily sum (C+O): ${closed+openp:,.2f} <= -${limit:,.2f}\n"
                f"Reset TZ: {tzname}\n")
        self._send_email(subject, body)

    def breach_overall(self, equity: float, min_equity: float):
        if self._overall_breach_sent:
            return
        self._overall_breach_sent = True
        subject = "FTMO BREACH — Max Overall Loss"
        body = (f"*** OVERALL BREACH ***\n"
                f"Equity: ${equity:,.2f}\nRequired minimum equity: ${min_equity:,.2f}\n")
        self._send_email(subject, body)

    def phase_passed(self, phase: int, gain: float, target: float):
        if phase in self._phase_sent:
            return
        self._phase_sent.add(phase)
        subject = f"FTMO Phase {phase} PASSED"
        body = (f"Congratulations — Phase {phase} target reached!\n"
                f"Gain: ${gain:,.2f} >= Target: ${target:,.2f}\n")
        self._send_email(subject, body)

    def daily_report(self, csv_path: str, net_pnl: float, date_label: str, summary: str = None):
        subject = f"FTMO PROP CHALLENGE (Robbie) — {date_label}"
        body = (summary or "") + f"\nNet PnL (Closed+Open at send time): ${net_pnl:,.2f}\nFile: {csv_path}\n"
        self._send_email(subject, body, attachments=[csv_path] if csv_path else None)

alerter = Alerter()

# ===================== PROP RULES ===================== #
class PropRules:
    def __init__(self, daily_loss_pct: float, overall_loss_pct: float,
                 phase1_target_pct: float, phase2_target_pct: float, reset_tz: timezone,
                 reset_hour: int, reset_minute: int, phase: int = 1,
                 auto_close_on_breach: bool = True, alerter: 'Alerter' = None):
        self.daily_loss_pct = float(daily_loss_pct)
        self.overall_loss_pct = float(overall_loss_pct)
        self.phase1_target_pct = float(phase1_target_pct)
        self.phase2_target_pct = float(phase2_target_pct)
        self.phase = 1 if phase == 1 else 2
        self.reset_tz = reset_tz
        self.reset_hour = reset_hour
        self.reset_minute = reset_minute
        self.auto_close = auto_close_on_breach
        self.alerter = alerter
        self.today_anchor_equity = None
        self.last_reset_at = None

    @staticmethod
    def get_mt5_equity():
        acc = mt5.account_info()
        if acc is None:
            raise RuntimeError("[PROP] Account info unavailable")
        return float(acc.equity)

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
            end_utc = end_tz_dt.astimezone(UTC).replace(tzinfo=None)
            deals = mt5.history_deals_get(start_utc, end_utc)

            cols = ['time','ticket','position_id','order','symbol','side','entry','volume','price',
                    'profit','commission','swap','magic','comment']
            rows = []
            closed_realized = 0.0
            if deals:
                for d in deals:
                    side = DEAL_TYPE_MAP.get(getattr(d,'type',None), str(getattr(d,'type','')))
                    entry = ENTRY_MAP.get(getattr(d,'entry',None), str(getattr(d,'entry','')))
                    tstr = _fmt_dt_utc_to_tz(getattr(d,'time',0), self.reset_tz)
                    profit = float(getattr(d,'profit',0.0))
                    if getattr(d, 'entry', None) in (getattr(mt5, "DEAL_ENTRY_OUT", 1),
                                                     getattr(mt5, "DEAL_ENTRY_OUT_BY", 3)):
                        closed_realized += profit
                    rows.append({
                        'time': tstr, 'ticket': getattr(d,'ticket',''), 'position_id': getattr(d,'position_id',''),
                        'order': getattr(d,'order',''), 'symbol': getattr(d,'symbol',''), 'side': side,
                        'entry': entry, 'volume': getattr(d,'volume',0.0), 'price': getattr(d,'price',0.0),
                        'profit': profit, 'commission': getattr(d,'commission',0.0),
                        'swap': getattr(d,'swap',0.0), 'magic': getattr(d,'magic',0), 'comment': getattr(d,'comment',''),
                    })
            df = pd.DataFrame(rows, columns=cols)

            open_pnl_now = self.get_open_pnl()
            daily_sum = closed_realized + open_pnl_now

            start_equity = self.today_anchor_equity
            end_equity = self.get_mt5_equity()
            eq_change = (end_equity - start_equity) if start_equity is not None else None

            day_str = start_tz_dt.strftime('%Y%m%d')
            out_path = os.path.join(report_dir, f'FTMO-Robbie_daily_{day_str}.csv')
            df.to_csv(out_path, index=False)
            print(f"[REPORT] Saved daily report → {out_path} "
                  f"(closed_realized=${closed_realized:,.2f}, open=${open_pnl_now:,.2f}, daily_sum=${daily_sum:,.2f})")

            if self.alerter and alerts_enabled:
                summary_lines = [
                    f"Window: {start_tz_dt.strftime('%Y-%m-%d %H:%M %Z')} → {end_tz_dt.strftime('%Y-%m-%d %H:%M %Z')}",
                    f"Closed PnL today:  ${closed_realized:,.2f}",
                    f"Open PnL now:      ${open_pnl_now:,.2f}",
                    f"Daily sum (C+O):   ${daily_sum:,.2f}",
                    (f"Start equity:      ${start_equity:,.2f}" if start_equity is not None else "Start equity: n/a"),
                    f"End equity now:    ${end_equity:,.2f}",
                ]
                if eq_change is not None:
                    summary_lines.append(f"Change (end-start): ${eq_change:,.2f}")
                summary = "\n".join(summary_lines) + "\n"
                self.alerter.daily_report(out_path, daily_sum, start_tz_dt.strftime('%Y-%m-%d'), summary=summary)
        except Exception as e:
            print(f"[REPORT] Failed to build/send daily report: {e}")

    def current_daily_loss(self):
        self.ensure_daily_anchor()
        closed = self.today_closed_pnl()
        openp  = self.get_open_pnl()
        return -(closed + openp), closed, openp

    def remaining_daily_risk(self):
        loss, closed, openp = self.current_daily_loss()
        initial_equity = self.today_anchor_equity or self.get_mt5_equity()
        daily_limit = initial_equity * (self.daily_loss_pct / 100.0)
        remaining = daily_limit - max(0.0, loss)
        return max(0.0, remaining), loss, closed, openp, daily_limit, initial_equity

    def breached_daily(self):
        remaining, loss, _, _, _, _ = self.remaining_daily_risk()
        return remaining <= 0.0, loss

    def breached_overall(self):
        eq = self.get_mt5_equity()
        # overall limit is relative to START OF CHALLENGE; here we approximate using today's anchor as baseline.
        min_equity = (self.today_anchor_equity or eq) * (1.0 - self.overall_loss_pct / 100.0)
        return eq < min_equity, eq, min_equity

    def profit_target_hit(self):
        eq = self.get_mt5_equity()
        anchor = self.today_anchor_equity or eq
        gain = eq - anchor
        target = anchor * (self.phase1_target_pct / 100.0) if self.phase == 1 else anchor * (self.phase2_target_pct / 100.0)
        return gain >= target, gain, target

    def would_breach_with_order(self, stop_loss_risk_usd: float) -> bool:
        remaining_daily, _, _, _, daily_limit, _ = self.remaining_daily_risk()
        if stop_loss_risk_usd > remaining_daily:
            print(f"[PROP] Order veto: SL risk ${stop_loss_risk_usd:.2f} > remaining daily ${remaining_daily:.2f} "
                  f"(limit ${daily_limit:.2f})")
            return True
        eq = self.get_mt5_equity()
        min_eq = (self.today_anchor_equity or eq) * (1.0 - self.overall_loss_pct / 100.0)
        if (eq - stop_loss_risk_usd) < min_eq:
            print("[PROP] Order veto: worst-case SL would breach overall max loss")
            return True
        return False

    def enforce_breaches(self):
        self.ensure_daily_anchor()
        daily_breached, loss = self.breached_daily()
        _, closed, openp, _, daily_limit, _ = self.remaining_daily_risk()
        overall_breached, eq, min_eq = self.breached_overall()

        breached = False
        if daily_breached:
            print(f"[BREACH] Max Daily Loss hit. Current daily loss ≈ ${loss:.2f}. No new trades.")
            breached = True
            if self.alerter and alerts_enabled and self.last_reset_at:
                day_key = self.last_reset_at.strftime('%Y-%m-%d')
                self.alerter.breach_daily(day_key, closed, openp, daily_limit, self.reset_tz.zone)

        if overall_breached:
            print(f"[BREACH] Max Overall Loss hit. Equity ${eq:.2f} < minimum ${min_eq:.2f}. No new trades.")
            breached = True
            if self.alerter and alerts_enabled:
                self.alerter.breach_overall(eq, min_eq)

        if breached and self.auto_close:
            pos = mt5.positions_get()
            if pos:
                print("[ACTION] Closing all open positions due to rule breach…")
                for p in pos:
                    _close_position_ticket(p)
        return breached

prop = PropRules(
    daily_loss_pct=max_daily_loss_pct,
    overall_loss_pct=max_overall_loss_pct,
    phase1_target_pct=phase1_target_pct,
    phase2_target_pct=phase2_target_pct,
    reset_tz=prop_reset_tz,
    reset_hour=prop_reset_hour,
    reset_minute=prop_reset_minute,
    phase=current_phase,
    auto_close_on_breach=auto_close_on_breach,
    alerter=alerter,
)

# ===================== STRATEGY PROFILES ===================== #
@dataclass
class StrategyProfile:
    name: str
    symbol: str                 # MT5 symbol
    oanda_symbol: str           # OANDA instrument
    granularity: str            # "M5", "M15", ...
    timeframe: int              # mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, ...
    magic_number: int           # unique per profile
    comment_tag: str            # appears on trades/comments
    session_windows: List[Tuple[str, str]]
    risk_pct_of_equity: float   # % of CURRENT equity
    sl_usd_distance: float
    tp1_usd_distance: Optional[float]
    tp1_fraction: Optional[float]     # None → no partial; BE can still move if be_after_tp1=True
    tp2_usd_distance: Optional[float]
    tp2_fraction: Optional[float]     # None → no partial
    tp2_move_sl_to_usd: Optional[float]  # None → do not move SL at TP2; else move SL to entry ± this USD after TP2 trigger
    be_after_tp1: bool
    be_buffer_usd: float
    be_use_spread_buffer: bool
    use_heikin_ashi: bool
    atr_period: int
    atr_mult: float

    # day-of-week controls and optional risk overrides
    trade_monday: bool = True  # if False → block NEW entries on Mondays
    trade_friday: bool = True  # if False → block NEW entries on Fridays
    monday_risk_pct_override: Optional[float] = None  # if set and Monday trading is enabled, use this risk%
    friday_risk_pct_override: Optional[float] = None  # if set and Friday trading is enabled, use this risk%

PROFILES: List[StrategyProfile] = [
    StrategyProfile(
        name =                  "XAU-5M",
        symbol =                "XAUUSD",
        oanda_symbol =          "XAU_USD",
        granularity =           "M5",
        timeframe =             mt5.TIMEFRAME_M5,
        magic_number =          112200,
        comment_tag =           "CEBot-Prop-5M",
        session_windows =       [("06:00", "12:00"), ("13:00", "18:00")],
        risk_pct_of_equity =    0.30,
        sl_usd_distance =       6.0,
        tp1_usd_distance =      7.0,
        tp1_fraction =          0.50,
        tp2_usd_distance =      25.0,
        tp2_fraction =          0.25,
        tp2_move_sl_to_usd =    20.0,
        be_after_tp1 =          True,
        be_buffer_usd =         0.20,
        be_use_spread_buffer =  True,
        use_heikin_ashi =       True,
        atr_period =            1,
        atr_mult =              1.85,

        trade_monday = True,                      # False to block new entries on Mondays
        trade_friday = True,                      # False to block new entries on Fridays
        monday_risk_pct_override = None,          # e.g., 0.20 to use 0.20% on Mondays (if enabled)
        friday_risk_pct_override = None,          # e.g., 0.20 to use 0.20% on Fridays (if enabled)
    ),
    StrategyProfile(
        name =                  "XAU-15M",
        symbol =                "XAUUSD",
        oanda_symbol =          "XAU_USD",
        granularity =           "M15",
        timeframe =             mt5.TIMEFRAME_M15,
        magic_number =          112201,
        comment_tag =           "CEBot-Prop-15M",
        session_windows =       [("01:00", "20:55")],
        risk_pct_of_equity =    0.50,
        sl_usd_distance =       10.0,
        tp1_usd_distance =      15.0,
        tp1_fraction =          0.50,
        tp2_usd_distance =      35.0,
        tp2_fraction =          0.30,
        tp2_move_sl_to_usd =    30.0,
        be_after_tp1 =          True,
        be_buffer_usd =         1.00,
        be_use_spread_buffer =  True,
        use_heikin_ashi =       True,
        atr_period =            1,
        atr_mult =              1.85,

        trade_monday = True,                      # False to block new entries on Mondays
        trade_friday = True,                      # False to block new entries on Fridays
        monday_risk_pct_override = None,          # e.g., 0.20 to use 0.20% on Mondays (if enabled)
        friday_risk_pct_override = None,          # e.g., 0.20 to use 0.20% on Fridays (if enabled)
    ),
]

# ===================== CONNECT TO MT5 ===================== #
print("MT5 Path Exists?", os.path.exists(mt5_terminal_path))
if not mt5.initialize(login=int(mt5_login) if str(mt5_login).isdigit() else None,
                      password=mt5_password,
                      server=mt5_server,
                      path=mt5_terminal_path):
    print("[ERROR] MT5 initialization failed:", mt5.last_error())
    sys.exit(1)

term = mt5.terminal_info()
if term and hasattr(term, "trade_allowed") and not term.trade_allowed:
    print("[FATAL] MT5 AutoTrading is DISABLED in the terminal. Enable the green 'Algo Trading' button "
          "and allow EAs in Tools > Options > Expert Advisors.")

account = mt5.account_info()
if account is None:
    raise RuntimeError(f"Failed to retrieve account info: {mt5.last_error()}\n")

print(f"\nMT5 ACCOUNT CONNECTED!\nACCOUNT: {account.login}\nBALANCE: ${account.balance:.2f}\n")

# (Optional) show symbol info for profiles' symbols
seen_symbols = set()
for prof in PROFILES:
    if prof.symbol in seen_symbols:
        continue
    seen_symbols.add(prof.symbol)
    symbol_info = mt5.symbol_info(prof.symbol)
    if symbol_info:
        print(f"[INFO] {prof.symbol} Lot Range: min={symbol_info.volume_min}, max={symbol_info.volume_max}, step={symbol_info.volume_step}")
        min_sl_usd = (getattr(symbol_info, "trade_stops_level", 0) or 0) * symbol_info.point
        print(f"[INFO] {prof.symbol} Min stop distance ≈ ${min_sl_usd:.2f}")
    else:
        print(f"[ERROR] Unable to fetch symbol info for {prof.symbol}")

Path(report_dir).mkdir(parents=True, exist_ok=True)

# ===================== EMAIL SMOKE TEST ===================== #
def _email_smoke_test():
    msg = EmailMessage()
    msg['Subject'] = 'FTMO PROP CHALLENGE - Robbie'
    msg['From']    = smtp_user or 'bot@localhost'
    msg['To']      = ', '.join(email_to) if email_to else (smtp_user or 'me@localhost')
    msg.set_content('If you can read this, SMTP is working. :)')
    import smtplib
    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as s:
        s.starttls()
        if smtp_user:
            s.login(smtp_user, smtp_pass)
        s.send_message(msg)
    print("[SMTP] Test email sent successfully.")

try:
    if email_enabled:
        _email_smoke_test()
except Exception as e:
    print(f"[SMTP] Test failed: {e}")

# ===================== DATA FETCH ===================== #
def fetch_oanda_candles(symbol: str, granularity: str, count: int = num_candles):
    url = f"{oanda_api_base}/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {oanda_token}"}
    params = {"granularity": granularity, "count": count, "price": "M"}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=(5, 15))
    except requests.RequestException as e:
        print(f"[ERROR] OANDA network error: {e.__class__.__name__}: {e}")
        return None

    if r.status_code != 200:
        print("[ERROR] Failed to fetch candles from OANDA:", r.status_code, r.text[:300])
        return None

    raw_candles = r.json().get("candles", [])
    data = {"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []}

    for c in raw_candles:
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

# ===================== SESSION HELPERS ===================== #
def _session_window_today(hhmm: str, tz):
    """Build a localized datetime for today at hh:mm in the given tz (pytz)."""
    h, m = map(int, hhmm.split(":"))
    naive = datetime.combine(datetime.now(tz).date(), dtime(h, m))
    # pytz localization keeps BST/GMT rules correct; London is currently GMT
    return tz.localize(naive)

def in_session_for(now_local: datetime, profile: 'StrategyProfile') -> bool:
    for start, end in profile.session_windows:
        start_dt = _session_window_today(start, local_tz)
        end_dt   = _session_window_today(end,   local_tz)
        if start_dt <= now_local < end_dt:
            return True
    return False

def session_bounds_for(now_local: datetime, profile: 'StrategyProfile'):
    """Returns (start_dt, end_dt) for the window that currently contains now_local, else (None, None)."""
    for start, end in profile.session_windows:
        start_dt = _session_window_today(start, local_tz)
        end_dt   = _session_window_today(end,   local_tz)
        if start_dt <= now_local < end_dt:
            return start_dt, end_dt
    return None, None

def next_bar_close(now_local: datetime, profile: 'StrategyProfile'):
    step = 5 if profile.granularity == "M5" else 15 if profile.granularity == "M15" else 5
    base = now_local.replace(second=0, microsecond=0)
    mins_to_add = (step - (base.minute % step)) % step
    target = base + timedelta(minutes=mins_to_add)
    if target <= now_local:
        target += timedelta(minutes=step)
    return target


# ===================== DAY-OF-WEEK & RISK HELPERS ===================== #
def is_trading_day_for(now_local: datetime, profile: 'StrategyProfile') -> bool:
    """
    Gate NEW ENTRIES by weekday according to per-profile flags.
    Management of existing positions still runs regardless of this gate.
    """
    wd = now_local.weekday()  # Monday=0 ... Sunday=6
    if wd >= 5:
        return False  # block Sat/Sun entries
    if wd == 0:
        return bool(profile.trade_monday)
    if wd == 4:
        return bool(profile.trade_friday)
    return True  # Tue–Thu


def effective_risk_pct_for(now_local: datetime, profile: 'StrategyProfile') -> float:
    """
    Returns the risk% to use RIGHT NOW for entries:
    - Monday: use monday_risk_pct_override if provided and Monday is enabled; else base risk_pct_of_equity
    - Friday: use friday_risk_pct_override if provided and Friday is enabled; else base risk_pct_of_equity
    - Tue–Thu: base risk_pct_of_equity
    """
    wd = now_local.weekday()
    if wd == 0:   # Monday
        if not profile.trade_monday:
            return profile.risk_pct_of_equity  # not used because entries are blocked, but safe default
        return profile.monday_risk_pct_override if profile.monday_risk_pct_override is not None else profile.risk_pct_of_equity
    if wd == 4:   # Friday
        if not profile.trade_friday:
            return profile.risk_pct_of_equity
        return profile.friday_risk_pct_override if profile.friday_risk_pct_override is not None else profile.risk_pct_of_equity
    return profile.risk_pct_of_equity


# ===================== INDICATORS ===================== #
def calculate_heikin_ashi(df):
    ha_df = pd.DataFrame(index=df.index)
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha_df['ha_close'].iloc[i - 1]) / 2)
    ha_df['ha_open'] = ha_open
    ha_df['ha_high'] = pd.concat([df['high'], ha_df['ha_open'], ha_df['ha_close']], axis=1).max(axis=1)
    ha_df['ha_low']  = pd.concat([df['low'],  ha_df['ha_open'], ha_df['ha_close']], axis=1).min(axis=1)
    return ha_df

def calculate_indicators(df, useHeikinAshi=True, atrPeriod=1, atrMult=1.85):
    if useHeikinAshi:
        ha = calculate_heikin_ashi(df)
        o = ha['ha_open']; h = ha['ha_high']; l = ha['ha_low']; c = ha['ha_close']
    else:
        o = df['open']; h = df['high']; l = df['low']; c = df['close']

    tr = pd.DataFrame(index=df.index)
    tr['o'] = o; tr['h'] = h; tr['l'] = l; tr['c'] = c
    tr['c_prev'] = tr['c'].shift(1)

    def _tr(row):
        if pd.isna(row['c_prev']):
            return row['h'] - row['l']
        return max(row['h'] - row['l'], abs(row['h'] - row['c_prev']), abs(row['l'] - row['c_prev']))
    tr['true_range'] = tr.apply(_tr, axis=1)

    n = int(max(1, atrPeriod))
    if n == 1:
        tr['atr'] = tr['true_range']
    else:
        vals = tr['true_range'].to_numpy()
        rma = [None] * len(vals)
        if len(vals) >= n:
            sma_seed = float(pd.Series(vals[:n]).mean())
            rma[n-1] = sma_seed
            alpha = 1.0 / n
            for i in range(n, len(vals)):
                rma[i] = rma[i-1] + alpha * (vals[i] - rma[i-1])
        tr['atr'] = pd.Series(rma, index=tr.index).ffill().bfill()

    atr_val = atrMult * tr['atr']
    hh = tr['h'].rolling(window=n, min_periods=n).max()
    ll = tr['l'].rolling(window=n, min_periods=n).min()
    long_stop  = hh - atr_val
    short_stop = ll + atr_val

    lss = long_stop.copy()
    sss = short_stop.copy()
    for i in range(len(tr)):
        if i == 0: continue
        long_prev  = lss.iloc[i-1] if pd.notna(lss.iloc[i-1]) else long_stop.iloc[i]
        short_prev = sss.iloc[i-1] if pd.notna(sss.iloc[i-1]) else short_stop.iloc[i]
        if tr['c'].iloc[i-1] > long_prev:
            lss.iloc[i] = max(long_stop.iloc[i], long_prev)
        else:
            lss.iloc[i] = long_stop.iloc[i]
        if tr['c'].iloc[i-1] < short_prev:
            sss.iloc[i] = min(short_stop.iloc[i], short_prev)
        else:
            sss.iloc[i] = short_stop.iloc[i]

    dir_vals = [1]
    for i in range(1, len(tr)):
        if tr['c'].iloc[i] > sss.iloc[i-1]:
            dir_vals.append(1)
        elif tr['c'].iloc[i] < lss.iloc[i-1]:
            dir_vals.append(-1)
        else:
            dir_vals.append(dir_vals[-1])

    tr['dir'] = dir_vals
    tr['dir_prev'] = tr['dir'].shift(1)
    tr['buy_signal']  = (tr['dir'] ==  1) & (tr['dir_prev'] == -1)
    tr['sell_signal'] = (tr['dir'] == -1) & (tr['dir_prev'] ==  1)

    tr['long_stop_smooth']  = lss
    tr['short_stop_smooth'] = sss
    tr['ha_open'] = o
    tr['ha_high'] = h
    tr['ha_low']  = l
    tr['ha_c']    = c
    return tr

# ===================== POSITION / ORDER HELPERS ===================== #
def _round_to_step(value, step):
    steps = math.floor(value / step)
    return max(steps * step, 0.0)

def _get_position(symbol, magic=None):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    for p in positions:
        if magic is None or p.magic == magic:
            return p
    return None

def _close_position_ticket(position):
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
        "deviation": slippage,
        "magic": position.magic,
        "comment": "PropRules Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Failed to close position: {getattr(res, 'retcode', None)}, {getattr(res, 'comment', None)}")
    else:
        print(f"[OK] POSITION CLOSED: {res}")

def compute_sl_price(symbol, action_type, entry_price, sl_usd):
    si = mt5.symbol_info(symbol)
    if si is None:
        return None
    digits = si.digits
    point = si.point
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

def _can_place_sl_now(position, sl_price):
    si = mt5.symbol_info(position.symbol)
    tick = mt5.symbol_info_tick(position.symbol)
    if si is None or tick is None:
        return False

    point = si.point
    stops_pts = getattr(si, "trade_stops_level", 0) or 0
    min_dist = stops_pts * point

    if position.type == mt5.ORDER_TYPE_BUY:
        max_allowed_sl = tick.bid - min_dist if min_dist > 0 else tick.bid
        return sl_price <= max_allowed_sl and sl_price < tick.bid
    else:
        min_allowed_sl = tick.ask + min_dist if min_dist > 0 else tick.ask
        return sl_price >= min_allowed_sl and sl_price > tick.ask

def _attempt_set_sl_with_magic(position, sl_price, magic, comment="Set SL (retry)"):
    mod = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": position.symbol,
        "position": position.ticket,
        "sl": sl_price,
        "magic": magic,
        "comment": comment,
    }
    res = mt5.order_send(mod)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[OK] SL set to {sl_price} on ticket {position.ticket}")
        return True
    print(f"[WARN] SL set attempt failed (ticket {position.ticket}): "
          f"{getattr(res,'retcode',None)} {getattr(res,'comment',None)}")
    return False

def compute_tp_price(symbol, action_type, entry_price, tp_usd):
    if tp_usd is None:
        return None
    si = mt5.symbol_info(symbol)
    if si is None:
        return None
    digits = si.digits
    tp = entry_price + float(tp_usd) if action_type == mt5.ORDER_TYPE_BUY else entry_price - float(tp_usd)
    return round(tp, digits)

def _round_volume_to_step(si, vol):
    vol = math.floor(vol / si.volume_step) * si.volume_step
    return max(si.volume_min, min(vol, si.volume_max))

def _partial_close(position, fraction):
    si = mt5.symbol_info(position.symbol)
    if si is None:
        return False
    part_vol_raw = position.volume * float(fraction)
    part_vol = _round_volume_to_step(si, part_vol_raw)
    remain_vol = round(position.volume - part_vol, 8)
    if remain_vol > 0 and remain_vol < si.volume_min:
        min_keep = si.volume_min
        part_vol = _round_volume_to_step(si, position.volume - min_keep)
        remain_vol = round(position.volume - part_vol, 8)
    if part_vol <= 0:
        return False

    opp_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        return False
    close_price = tick.bid if opp_type == mt5.ORDER_TYPE_SELL else tick.ask

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": part_vol,
        "type": opp_type,
        "position": position.ticket,
        "price": close_price,
        "deviation": slippage,
        "magic": position.magic,
        "comment": "PartialTP",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[OK] Partial closed {part_vol} from ticket {position.ticket}")
        return True
    print(f"[WARN] Partial close failed: {getattr(res,'retcode',None)} {getattr(res,'comment',None)}")
    return False

# ===================== RISK SIZING (Current Equity) ===================== #
def compute_lot_for_risk_dynamic_equity(symbol, sl_usd, risk_pct_of_equity: float):
    si = mt5.symbol_info(symbol)
    if si is None:
        print(f"[ERROR] Symbol info not found for {symbol}")
        return None, 0.0

    acc = mt5.account_info()
    if acc is None:
        print("[ERROR] Account info unavailable for equity sizing.")
        return None, 0.0

    risk_amount = float(acc.equity) * (float(risk_pct_of_equity) / 100.0)
    if risk_amount <= 0:
        return None, 0.0

    value_per_1usd_per_lot = None
    tv = getattr(si, "trade_tick_value", None)
    ts = getattr(si, "trade_tick_size", None)
    if tv not in (None, 0) and ts not in (None, 0):
        value_per_1usd_per_lot = float(tv) / float(ts)
    if value_per_1usd_per_lot is None:
        tv2 = getattr(si, "tick_value", None)
        ts2 = getattr(si, "tick_size", None)
        if tv2 not in (None, 0) and ts2 not in (None, 0):
            value_per_1usd_per_lot = float(tv2) / float(ts2)
    if value_per_1usd_per_lot is None:
        tv3 = getattr(si, "tick_value", None)
        pt  = getattr(si, "point", None)
        if tv3 not in (None, 0) and pt not in (None, 0):
            value_per_1usd_per_lot = float(tv3) / float(pt)
    if value_per_1usd_per_lot is None:
        cs = getattr(si, "trade_contract_size", None)
        if cs not in (None, 0):
            value_per_1usd_per_lot = float(cs)

    if value_per_1usd_per_lot in (None, 0):
        print("[ERROR] Cannot derive $ value per $1 move for 1 lot; missing tick fields.")
        return None, 0.0

    risk_per_lot = value_per_1usd_per_lot * float(sl_usd)
    if risk_per_lot <= 0:
        return None, 0.0

    raw_lot = risk_amount / risk_per_lot
    lot = _round_to_step(raw_lot, si.volume_step)
    lot = max(si.volume_min, min(lot, si.volume_max))
    return lot, risk_amount

# ===================== SL RETRY (PROFILE-AWARE) ===================== #
def _maybe_retry_initial_sl_profile(position, profile: StrategyProfile):
    # already has SL?
    if getattr(position, "sl", 0) not in (None, 0.0):
        pending_sl.pop(position.ticket, None)
        return True

    entry = pending_sl.get(position.ticket)
    if not entry:
        desired_sl = compute_sl_price(position.symbol, position.type, position.price_open, profile.sl_usd_distance)
        if desired_sl is not None:
            pending_sl[position.ticket] = {"desired_sl": desired_sl, "last_try": None, "magic": profile.magic_number}
        return False

    if entry.get("magic") != profile.magic_number:
        return False

    now = datetime.now(local_tz)
    last_try = entry.get("last_try")
    if last_try and (now - last_try).total_seconds() < SL_RETRY_EVERY_SECS:
        return False

    sl_price = entry["desired_sl"]
    entry["last_try"] = now

    if _can_place_sl_now(position, sl_price):
        if _attempt_set_sl_with_magic(position, sl_price, profile.magic_number, comment="Set SL post-fill (retry)"):
            pending_sl.pop(position.ticket, None)
            return True
        return False
    return False

def _periodic_sl_guard_multi():
    """Every SL_RETRY_EVERY_SECS: scan positions per profile and try to attach SL if missing."""
    global _last_sl_retry_at, _need_sl_seed_scan
    now = datetime.now(local_tz)
    if _last_sl_retry_at and (now - _last_sl_retry_at).total_seconds() < SL_RETRY_EVERY_SECS:
        return
    _last_sl_retry_at = now

    for profile in PROFILES:
        position = _get_position(profile.symbol, profile.magic_number)
        if position:
            if getattr(position, "sl", 0) in (None, 0.0) and position.ticket not in pending_sl:
                desired_sl = compute_sl_price(position.symbol, position.type, position.price_open, profile.sl_usd_distance)
                if desired_sl is not None:
                    pending_sl[position.ticket] = {"desired_sl": desired_sl, "last_try": None, "magic": profile.magic_number}
                    print(f"[{profile.name}] Queued SL retry for ticket {position.ticket}: target={desired_sl}")
            _maybe_retry_initial_sl_profile(position, profile)
            if getattr(position, "sl", 0) not in (None, 0.0):
                _need_sl_seed_scan = False
        else:
            if _need_sl_seed_scan:
                print(f"[{profile.name}] Waiting for position to appear to seed SL…")

# ===================== BE MOVE (PROFILE-AWARE) ===================== #
def _move_sl_to_breakeven_profile(position, profile: StrategyProfile):
    symbol = position.symbol
    si = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if si is None or tick is None:
        return False

    entry = float(position.price_open)
    point = si.point
    stops_pts = getattr(si, "trade_stops_level", 0) or 0
    min_dist = stops_pts * point

    spread = (tick.ask - tick.bid) if profile.be_use_spread_buffer else 0.0
    be_buf = max(float(profile.be_buffer_usd), spread)

    if position.type == mt5.ORDER_TYPE_BUY:
        desired_sl = round(entry + be_buf, si.digits)
        max_allowed_sl = tick.bid - min_dist if min_dist > 0 else tick.bid
        if desired_sl <= max_allowed_sl and desired_sl < tick.bid:
            return _attempt_set_sl_with_magic(position, desired_sl, profile.magic_number, comment="SL->BE")
        return False
    else:
        desired_sl = round(entry - be_buf, si.digits)
        min_allowed_sl = tick.ask + min_dist if min_dist > 0 else tick.ask
        if desired_sl >= min_allowed_sl and desired_sl > tick.ask:
            return _attempt_set_sl_with_magic(position, desired_sl, profile.magic_number, comment="SL->BE")
        return False

def _compute_sl_at_entry_offset(symbol: str, position, offset_usd: float):
    """
    Returns the SL price at 'entry ± offset_usd' in the profit-protecting direction.
    BUY  → entry + offset
    SELL → entry - offset
    Enforces broker min stop distance vs current bid/ask; returns None if not currently placeable.
    """
    si = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if si is None or tick is None:
        return None

    entry = float(position.price_open)
    point = si.point
    stops_pts = getattr(si, "trade_stops_level", 0) or 0
    min_dist = stops_pts * point

    if position.type == mt5.ORDER_TYPE_BUY:
        desired_sl = round(entry + float(offset_usd), si.digits)
        max_allowed_sl = tick.bid - min_dist if min_dist > 0 else tick.bid
        if desired_sl <= max_allowed_sl and desired_sl < tick.bid:
            return desired_sl
        return None
    else:
        desired_sl = round(entry - float(offset_usd), si.digits)
        min_allowed_sl = tick.ask + min_dist if min_dist > 0 else tick.ask
        if desired_sl >= min_allowed_sl and desired_sl > tick.ask:
            return desired_sl
        return None


# ===================== ORDER SEND (PROFILE-AWARE) ===================== #
def _filling_mode_candidates():
    modes = []
    if hasattr(mt5, "ORDER_FILLING_RETURN"): modes.append(mt5.ORDER_FILLING_RETURN)
    if hasattr(mt5, "ORDER_FILLING_FOK"):    modes.append(mt5.ORDER_FILLING_FOK)
    if hasattr(mt5, "ORDER_FILLING_IOC"):    modes.append(mt5.ORDER_FILLING_IOC)
    return modes or [mt5.ORDER_FILLING_IOC]

def send_order(profile: StrategyProfile, action_type, lot=None):
    global _need_sl_seed_scan

    # HARD SESSION VETO: never place orders outside this profile's session
    if not in_session_for(datetime.now(local_tz), profile):
        print(f"[{profile.name}] [BLOCK] Outside session — order veto.")
        return False

    # Day-of-week gate for NEW ENTRIES
    now_local = datetime.now(local_tz)
    if not is_trading_day_for(now_local, profile):
        dayname = now_local.strftime("%A")
        print(f"[{profile.name}] [BLOCK] {dayname} entries disabled by profile settings.")
        return False

    if prop.enforce_breaches():
        print(f"[{profile.name}] [BLOCK] Prop rule breached — order blocked.")
        return False

    symbol = profile.symbol
    if not mt5.symbol_select(symbol, True):
        print(f"[{profile.name}] [ERROR] Failed to select symbol {symbol}")
        return False
    si = mt5.symbol_info(symbol)
    if si is None:
        print(f"[{profile.name}] [ERROR] Symbol info not found for {symbol}")
        return False
    if hasattr(si, "trade_allowed") and not si.trade_allowed:
        print(f"[{profile.name}] [ERROR] Trading is not allowed for {symbol}")
        return False

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[{profile.name}] [ERROR] Failed to get tick data for {symbol}")
        return False
    entry_price = tick.ask if action_type == mt5.ORDER_TYPE_BUY else tick.bid

    sl_price_snapshot = compute_sl_price(symbol, action_type, entry_price, profile.sl_usd_distance)
    if sl_price_snapshot is None:
        print(f"[{profile.name}] [ERROR] Could not compute SL price.")
        return False
    sl_distance = abs(entry_price - sl_price_snapshot)

    if lot is None:
        effective_risk_pct = effective_risk_pct_for(datetime.now(local_tz), profile)
        lot, risk_amount = compute_lot_for_risk_dynamic_equity(symbol, sl_distance, effective_risk_pct)
        if lot is None or lot <= 0:
            print(f"[{profile.name}] [ERROR] Computed lot is invalid; aborting order.")
            return False
        print(f"[{profile.name}] [RISK] Risk: {effective_risk_pct:.2f}% of equity "
              f"(${risk_amount:.2f}) | SL dist: ${sl_distance:.2f} | Lot: {lot}")

    lot = _round_to_step(lot, si.volume_step)
    lot = max(si.volume_min, min(lot, si.volume_max))
    if lot < si.volume_min or lot > si.volume_max:
        print(f"[{profile.name}] [ERROR] Lot size {lot} out of range: min={si.volume_min}, max={si.volume_max}")
        return False

    v_per_1usd = None
    if getattr(si, 'trade_tick_value', 0) and getattr(si, 'trade_tick_size', 0):
        v_per_1usd = float(si.trade_tick_value) / float(si.trade_tick_size)
    elif getattr(si, 'tick_value', 0) and getattr(si, 'tick_size', 0):
        v_per_1usd = float(si.tick_value) / float(si.tick_size)
    elif getattr(si, 'tick_value', 0) and getattr(si, 'point', 0):
        v_per_1usd = float(si.tick_value) / float(si.point)
    elif getattr(si, 'trade_contract_size', 0):
        v_per_1usd = float(si.trade_contract_size)

    if v_per_1usd:
        worst_case_loss = v_per_1usd * sl_distance * float(lot)
        if prop.would_breach_with_order(worst_case_loss):
            print(f"[{profile.name}] [BLOCK] Order would breach prop rules (worst-case SL).")
            return False
    else:
        print(f"[{profile.name}] [WARN] Could not derive per-$ value reliably; skipping pre-trade veto.")

    request_base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": action_type,
        "price": entry_price,
        "deviation": slippage,
        "magic": profile.magic_number,
        "comment": profile.comment_tag,
        "type_time": mt5.ORDER_TIME_GTC,
    }

    last_res = None
    for fm in _filling_mode_candidates():
        req = dict(request_base, type_filling=fm)
        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[{profile.name}] [OK] ORDER PLACED (fill_mode={fm}): "
                  f"ticket={res.order}, price={entry_price}, lot={lot}")

            pos = _get_position(symbol, profile.magic_number)
            if pos:
                filled = pos.price_open
                desired_sl = compute_sl_price(symbol, action_type, filled, profile.sl_usd_distance)
                if desired_sl:
                    if not _attempt_set_sl_with_magic(pos, desired_sl, profile.magic_number, comment="Set SL post-fill"):
                        pending_sl[pos.ticket] = {"desired_sl": desired_sl, "last_try": None, "magic": profile.magic_number}
                        print(f"[{profile.name}] [INFO] SL queued for retry: ticket {pos.ticket} → {desired_sl}")
            else:
                print(f"[{profile.name}] [WARN] Position not visible yet; will scan to set SL.")
                _need_sl_seed_scan = True

            return True
        last_res = res

    print(f"[{profile.name}] [ERROR] ORDER FAILED across all filling modes.")
    if last_res:
        print(f"    retcode={last_res.retcode}, comment={last_res.comment}")
    else:
        print(f"    order_send returned None. last_error={mt5.last_error()}")
    return False

# ===================== EXECUTION WRAPPER (PROFILE-AWARE) ===================== #
def attempt_execution_for_signal(profile: StrategyProfile, desired_side: str) -> bool:

    # Guard against any execution attempts outside hours (incl. backfills)
    if not in_session_for(datetime.now(local_tz), profile):
        return False

    if prop.enforce_breaches():
        return False

    # Day-of-week entry gate
    now_local = datetime.now(local_tz)
    if not is_trading_day_for(now_local, profile):
        # Management of open positions is handled elsewhere; we just refuse new entries here.
        return False

    hit, _, _ = prop.profit_target_hit()
    if hit:
        return False

    position = _get_position(profile.symbol, profile.magic_number)
    open_side = None
    if position:
        open_side = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'

    if open_side == desired_side:
        return True

    if open_side and open_side != desired_side:
        _close_position_ticket(position)
        time.sleep(0.5)

    order_type = mt5.ORDER_TYPE_BUY if desired_side == 'BUY' else mt5.ORDER_TYPE_SELL
    return bool(send_order(profile, order_type))

# ===================== POSITION MANAGEMENT (PROFILE-AWARE) ===================== #
def manage_open_position_for(profile: StrategyProfile):
    if prop.enforce_breaches():
        return

    # If we're outside the profile's session, optionally close and stop managing
    if not in_session_for(datetime.now(local_tz), profile):
        if auto_close_at_session_end:
            position = _get_position(profile.symbol, profile.magic_number)
            if position:
                _close_position_ticket(position)
        return

    position = _get_position(profile.symbol, profile.magic_number)
    if not position:
        # prune stale state for this profile's tickets
        for t in list(pos_state.keys()):
            # (We only clear on explicit no-position detection)
            pass
        return

    tick = mt5.symbol_info_tick(position.symbol)
    si   = mt5.symbol_info(position.symbol)
    if tick is None or si is None:
        return

    ticket = position.ticket
    state = pos_state.get(ticket, {
        "partial50_done": False,
        "partial25_done": False,
        "moved_to_be": False,
        "tp2_sl_moved": False,
    })

    if not _maybe_retry_initial_sl_profile(position, profile):
        return

    tp1_price = compute_tp_price(position.symbol, position.type, position.price_open, profile.tp1_usd_distance)
    tp2_price = compute_tp_price(position.symbol, position.type, position.price_open, profile.tp2_usd_distance) \
                if profile.tp2_usd_distance is not None else None

    def _touched(target_price, pos_type):
        if target_price is None:
            return False
        if pos_type == mt5.ORDER_TYPE_BUY:
            return tick.bid >= target_price
        else:
            return tick.ask <= target_price

    tp1_hit = _touched(tp1_price, position.type) if tp1_price is not None else False

    # TP1 partial (optional)
    if tp1_hit and not state["partial50_done"] and profile.tp1_fraction:
        if _partial_close(position, profile.tp1_fraction):
            state["partial50_done"] = True
            pos_state[ticket] = state
            time.sleep(0.3)
            position = _get_position(profile.symbol, profile.magic_number)
            if not position:
                return
            tick = mt5.symbol_info_tick(position.symbol)
            si   = mt5.symbol_info(position.symbol)
            if tick is None or si is None:
                return

    # Move SL -> BE after TP1, even if no partial
    if profile.be_after_tp1 and tp1_hit and not state.get("moved_to_be", False):
        if _move_sl_to_breakeven_profile(position, profile):
            state["moved_to_be"] = True
            pos_state[ticket] = state

    # === TP2 actions (independent) ===
    tp2_hit = _touched(tp2_price, position.type) if profile.tp2_usd_distance is not None else False
    if tp2_hit:
        # 1) Optional partial at TP2
        if profile.tp2_fraction and not state.get("partial25_done", False):
            if _partial_close(position, profile.tp2_fraction):
                state["partial25_done"] = True
                pos_state[ticket] = state
                # refresh handles/prices after partial
                time.sleep(0.2)
                position = _get_position(profile.symbol, profile.magic_number)
                if not position:
                    return
                tick = mt5.symbol_info_tick(position.symbol)
                si = mt5.symbol_info(position.symbol)
                if tick is None or si is None:
                    return

        # 2) Optional SL move to entry ± tp2_move_sl_to_usd
        if (profile.tp2_move_sl_to_usd is not None) and (not state.get("tp2_sl_moved", False)):
            desired_sl = _compute_sl_at_entry_offset(position.symbol, position, profile.tp2_move_sl_to_usd)
            if desired_sl is not None:
                if _attempt_set_sl_with_magic(position, desired_sl, profile.magic_number, comment="SL->TP2Lock"):
                    state["tp2_sl_moved"] = True
                    pos_state[ticket] = state


# ===================== LEGACY PENDING SIGNAL (OPTIONAL) ===================== #
def _maybe_retry_pending():
    global pending_signal, pending_since, last_retry_at
    if not pending_signal:
        return

    # OPTIONAL: skip pending retries outside the active session of either profile that created it
    # If you track which profile set `pending_signal`, check that profile here. Otherwise skip globally:
    if all(not in_session_for(datetime.now(local_tz), p) for p in PROFILES):
        return

    hit, _, _ = prop.profit_target_hit()
    if hit:
        return
    now_local = datetime.now(local_tz)
    if last_retry_at and (now_local - last_retry_at).total_seconds() < RETRY_EVERY_SECS:
        return
    if prop.enforce_breaches():
        return
    print(f"[RETRY] Pending {pending_signal} — attempting execution…")
    # Here we could bind a default profile, but keeping this feature minimal
    last_retry_at = now_local

# ===================== BOOT + OPTIONAL REPORT ===================== #
print("[BOOT] PropRules engine active.")
for p in PROFILES:
    print(f"  - {p.name}: magic={p.magic_number} tf={p.granularity} sessions={p.session_windows}")

if FORCE_DAILY_REPORT_ON_START:
    try:
        now_tz = datetime.now(prop_reset_tz)
        start_tz = now_tz - timedelta(days=1)
        print(f"[REPORT] Forcing daily report for window: "
              f"{start_tz.strftime('%Y-%m-%d %H:%M %Z')} → {now_tz.strftime('%Y-%m-%d %H:%M %Z')}")
        prop._send_daily_report_for_window(start_tz, now_tz)
    except Exception as e:
        print(f"[REPORT] Startup forced daily report failed: {e}")

# ===================== MAIN MULTI-PROFILE LOOP ===================== #
last_candle_time_by_profile = {p.name: None for p in PROFILES}
saw_candle_after_session_start_by_profile = {p.name: False for p in PROFILES}

# next time each profile should finalize a bar (non-blocking scheduler)
next_target_by_profile = {p.name: None for p in PROFILES}

# Only print "IN SESSION" once per session
in_session_announced = {p.name: False for p in PROFILES}

# Track previous in/out-of-session state to print banners and close exactly at edges
last_in_session = {p.name: False for p in PROFILES}


try:
    while True:
        for profile in PROFILES:
            # global breach gate
            if prop.enforce_breaches():
                pending_signal = None
                pending_since  = None
                last_retry_at  = None
                time.sleep(0.25)
                continue

            now_local = datetime.now(local_tz)

            is_now_in_session = in_session_for(now_local, profile)

            # Edge: just entered session
            if is_now_in_session and not last_in_session[profile.name]:
                sess_start, sess_end = session_bounds_for(now_local, profile)
                dayname = now_local.strftime('%A')
                day_ok = is_trading_day_for(now_local, profile)
                gate_note = "" if day_ok else " (entries disabled today)"
                print(
                    f"{now_local.strftime('%H:%M:%S')} | [{profile.name} {profile.symbol}] IN SESSION: "
                    f"{sess_start.strftime('%H:%M')}–{sess_end.strftime('%H:%M')} | {dayname}{gate_note}")

                in_session_announced[profile.name] = True

            # Edge: just exited session
            if (not is_now_in_session) and last_in_session[profile.name]:
                print(f"{now_local.strftime('%H:%M:%S')} | [{profile.name} {profile.symbol}] SESSION ENDED")
                if auto_close_at_session_end:
                    pos = _get_position(profile.symbol, profile.magic_number)
                    if pos:
                        _close_position_ticket(pos)
                # reset schedulers/state for a clean next session
                next_target_by_profile[profile.name] = None
                saw_candle_after_session_start_by_profile[profile.name] = False
                in_session_announced[profile.name] = False

            # remember state for next tick
            last_in_session[profile.name] = is_now_in_session

            if not is_now_in_session:
                _periodic_sl_guard_multi()
                time.sleep(0.1)
                continue

            # ---- non-blocking scheduler for bar closes ----
            sess_start, sess_end = session_bounds_for(now_local, profile)
            if not in_session_announced[profile.name]:
                dayname = now_local.strftime('%A')
                day_ok = is_trading_day_for(now_local, profile)
                gate_note = "" if day_ok else " (entries disabled today)"
                print(
                    f"{now_local.strftime('%H:%M:%S')} | [{profile.name} {profile.symbol}] IN SESSION: "
                    f"{sess_start.strftime('%H:%M')}–{sess_end.strftime('%H:%M')} | {dayname}{gate_note}")

                in_session_announced[profile.name] = True

            # if session is about to end, cap the target to session end
            target = next_target_by_profile[profile.name]
            if target is None:
                # first schedule for this profile in this session round
                t = next_bar_close(now_local, profile)
                if sess_end and t > sess_end:
                    t = sess_end
                next_target_by_profile[profile.name] = t
                log(f"[{profile.name}] waiting for {profile.granularity} close at {t.strftime('%H:%M:%S %Z')} ...")

                # If we hit session end in the meantime, cancel work immediately
                if sess_end and datetime.now(local_tz) >= sess_end:
                    log(f"[{profile.name}] SESSION ENDED; cancelling scheduled work.")
                    if auto_close_at_session_end:
                        pos = _get_position(profile.symbol, profile.magic_number)
                        if pos:
                            _close_position_ticket(pos)
                    next_target_by_profile[profile.name] = None
                    saw_candle_after_session_start_by_profile[profile.name] = False
                    continue

            else:
                # if the cap moved (session end shifted day), re-cap it
                if sess_end and target > sess_end:
                    next_target_by_profile[profile.name] = sess_end
                    target = sess_end

            # run ongoing maintenance each tick for ALL profiles
            prop.enforce_breaches()
            _maybe_retry_pending()
            manage_open_position_for(profile)
            _periodic_sl_guard_multi()

            # if we haven't reached the target yet, give other profiles a turn
            if (next_target_by_profile[profile.name] - datetime.now(local_tz)).total_seconds() > 0:
                continue

            # we've reached or passed the target → treat as bar close for this profile
            time.sleep(0.2)  # tiny settle; avoids racing the data API

            # session end check
            if sess_end and datetime.now(local_tz) >= sess_end:
                log(f"[{profile.name}] SESSION ENDED BEFORE DATA.")
                pos = _get_position(profile.symbol, profile.magic_number)
                if auto_close_at_session_end and pos:
                    _close_position_ticket(pos)
                next_target_by_profile[profile.name] = None
                saw_candle_after_session_start_by_profile[profile.name] = False
                continue

            # fetch candles for this profile
            df = fetch_oanda_candles(symbol=profile.oanda_symbol, granularity=profile.granularity, count=num_candles)
            if df is None or df.empty:
                log(f"[{profile.name}] [TIMEOUT] No candle. Skipping.")
                continue

            latest_candle_time = df.index[-1]
            prev_last = last_candle_time_by_profile[profile.name]
            if prev_last is not None and latest_candle_time <= prev_last:
                log(f"[{profile.name}] [WAIT] No new candle yet.")
                continue

            last_candle_time_by_profile[profile.name] = latest_candle_time
            # schedule the next bar close now that we processed this one
            nt = next_bar_close(datetime.now(local_tz), profile)
            if sess_end and nt > sess_end:
                nt = sess_end
            next_target_by_profile[profile.name] = nt
            log(f"[{profile.name}] New {profile.granularity} candle: {latest_candle_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            log(f"[{profile.name}] next {profile.granularity} close at {nt.strftime('%H:%M:%S %Z')} ...")

            # If we hit session end in the meantime, cancel work immediately
            if sess_end and datetime.now(local_tz) >= sess_end:
                log(f"[{profile.name}] SESSION ENDED; cancelling scheduled work.")
                if auto_close_at_session_end:
                    pos = _get_position(profile.symbol, profile.magic_number)
                    if pos:
                        _close_position_ticket(pos)
                next_target_by_profile[profile.name] = None
                saw_candle_after_session_start_by_profile[profile.name] = False
                continue

            # first post-session bar enforcement
            if require_post_start_candle and not saw_candle_after_session_start_by_profile[profile.name]:
                if sess_start and latest_candle_time <= sess_start:
                    log(f"[{profile.name}] [WAIT] first bar after session open not closed yet.")
                    continue
                saw_candle_after_session_start_by_profile[profile.name] = True

            # indicators + signals
            tr = calculate_indicators(df, useHeikinAshi=profile.use_heikin_ashi,
                                      atrPeriod=profile.atr_period, atrMult=profile.atr_mult)
            latest = tr.iloc[-1]
            signal = 'BUY' if bool(latest['buy_signal']) else ('SELL' if bool(latest['sell_signal']) else None)
            prev_signal = None
            if len(tr) >= 2:
                prev = tr.iloc[-2]
                prev_signal = 'BUY' if prev['buy_signal'] else ('SELL' if prev['sell_signal'] else None)

            # position state print
            position = _get_position(profile.symbol, profile.magic_number)
            open_side = None
            if position:
                open_side = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'
                print(f"[{profile.name}] OPEN: {open_side}, vol: {position.volume}, entry: {position.price_open}")
            else:
                print(f"[{profile.name}] No open position.")

            # ensure initial SL queued if missing
            if position and getattr(position, "sl", 0) in (None, 0.0) and position.ticket not in pending_sl:
                desired_sl = compute_sl_price(position.symbol, position.type, position.price_open, profile.sl_usd_distance)
                if desired_sl:
                    pending_sl[position.ticket] = {"desired_sl": desired_sl, "last_try": None, "magic": profile.magic_number}
                    print(f"[{profile.name}] Queued SL retry for ticket {position.ticket}: target={desired_sl}")

            # manage open position
            manage_open_position_for(profile)

            # session end guard
            if sess_end and datetime.now(local_tz) >= sess_end:
                log(f"[{profile.name}] SESSION ENDED POST-CALC.")
                if auto_close_at_session_end and position:
                    _close_position_ticket(position)
                pending_signal = None
                pending_since  = None
                last_retry_at  = None
                continue

            # profit target (global)
            hit, gain, target_amt = prop.profit_target_hit()
            if hit:
                print(f"[{profile.name}] [TARGET] Phase {prop.phase} hit: Gain ${gain:.2f} ≥ ${target_amt:.2f}. Halting new trades.")
                if alerts_enabled and alerter:
                    alerter.phase_passed(prop.phase, gain, target_amt)
                pending_signal = None
                pending_since  = None
                last_retry_at  = None
                time.sleep(0.25)
                continue

            # execute signal / backfill
            if signal:
                if open_side != signal:
                    # small diag (daily remaining is relative to anchor)
                    try:
                        rem, loss, closed, openp, limit, _ = prop.remaining_daily_risk()
                        log(f"[DIAG] Remaining daily risk: ${rem:,.2f} | Daily loss: ${loss:,.2f} "
                            f"(closed=${closed:,.2f}, open=${openp:,.2f}, limit=${limit:,.2f})")
                    except Exception as e:
                        log(f"[DIAG] risk calc failed: {e}")

                    print(f"[{profile.name}] [TRADE] signal={signal} | open={open_side or 'NONE'}")
                    ok = attempt_execution_for_signal(profile, signal)
                    if not ok:
                        print(f"[{profile.name}] [INFO] order send failed; will try next bar.")
            else:
                if prev_signal and open_side != prev_signal:
                    print(f"[{profile.name}] [BACKFILL] prev bar had {prev_signal} — attempting.")
                    attempt_execution_for_signal(profile, prev_signal)

        # short idle between profile rounds
        time.sleep(0.1)

except KeyboardInterrupt:
    log("[INFO] Stopped by user (CTRL-C).")
finally:
    try:
        mt5.shutdown()
        log("[INFO] MT5 shutdown complete.")
    except Exception as e:
        log(f"[WARN] MT5 shutdown error: {e}")
