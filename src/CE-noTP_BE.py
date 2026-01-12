"""
Chandelier Exit Strategy for XAUUSD — FTMO-Style 2-Step Prop Rules (Demo)
Using Heikin-Ashi Candles | XAUUSD | 5 Minute TF

- Single-file version (no utils/; no HTTPAdapter)
- Uses requests.get(...) directly for OANDA candles
- Prints raw OANDA OHLCV and Heikin-Ashi + signals each bar

- FORCE_DAILY_REPORT_ON_START = False  # change to True if you want the startup 24h report
"""

import os
import sys
import time
import math
from datetime import datetime, timedelta
import pandas as pd
from pytz import timezone, UTC

import requests  # <— simple HTTP only
import MetaTrader5 as mt5

from email.message import EmailMessage
from pathlib import Path

# === USER CONFIG: MARKETS & SESSION === #
mt5_symbol        = "XAUUSD"
oanda_symbol      = "XAU_USD"
mt5_timeframe     = mt5.TIMEFRAME_M5
num_candles       = 500
use_heikin_ashi   = True
atr_period        = 1
atr_mult          = 1.85
slippage          = 10
magic_number      = 114477

# local/logging tz (affects printing & session windows only)
local_tz = timezone('Europe/London')

# --- LOGGING HELPERS --- #
def log(msg: str):
    ts = datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts} | {msg}", flush=True)

# === RISK & STOPS (STATIC VS INITIAL ACCOUNT) === #
# Risk is percentage of INITIAL account size (prop-style), not floating balance.
risk_per_trade_pct_of_initial = 0.50   # % of initial account size risked per trade
sl_usd_distance                = 5.5    # fixed $ distance from entry (price units)

# === PARTIAL TP & BREAK-EVEN MGMT === #
# Two-tier partials:
tp1_usd_distance      = 7.5      # first target distance from entry (price units)
tp1_fraction          = 0.00     # close 50% at first target

tp2_usd_distance      = None     # second target distance from entry (price units)
tp2_fraction          = None     # close 25% at second target

# Break-even: ONLY after tp1 (50%) is booked
breakeven_buffer_usd  = 0.20     # extra beyond entry to cover slippage/spread
use_spread_for_be_buffer = True  # if True, buffer = max(break-even_buffer_usd, current spread)

# --- SL retry cadence ---
SL_RETRY_EVERY_SECS = 10
_last_sl_retry_at = None
_need_sl_seed_scan = False  # set True after order placed if position not yet visible


# (legacy single TP vars kept unused on purpose to avoid breaking other parts)
# tp_usd_distance     = 10.0
# partial_tp_fraction = 0.50


# === PROP RULES CONFIG (FTMO-STYLE, 2-STEP) === #
account_size_usd         = 100_000.00
phase1_target_pct        = 10.0
phase2_target_pct        = 5.0
max_daily_loss_pct       = 5.0
max_overall_loss_pct     = 10.0

prop_reset_tz            = timezone('Europe/Prague')
prop_reset_hour          = 0
prop_reset_minute        = 0

current_phase            = 1
auto_close_on_breach     = True

# === TRADING SESSION WINDOWS (Europe/London local time) === #
session_windows = [
    ("05:00", "12:00"),  # London morning
    ("12:00", "17:00"),  # US session + London/NY overlap
]
auto_close_at_session_end   = True
require_post_start_candle   = True

# --- pending trade state for auto-retry/backfill --- #
pending_signal = None        # 'BUY' or 'SELL' queued for execution
pending_since  = None        # datetime when it was queued
last_retry_at  = None        # last retry timestamp
RETRY_EVERY_SECS = 15        # retry throttle while waiting

# Track per-position actions to avoid repeats
# ticket -> {"partial50_done": bool, "partial25_done": bool, "moved_to_be": bool}
pos_state = {}

# ticket -> {"desired_sl": float}
pending_sl = {}

# === ALERTS & REPORTS (EMAIL OPTIONAL) === #
alerts_enabled           = True
email_enabled            = True

report_dir               = r"C:\Users\anish\OneDrive\Desktop\Anish\A - EAs Reports\FTMO-Robbie"

smtp_host = "smtp.gmail.com"
smtp_port = 587
smtp_user = "anishv2610@gmail.com"
smtp_pass = "lpignmmkhgymwlpi"
email_to  = ["ea.prop0925@gmail.com"]

# === STARTUP REPORT TOGGLE ===
# If True: send a "yesterday → now" FTMO daily report + CSV ON STARTUP (useful for testing).
# Set to False once you’re done testing so it won’t email every time you restart.
FORCE_DAILY_REPORT_ON_START = False

# === ACCOUNT LOGIN CONFIG (UNCHANGED) === #
mt5_login = 52583222
mt5_password = "98qJk@gb2qmlDP"
mt5_server = "ICMarketsSC-Demo"
mt5_terminal_path = r"C:\MT5\EA-Prop_FTMO\FTMO-Robbie\terminal64.exe"

# === OANDA CONFIG (simple requests) — UNCHANGED === #
oanda_token       = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
oanda_api_base    = "https://api-fxpractice.oanda.com/v3"

# ============================================================
# CONNECT TO MT5
# ============================================================
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
    # Optionally: sys.exit(1)

account = mt5.account_info()
if account is None:
    raise RuntimeError(f"Failed to retrieve account info: {mt5.last_error()}\n")

print(f"\nMT5 ACCOUNT CONNECTED!\nACCOUNT: {account.login}\nBALANCE: ${account.balance:.2f}\n")

symbol_info = mt5.symbol_info(mt5_symbol)
if symbol_info:
    print(f"[INFO] {mt5_symbol} Lot Range: min={symbol_info.volume_min}, max={symbol_info.volume_max}, step={symbol_info.volume_step}")
    min_sl_usd = (getattr(symbol_info, "trade_stops_level", 0) or 0) * symbol_info.point
    print(f"[INFO] Min stop distance enforced by broker ≈ ${min_sl_usd:.2f}")
else:
    print(f"[ERROR] Unable to fetch symbol info for {mt5_symbol}")

# ensure report directory exists
Path(report_dir).mkdir(parents=True, exist_ok=True)

# ============================================================
# EMAIL TEST (optional)
# ============================================================
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

# ============================================================
# SIMPLE OANDA FETCH (requests.get)
# ============================================================
def fetch_oanda_candles(symbol=oanda_symbol, granularity="M5", count=num_candles):
    """
    Fetch candle data from OANDA REST API using requests.get.
    Returns a pandas DataFrame indexed by local_tz time.
    """
    url = f"{oanda_api_base}/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {oanda_token}"}
    params = {"granularity": granularity, "count": count, "price": "M"}  # Midpoint prices

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
            # make timezone-aware and convert to local
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

    df = pd.DataFrame(data).set_index("time")
    return df

# ============================================================
# ALERTER
# ============================================================
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

# ============================================================
# PROP RULES
# ============================================================
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
            if now < self.last_reset_at:
                pass
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
            daily_loss_amount = -daily_sum
            limit = self.daily_limit
            remaining = max(0.0, limit - max(0.0, daily_loss_amount))

            start_equity = None
            if self.last_reset_at and abs((start_tz_dt - self.last_reset_at).total_seconds()) < 120:
                start_equity = float(self.today_anchor_equity) if self.today_anchor_equity is not None else None

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
                    f"Max Daily Loss:    ${limit:,.2f}",
                    f"Remaining today:   ${remaining:,.2f}",
                    (f"Start equity:       ${start_equity:,.2f}" if start_equity is not None else "Start equity:       n/a"),
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
        _, closed, openp = self.current_daily_loss()
        overall_breached, eq, min_eq = self.breached_overall()

        breached = False
        if daily_breached:
            print(f"[BREACH] Max Daily Loss hit. Current daily loss ≈ ${loss:.2f}. No new trades.")
            breached = True
            if self.alerter and alerts_enabled and self.last_reset_at:
                day_key = self.last_reset_at.strftime('%Y-%m-%d')
                self.alerter.breach_daily(day_key, closed, openp, self.daily_limit, self.reset_tz.zone)

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

alerter = Alerter()

prop = PropRules(
    initial_account_size=account_size_usd,
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

# ============================================================
# UTILS
# ============================================================

def force_daily_report_now():
    try:
        # Use the reset timezone directly
        now_tz = datetime.now(prop_reset_tz)
        start_tz = now_tz - timedelta(days=1)
        print(f"[REPORT] Forcing daily report for window: "
              f"{start_tz.strftime('%Y-%m-%d %H:%M %Z')} → {now_tz.strftime('%Y-%m-%d %H:%M %Z')}")
        prop._send_daily_report_for_window(start_tz, now_tz)
    except Exception as e:
        print(f"[REPORT] Startup forced daily report failed: {e}")


def print_daily_risk_diag():
    try:
        rem, loss, closed, openp = prop.remaining_daily_risk()
        log(f"[DIAG] Remaining daily risk: ${rem:,.2f} | Daily loss so far: ${loss:,.2f} "
            f"(closed=${closed:,.2f}, open=${openp:,.2f})")
    except Exception as e:
        log(f"[DIAG] Failed to compute daily risk: {e}")

def _round_to_step(value, step):
    steps = math.floor(value / step)
    return max(steps * step, 0.0)

def _make_dt(base_dt, hhmm, tz):
    h, m = map(int, hhmm.split(":"))
    return base_dt.astimezone(tz).replace(hour=h, minute=m, second=0, microsecond=0)

def in_session(now_local):
    for start, end in session_windows:
        start_dt = _make_dt(now_local, start, local_tz)
        end_dt   = _make_dt(now_local, end,   local_tz)
        if start_dt <= now_local < end_dt:
            return True
    return False

def current_session_bounds(now_local):
    for start, end in session_windows:
        start_dt = _make_dt(now_local, start, local_tz)
        end_dt   = _make_dt(now_local, end,   local_tz)
        if start_dt <= now_local < end_dt:
            return start_dt, end_dt
    return None, None

def _next_5m_close(now):
    """
    Return the next 5-minute bar close time strictly in the future.
    Example: at 13:40:02 → returns 13:45:00.
    """
    base = now.replace(second=0, microsecond=0)
    mins_to_add = (5 - (base.minute % 5)) % 5
    target = base + timedelta(minutes=mins_to_add)
    if target <= now:
        target += timedelta(minutes=5)
    return target

def countdown_to_next_5m(tz, until_dt=None, prop_obj=None):
    target = _next_5m_close(datetime.now(tz))
    if until_dt is not None and target > until_dt:
        target = until_dt
        log(f"[*] Session ends before the next 5-min close → waiting until {target.strftime('%H:%M:%S %Z')} ...")
    else:
        log(f"[*] Waiting until next 5-min close at {target.strftime('%H:%M:%S %Z')} ...")
    while True:
        if prop_obj is not None:
            prop_obj.enforce_breaches()
        _maybe_retry_pending()
        _maybe_manage_open_position()
        _periodic_sl_guard()
        now = datetime.now(tz)
        if (target - now).total_seconds() <= 0:
            break
        time.sleep(0.25)
    time.sleep(2)  # allow API to finalize the bar
    return target

def countdown_to(target_dt, tz, prop_obj=None):
    log(f"[*] Waiting until {target_dt.strftime('%H:%M:%S %Z')} ...")
    while True:
        if prop_obj is not None:
            prop_obj.enforce_breaches()
        _maybe_retry_pending()
        _maybe_manage_open_position()
        _periodic_sl_guard()
        now = datetime.now(tz)
        if (target_dt - now).total_seconds() <= 0:
            break
        time.sleep(0.25)
    time.sleep(2)

# ============================================================
# STRATEGY: HEIKIN-ASHI + CHANDELIER
# ============================================================
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

# ============================================================
# POSITION / ORDER HELPERS
# ============================================================
def _get_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    for p in positions:
        if p.magic == magic_number:
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
        "magic": magic_number,
        "comment": "PropRules Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Failed to close position: {getattr(res, 'retcode', None)}, {getattr(res, 'comment', None)}")
    else:
        print(f"[OK] POSITION CLOSED: {res}")

def compute_lot_for_risk_static_initial(symbol, sl_usd, risk_pct_initial):
    si = mt5.symbol_info(symbol)
    if si is None:
        print(f"[ERROR] Symbol info not found for {symbol}")
        return None, 0.0

    risk_amount = float(account_size_usd) * (float(risk_pct_initial) / 100.0)
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

def _periodic_sl_guard():
    """
    Every SL_RETRY_EVERY_SECS:
      - If we have a position with no SL and no queue, queue it.
      - If we have a queued SL for a visible position, retry (throttled inside _maybe_retry_initial_sl).
      - If position not yet visible after a recent fill, keep scanning and seed SL as soon as it appears.
    """
    global _last_sl_retry_at, _need_sl_seed_scan

    now = datetime.now(local_tz)
    if _last_sl_retry_at and (now - _last_sl_retry_at).total_seconds() < SL_RETRY_EVERY_SECS:
        return
    _last_sl_retry_at = now

    position = _get_position(mt5_symbol)

    # If a position exists but has no SL and isn't queued, queue it now.
    if position:
        if getattr(position, "sl", 0) in (None, 0.0) and position.ticket not in pending_sl:
            _queue_initial_sl(position)

        # If queued, try (throttled by _maybe_retry_initial_sl itself)
        _maybe_retry_initial_sl(position)

        # If SL is finally present, clear the seed-scan flag
        if getattr(position, "sl", 0) not in (None, 0.0):
            _need_sl_seed_scan = False
    else:
        # No position visible yet; if a recent order said to seed, keep waiting
        if _need_sl_seed_scan:
            print("[INFO] Waiting for position to appear to seed SL…")

def compute_tp_price(symbol, action_type, entry_price, tp_usd):
    """Return absolute TP price at a fixed distance from entry, or None if tp_usd is None."""
    if tp_usd is None:
        return None
    si = mt5.symbol_info(symbol)
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

def _partial_close(position, fraction):
    """Close a fraction of the position at market; return True on success."""
    si = mt5.symbol_info(position.symbol)
    if si is None:
        return False
    part_vol_raw = position.volume * float(fraction)
    part_vol = _round_volume_to_step(si, part_vol_raw)

    # Ensure at least min volume remains (if possible)
    remain_vol = position.volume - part_vol
    remain_vol = round(remain_vol, 8)
    if remain_vol > 0 and remain_vol < si.volume_min:
        # Too small to leave a runner; either reduce part close or skip
        # Prefer to reduce the partial so that remainder >= min
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
        "magic": magic_number,
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

def _move_sl_to_breakeven(position, buffer_usd=0.0):
    """
    Move SL to entry ± buffer. Only do it if broker min stop distance from CURRENT price allows it.
    Returns True if SL was moved, False if not possible yet.
    """
    symbol = position.symbol
    si = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if si is None or tick is None:
        return False

    entry = float(position.price_open)
    point = si.point
    stops_pts = getattr(si, "trade_stops_level", 0) or 0
    min_dist = stops_pts * point

    spread = (tick.ask - tick.bid) if use_spread_for_be_buffer else 0.0
    be_buf = max(float(buffer_usd), spread)

    if position.type == mt5.ORDER_TYPE_BUY:
        desired_sl = round(entry + be_buf, si.digits)
        # SL for a long must be <= bid - min_dist
        max_allowed_sl = tick.bid - min_dist if min_dist > 0 else tick.bid
        if desired_sl <= max_allowed_sl and desired_sl < tick.bid:
            mod = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": position.ticket,
                "sl": desired_sl,
                "magic": magic_number,
                "comment": "SL->BE",
            }
            res = mt5.order_send(mod)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[OK] SL moved to BE+buffer at {desired_sl} (buy).")
                return True
            print(f"[WARN] BE SL set failed: {getattr(res,'retcode',None)} {getattr(res,'comment',None)}")
            return False
        # Not yet allowed (price not far enough to satisfy min distance)
        return False

    else:  # SELL
        desired_sl = round(entry - be_buf, si.digits)
        # SL for a short must be >= ask + min_dist
        min_allowed_sl = tick.ask + min_dist if min_dist > 0 else tick.ask
        if desired_sl >= min_allowed_sl and desired_sl > tick.ask:
            mod = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": position.ticket,
                "sl": desired_sl,
                "magic": magic_number,
                "comment": "SL->BE",
            }
            res = mt5.order_send(mod)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[OK] SL moved to BE+buffer at {desired_sl} (sell).")
                return True
            print(f"[WARN] BE SL set failed: {getattr(res,'retcode',None)} {getattr(res,'comment',None)}")
            return False
        return False

def _can_place_sl_now(position, sl_price):
    """Return True if broker min-distance rules allow setting sl_price now."""
    si = mt5.symbol_info(position.symbol)
    tick = mt5.symbol_info_tick(position.symbol)
    if si is None or tick is None:
        return False

    point = si.point
    stops_pts = getattr(si, "trade_stops_level", 0) or 0
    min_dist = stops_pts * point

    if position.type == mt5.ORDER_TYPE_BUY:
        # SL must be <= bid - min_dist AND below current bid
        max_allowed_sl = tick.bid - min_dist if min_dist > 0 else tick.bid
        return sl_price <= max_allowed_sl and sl_price < tick.bid
    else:
        # SELL: SL must be >= ask + min_dist AND above current ask
        min_allowed_sl = tick.ask + min_dist if min_dist > 0 else tick.ask
        return sl_price >= min_allowed_sl and sl_price > tick.ask


def _attempt_set_sl(position, sl_price, comment="Set SL (retry)"):
    """Try to set SL for an existing position; return True on success."""
    mod = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": position.symbol,
        "position": position.ticket,
        "sl": sl_price,
        "magic": magic_number,
        "comment": comment,
    }
    res = mt5.order_send(mod)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[OK] SL set to {sl_price} on ticket {position.ticket}")
        return True
    print(f"[WARN] SL set attempt failed (ticket {position.ticket}): "
          f"{getattr(res,'retcode',None)} {getattr(res,'comment',None)}")
    return False


def _queue_initial_sl(position):
    desired_sl = compute_sl_price(position.symbol, position.type, position.price_open, sl_usd_distance)
    if desired_sl is None:
        return
    pending_sl[position.ticket] = {"desired_sl": desired_sl, "last_try": None}
    print(f"[INFO] Queued SL retry for ticket {position.ticket}: target={desired_sl}")


def _maybe_retry_initial_sl(position):
    """
    If this ticket has a queued SL, try to set it now (throttled by SL_RETRY_EVERY_SECS).
    Returns True if SL is set (or no retry needed), False to keep retrying.
    """
    # If position already has an SL, clear queue.
    if getattr(position, "sl", 0) not in (None, 0.0):
        pending_sl.pop(position.ticket, None)
        return True

    entry = pending_sl.get(position.ticket)
    if not entry:
        return True

    now = datetime.now(local_tz)
    last_try = entry.get("last_try")
    if last_try and (now - last_try).total_seconds() < SL_RETRY_EVERY_SECS:
        return False  # too soon to retry

    sl_price = entry["desired_sl"]
    entry["last_try"] = now  # stamp attempt time

    if _can_place_sl_now(position, sl_price):
        if _attempt_set_sl(position, sl_price, comment="Set SL post-fill (retry)"):
            pending_sl.pop(position.ticket, None)
            return True
        # If server rejected but distance was OK, keep queued to try later.
        return False
    # Not yet allowed by min-distance vs current price; try again later.
    return False


def _maybe_manage_open_position():
    """
    Position management sequence (per your rules):
      1) Take 50% at tp1_usd_distance (e.g., $6)
      2) After (and only after) 50% is booked, move SL to BE+buffer (respecting broker min distance)
      3) Take another 25% at tp2_usd_distance (e.g., $10)
      4) Leave the runner until an opposite signal closes/flips it (handled by your signal logic)
    """
    # respect prop breaches (won't open new risk, but if account is halted, do nothing)
    if prop.enforce_breaches():
        return

    position = _get_position(mt5_symbol)

    if not position:
        # Clean stale states if no positions for our symbol
        for t in list(pos_state.keys()):
            pos_state.pop(t, None)
        for t in list(pending_sl.keys()):
            pending_sl.pop(t, None)
        return

    tick = mt5.symbol_info_tick(position.symbol)
    si   = mt5.symbol_info(position.symbol)
    if tick is None or si is None:
        return

    ticket = position.ticket
    state = pos_state.get(ticket, {"partial50_done": False, "partial25_done": False, "moved_to_be": False})

    # Ensure initial SL is on, retry if it was queued and broker now allows it.
    if not _maybe_retry_initial_sl(position):
        # SL not yet set due to broker distance; don't proceed to partials/BE yet.
        # We prefer to secure the protective SL first.
        return

    # Pre-compute TP prices for this entry
    tp1_price = compute_tp_price(position.symbol, position.type, position.price_open, tp1_usd_distance)
    tp2_price = compute_tp_price(position.symbol, position.type, position.price_open,
                                 tp2_usd_distance) if tp2_fraction else None
    # Require at least TP1
    if tp1_price is None:
        return

    # Helper: has price touched target for the current side (use executable-side price)
    def _touched(target_price, pos_type):
        if target_price is None:
            return False
        if pos_type == mt5.ORDER_TYPE_BUY:
            # closing a BUY uses SELL at bid; need bid >= target
            return tick.bid >= target_price
        else:
            # closing a SELL uses BUY at ask; need ask <= target
            return tick.ask <= target_price

    # If price overshoots straight to TP2 and we haven't taken TP1 yet, do TP1 first.
    tp1_hit = _touched(tp1_price, position.type)
    tp2_hit = _touched(tp2_price, position.type) if tp2_price is not None else False

    # 1) First partial: 50% at TP1
    if not state["partial50_done"] and tp1_hit:
        ok = _partial_close(position, tp1_fraction)
        if ok:
            state["partial50_done"] = True
            pos_state[ticket] = state

            # refresh position after partial (volume changed)
            time.sleep(0.3)
            position = _get_position(mt5_symbol)
            if not position:
                # whole position gone? then nothing else to do
                return
            tick = mt5.symbol_info_tick(position.symbol)
            si   = mt5.symbol_info(position.symbol)
            if tick is None or si is None:
                return

    # 2) Move SL to BE ONLY AFTER TP1 is done (keep retrying until broker allows)
    if state["partial50_done"] and not state["moved_to_be"]:
        if _move_sl_to_breakeven(position, breakeven_buffer_usd):
            state["moved_to_be"] = True
            pos_state[ticket] = state
        # If broker min distance blocks it now, we’ll try again next loop automatically.

    # 3) Second partial: at TP2 (only if configured)
    if tp2_price is not None:
        tp2_hit = _touched(tp2_price, position.type)

        if tp2_hit:
            # refresh latest objects — they might have changed after TP1 / BE move
            position = _get_position(mt5_symbol)
            if not position:
                return
            tick = mt5.symbol_info_tick(position.symbol)
            si = mt5.symbol_info(position.symbol)
            if tick is None or si is None:
                return

            if not state.get("partial25_done", False):
                ok2 = _partial_close(position, tp2_fraction)
                if ok2:
                    state["partial25_done"] = True
                    pos_state[ticket] = state

    # (Runner remains; it will be closed/ flipped by your normal opposite-signal logic elsewhere.)


def _filling_mode_candidates():
    modes = []
    if hasattr(mt5, "ORDER_FILLING_RETURN"): modes.append(mt5.ORDER_FILLING_RETURN)
    if hasattr(mt5, "ORDER_FILLING_FOK"):    modes.append(mt5.ORDER_FILLING_FOK)
    if hasattr(mt5, "ORDER_FILLING_IOC"):    modes.append(mt5.ORDER_FILLING_IOC)
    return modes or [mt5.ORDER_FILLING_IOC]

def send_order(symbol, action_type, lot=None):
    global _need_sl_seed_scan  # ensure we can flip the seed-scan flag

    # Prop-rule gate
    if prop.enforce_breaches():
        print("[BLOCK] Prop rule breached — order blocked.")
        return False

    # Symbol + trading checks
    if not mt5.symbol_select(symbol, True):
        print(f"[ERROR] Failed to select symbol {symbol}")
        return False
    si = mt5.symbol_info(symbol)
    if si is None:
        print(f"[ERROR] Symbol info not found for {symbol}")
        return False
    if hasattr(si, "trade_allowed") and not si.trade_allowed:
        print(f"[ERROR] Trading is not allowed for {symbol}")
        return False

    # Entry price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[ERROR] Failed to get tick data for {symbol}")
        return False
    entry_price = tick.ask if action_type == mt5.ORDER_TYPE_BUY else tick.bid

    # SL preview at snapshot price (for risk calc + pre-trade veto)
    sl_price_snapshot = compute_sl_price(symbol, action_type, entry_price, sl_usd_distance)
    if sl_price_snapshot is None:
        print("[ERROR] Could not compute SL price.")
        return False
    sl_distance = abs(entry_price - sl_price_snapshot)

    # Auto lot (risk vs initial account)
    if lot is None:
        lot, risk_amount = compute_lot_for_risk_static_initial(symbol, sl_distance, risk_per_trade_pct_of_initial)
        if lot is None or lot <= 0:
            print("[ERROR] Computed lot is invalid; aborting order.")
            return False
        print(f"[RISK] Initial risk: ${risk_amount:.2f} | SL distance: ${sl_distance:.2f} | Lot: {lot}")

    # Clamp lot to broker bounds
    lot = _round_to_step(lot, si.volume_step)
    lot = max(si.volume_min, min(lot, si.volume_max))
    if lot < si.volume_min or lot > si.volume_max:
        print(f"[ERROR] Lot size {lot} out of range: min={si.volume_min}, max={si.volume_max}")
        return False

    # Pre-trade worst-case loss veto (if we can infer per-$ value)
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
            print("[BLOCK] Order would breach prop rules (worst-case SL) — aborted.")
            return False
    else:
        print("[WARN] Could not derive per-$ value reliably; skipping pre-trade breach veto (live breaches still enforced).")

    # Base request
    request_base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": action_type,
        "price": entry_price,
        "deviation": slippage,
        "magic": magic_number,
        "comment": "CEBot-Prop",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    # Try available filling modes
    last_res = None
    for fm in _filling_mode_candidates():
        req = dict(request_base, type_filling=fm)
        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[OK] ORDER PLACED (fill_mode={fm}): ticket={res.order}, price={entry_price}, lot={lot}")

            # Find the position and attach SL
            pos = _get_position(symbol)
            if pos:
                filled = pos.price_open
                desired_sl = compute_sl_price(symbol, action_type, filled, sl_usd_distance)
                if desired_sl:
                    # Try once immediately; if it fails, queue for periodic 10s retries
                    if not _attempt_set_sl(pos, desired_sl, comment="Set SL post-fill"):
                        pending_sl[pos.ticket] = {"desired_sl": desired_sl, "last_try": None}
                        print(f"[INFO] SL queued for retry on ticket {pos.ticket}: target={desired_sl}")
            else:
                # Position not visible yet — seed the 10s scanner to attach SL asap
                print("[WARN] Position not found after fill; cannot set SL (will check later).")
                _need_sl_seed_scan = True

            return True

        last_res = res

    # If we get here, all fill modes failed
    print(f"[ERROR] ORDER FAILED across all filling modes.")
    if last_res:
        print(f"        retcode={last_res.retcode}, comment={last_res.comment}")
    else:
        print(f"        order_send returned None. last_error={mt5.last_error()}")
    return False


# ============================================================
# PENDING / RETRY EXECUTION HELPERS
# ============================================================
def _attempt_execution_for_signal(desired_side: str) -> bool:
    """
    Try to realize desired 'BUY'/'SELL' on mt5_symbol.
    Close opposite if needed, then place market order (auto-sized).
    Returns True if we already have the correct side OR we placed it now.
    """
    if prop.enforce_breaches():
        return False

    # stop trying if target already hit
    hit, _, _ = prop.profit_target_hit()
    if hit:
        return False

    position = _get_position(mt5_symbol)
    open_side = None
    if position:
        open_side = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'

    if open_side == desired_side:
        return True

    if open_side and open_side != desired_side:
        _close_position_ticket(position)
        time.sleep(0.5)  # small pause for server update

    order_type = mt5.ORDER_TYPE_BUY if desired_side == 'BUY' else mt5.ORDER_TYPE_SELL
    return bool(send_order(mt5_symbol, order_type))

def _maybe_retry_pending():
    """
    While waiting (countdowns / polling), try to execute any queued signal.
    Respects prop breaches and session windows, throttled by RETRY_EVERY_SECS.
    """
    global pending_signal, pending_since, last_retry_at

    if not pending_signal:
        return

    # stop retrying if target hit
    hit, _, _ = prop.profit_target_hit()
    if hit:
        return

    now_local = datetime.now(local_tz)

    # session guard
    if not in_session(now_local):
        return

    # throttle
    if last_retry_at and (now_local - last_retry_at).total_seconds() < RETRY_EVERY_SECS:
        return

    # prop rules guard
    if prop.enforce_breaches():
        return

    print(f"[RETRY] Pending {pending_signal} — attempting execution…")
    ok = _attempt_execution_for_signal(pending_signal)
    last_retry_at = now_local
    if ok:
        print(f"[OK] Pending {pending_signal} executed.")
        pending_signal = None
        pending_since  = None
        last_retry_at  = None

# ============================================================
# MAIN LOOP
# ============================================================
last_candle_time = None
current_session_start = None
current_session_end   = None
saw_candle_after_session_start = False

# --- one-time startup SL seeding (after helpers are defined!) ---
try:
    existing = _get_position(mt5_symbol)
    if existing and getattr(existing, "sl", 0) in (None, 0.0):
        _queue_initial_sl(existing)
except Exception as _e:
    print(f"[INIT] Startup SL seed check skipped: {_e}")

print("[BOOT] PropRules engine active. Phase:", current_phase,
      " | Targets → P1:", f"${prop.phase1_target:.0f}", " P2:", f"${prop.phase2_target:.0f}",
      " | Limits → Daily:", f"${prop.daily_limit:.0f}", " Overall:", f"${prop.overall_limit:.0f}")

# === ONE-OFF: optionally email yesterday→now daily report with CSV on startup ===
if FORCE_DAILY_REPORT_ON_START:
    force_daily_report_now()
# === END ONE-OFF ===

try:
    while True:
        # Block everything while breached; also clear any queued pendings on breach
        if prop.enforce_breaches():
            pending_signal = None
            pending_since  = None
            last_retry_at  = None
            time.sleep(1)
            continue

        # 1) Wait until inside a session
        while True:
            now_local = datetime.now(local_tz)
            prop.enforce_breaches()
            if in_session(now_local):
                current_session_start, current_session_end = current_session_bounds(now_local)
                saw_candle_after_session_start = False
                log(f"[+] IN SESSION: {current_session_start.strftime('%H:%M:%S %Z')}–{current_session_end.strftime('%H:%M:%S %Z')}")
                break
            time.sleep(1)

        # 2) Wait for the next 5-minute close (strictly future)
        next_close = countdown_to_next_5m(local_tz, until_dt=current_session_end, prop_obj=prop)

        # If we already have the candle time-stamped at next_close - 5m,
        # wait to the *following* close to avoid spam/futile polling.
        if last_candle_time is not None and last_candle_time >= (next_close - timedelta(minutes=5)):
            log("[SKIP] Just-closed candle already fetched. Waiting for the next new bar...")
            next_next_close = next_close + timedelta(minutes=5)
            if next_next_close < current_session_end:
                countdown_to(next_next_close, local_tz, prop_obj=prop)
            # after waiting to the NEXT bar, restart the loop cleanly
            continue

        # Session might have ended while waiting
        if datetime.now(local_tz) >= current_session_end:
            log("[INFO] SESSION ENDED BEFORE NEXT BAR CLOSE.")
            if auto_close_at_session_end:
                pos = _get_position(mt5_symbol)
                if pos:
                    log("[ACTION] CLOSING OPEN POSITION AT SESSION END…")
                    _close_position_ticket(pos)
            # clear pending for next session
            pending_signal = None
            pending_since  = None
            last_retry_at  = None
            continue

        # 3) Poll for a truly new candle (short timeout) while retrying pendings
        retry_timeout = timedelta(seconds=20)
        start_time = datetime.now(local_tz)
        df = None
        prev_last = last_candle_time

        while True:
            df = fetch_oanda_candles(symbol=oanda_symbol, granularity="M5", count=num_candles)
            if df is None or df.empty:
                time.sleep(2)
                _maybe_retry_pending()
                _periodic_sl_guard()
                if datetime.now(local_tz) - start_time > retry_timeout:
                    log("[TIMEOUT] No new candle. Skipping cycle.")
                    df = None
                    break
                continue

            latest_candle_time = df.index[-1]

            if datetime.now(local_tz) >= current_session_end:
                log("[INFO] SESSION ENDED DURING DATA WAIT.")
                if auto_close_at_session_end:
                    pos = _get_position(mt5_symbol)
                    if pos:
                        _close_position_ticket(pos)
                df = None
                break

            if prev_last is not None and latest_candle_time <= prev_last:
                log("[WAIT] No new candle yet. Retrying in 2s…")
                time.sleep(2)
                _maybe_retry_pending()
                _periodic_sl_guard()
                if datetime.now(local_tz) - start_time > retry_timeout:
                    log("[TIMEOUT] No new candle. Skipping cycle.")
                    df = None
                    break
                continue

            last_candle_time = latest_candle_time
            log(f"[OK] New 5-min candle: {latest_candle_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            break

        if df is None or df.empty:
            continue

        # Enforce first post-session-start bar if requested
        if require_post_start_candle and not saw_candle_after_session_start:
            if current_session_start and last_candle_time <= current_session_start:
                log("[WAIT] First bar after session open not closed yet; waiting for the next close…")
                continue
            saw_candle_after_session_start = True

        # 4) Print raw OANDA candles (last 10)
        try:
            raw_to_show = df[['open','high','low','close','volume']].copy()
            raw_to_show.index = raw_to_show.index.strftime('%Y-%m-%d %H:%M')
            # print("\n= = = = =   RAW OANDA CANDLESTICK DATA (LAST 10 CANDLES)   = = = = =")
            # print(raw_to_show.tail(10))
        except Exception as e:
            log(f"[DEBUG] Raw candle print failed: {e}")

        # 5) Compute indicators & print HA table (last 10)
        tr = calculate_indicators(df, useHeikinAshi=use_heikin_ashi, atrPeriod=atr_period, atrMult=atr_mult)
        latest = tr.iloc[-1]
        try:
            debug_df = tr[['ha_c','ha_open','ha_high','ha_low','dir','buy_signal','sell_signal']].copy()
            debug_df['signal'] = debug_df.apply(lambda row: 'BUY' if row['buy_signal'] else ('SELL' if row['sell_signal'] else ''), axis=1)
            debug_df.index = debug_df.index.strftime('%Y-%m-%d %H:%M')
            # print("\n= = = = =   LAST 10 HEIKIN-ASHI CANDLES WITH SIGNALS  = = = = =")
            # print(debug_df[['ha_c','ha_open','ha_high','ha_low','dir','signal']].tail(10))
        except Exception as e:
            log(f"[DEBUG] HA table print failed: {e}")

        # 6) Determine latest and previous-bar signals
        signal = 'BUY' if bool(latest['buy_signal']) else ('SELL' if bool(latest['sell_signal']) else None)

        prev_signal = None
        if len(tr) >= 2:
            prev = tr.iloc[-2]
            if prev['buy_signal']:
                prev_signal = 'BUY'
            elif prev['sell_signal']:
                prev_signal = 'SELL'

        # 7) Position state
        position = _get_position(mt5_symbol)
        open_side = None
        if position:
            open_side = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'
            print(f"[INFO] OPEN POSITION: {open_side}, vol: {position.volume}, entry: {position.price_open}")
        else:
            print("[INFO] No open position currently.")

        if position and position.ticket not in pos_state:
            tp1_preview = compute_tp_price(position.symbol, position.type, position.price_open, tp1_usd_distance)
            tp2_preview = compute_tp_price(position.symbol, position.type, position.price_open,
                                           tp2_usd_distance) if tp2_fraction else None
            if tp2_preview is not None:
                print(
                    f"[INFO] Managing ticket {position.ticket}: entry={position.price_open}, "
                    f"TP1({int(tp1_fraction * 100)}%)={tp1_preview}, TP2({int(tp2_fraction * 100)}%)={tp2_preview}")
                pos_state[position.ticket] = {"partial50_done": False, "partial25_done": False, "moved_to_be": False}
            else:
                print(
                    f"[INFO] Managing ticket {position.ticket}: entry={position.price_open}, "
                    f"TP1({int(tp1_fraction * 100)}%)={tp1_preview}")
                pos_state[position.ticket] = {"partial50_done": False, "moved_to_be": False}

        # If position has no SL, queue it now
        if position and getattr(position, "sl", 0) in (None, 0.0) and position.ticket not in pending_sl:
            _queue_initial_sl(position)

        # Manage partial TP and BE (intrabar + per-cycle)
        _maybe_manage_open_position()

        # 8) End-of-session guard
        if datetime.now(local_tz) >= current_session_end:
            log("[INFO] SESSION ENDED POST-CALC.")
            if auto_close_at_session_end and position:
                _close_position_ticket(position)
            # Clear pending for next session
            pending_signal = None
            pending_since  = None
            last_retry_at  = None
            continue

        # 9) Profit target gate (halt new trades & stop retries)
        hit, gain, target = prop.profit_target_hit()
        if hit:
            print(f"[TARGET] Phase {prop.phase} target hit: Gain ${gain:.2f} ≥ ${target:.2f}. Halting new trades.")
            if alerts_enabled and alerter:
                alerter.phase_passed(prop.phase, gain, target)
            pending_signal = None
            pending_since  = None
            last_retry_at  = None
            time.sleep(1)
            continue

        # 10) Execute with pending/backfill/override
        now_local = datetime.now(local_tz)

        if signal:
            # If a pending exists and the live signal is opposite, override it
            if pending_signal and signal != pending_signal:
                print(f"[CANCELLED] Live signal {signal} overrides pending {pending_signal}.")
                ok = _attempt_execution_for_signal(signal)
                if ok:
                    pending_signal = None
                    pending_since  = None
                    last_retry_at  = None
                else:
                    pending_signal = signal
                    if pending_since is None:
                        pending_since = now_local
            else:
                # Normal path: act only if position isn’t already on the signal side
                if open_side != signal:
                    print_daily_risk_diag()
                    print(f"[TRADE] New signal={signal} | open={open_side or 'NONE'}")
                    ok = _attempt_execution_for_signal(signal)
                    if ok:
                        pending_signal = None
                        pending_since  = None
                        last_retry_at  = None
                    else:
                        if not pending_signal:
                            pending_signal = signal
                            pending_since  = now_local
                            print(f"[PENDING] Queued {signal} execution (will retry).")
        else:
            # No fresh latest-bar signal → consider backfill of previous bar
            if prev_signal and open_side != prev_signal and not pending_signal:
                print(f"[BACKFILL] Previous bar had {prev_signal} — attempting execution now.")
                ok = _attempt_execution_for_signal(prev_signal)
                if ok:
                    pending_signal = None
                    pending_since  = None
                    last_retry_at  = None
                else:
                    pending_signal = prev_signal
                    pending_since  = now_local
                    print(f"[PENDING] Queued {prev_signal} from previous bar (will retry).")
            else:
                print("[INFO] No actionable signal.")

        # 11) Final session guard
        if datetime.now(local_tz) >= current_session_end:
            log("[INFO] SESSION JUST ENDED.")
            pending_signal = None
            pending_since  = None
            last_retry_at  = None
            if auto_close_at_session_end:
                position = _get_position(mt5_symbol)
                if position:
                    _close_position_ticket(position)
            continue

except KeyboardInterrupt:
    log("[INFO] Stopped by user (CTRL-C).")
finally:
    try:
        mt5.shutdown()
        log("[INFO] MT5 shutdown complete.")
    except Exception as e:
        log(f"[WARN] MT5 shutdown error: {e}")
