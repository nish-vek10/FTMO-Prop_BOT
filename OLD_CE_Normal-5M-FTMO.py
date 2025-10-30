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
magic_number      = 123456

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
    ("06:00", "12:00"),  # London morning
    ("13:00", "18:00"),  # US session + London/NY overlap
]
auto_close_at_session_end   = True
require_post_start_candle   = True

# === ALERTS & REPORTS (EMAIL OPTIONAL) === #
alerts_enabled           = True
email_enabled            = True

report_dir               = r"C:\Users\anish\OneDrive\Desktop\Anish\A - EAs Reports"

smtp_host = "smtp.gmail.com"
smtp_port = 587
smtp_user = "anishv2610@gmail.com"
smtp_pass = "lpignmmkhgymwlpi"
email_to  = ["ea.prop0925@gmail.com"]

# === STARTUP REPORT TOGGLE ===
# If True: send a "yesterday → now" FTMO daily report + CSV ON STARTUP (useful for testing).
# Set to False once you’re done testing so it won’t email every time you restart.
FORCE_DAILY_REPORT_ON_START = False


# === ACCOUNT LOGIN CONFIG === #
mt5_login = 52498279
mt5_password = "eQSX7dfh!nIZOy"
mt5_server = "ICMarketsSC-Demo"
mt5_terminal_path = r"C:\MT5\EA-FTMO\terminal64.exe"

# === OANDA CONFIG (simple requests) === #
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
    msg['Subject'] = 'SMTP test — PropEA'
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
        subject = f"FTMO Daily Report — {date_label}"
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
            out_path = os.path.join(report_dir, f'FTMO_daily_{day_str}.csv')
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
    base = now.replace(second=0, microsecond=0)
    mins_to_add = (5 - (base.minute % 5)) % 5
    if mins_to_add == 0 and now.second == 0 and now.microsecond == 0:
        mins_to_add = 5
    return base + timedelta(minutes=mins_to_add)

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
        now = datetime.now(tz)
        if (target - now).total_seconds() <= 0:
            break
        time.sleep(0.25)
    time.sleep(2)
    return target

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

def _filling_mode_candidates():
    modes = []
    if hasattr(mt5, "ORDER_FILLING_RETURN"): modes.append(mt5.ORDER_FILLING_RETURN)
    if hasattr(mt5, "ORDER_FILLING_FOK"):    modes.append(mt5.ORDER_FILLING_FOK)
    if hasattr(mt5, "ORDER_FILLING_IOC"):    modes.append(mt5.ORDER_FILLING_IOC)
    return modes or [mt5.ORDER_FILLING_IOC]

def send_order(symbol, action_type, lot=None):
    if prop.enforce_breaches():
        print("[BLOCK] Prop rule breached — order blocked.")
        return

    if not mt5.symbol_select(symbol, True):
        print(f"[ERROR] Failed to select symbol {symbol}")
        return
    si = mt5.symbol_info(symbol)
    if si is None:
        print(f"[ERROR] Symbol info not found for {symbol}")
        return
    if hasattr(si, "trade_allowed") and not si.trade_allowed:
        print(f"[ERROR] Trading is not allowed for {symbol}")
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[ERROR] Failed to get tick data for {symbol}")
        return
    entry_price = tick.ask if action_type == mt5.ORDER_TYPE_BUY else tick.bid

    sl_price_snapshot = compute_sl_price(symbol, action_type, entry_price, sl_usd_distance)
    if sl_price_snapshot is None:
        print("[ERROR] Could not compute SL price.")
        return
    sl_distance = abs(entry_price - sl_price_snapshot)

    if lot is None:
        lot, risk_amount = compute_lot_for_risk_static_initial(symbol, sl_distance, risk_per_trade_pct_of_initial)
        if lot is None or lot <= 0:
            print("[ERROR] Computed lot is invalid; aborting order.")
            return
        print(f"[RISK] Initial risk: ${risk_amount:.2f} | SL distance: ${sl_distance:.2f} | Lot: {lot}")

    lot = _round_to_step(lot, si.volume_step)
    lot = max(si.volume_min, min(lot, si.volume_max))
    if lot < si.volume_min or lot > si.volume_max:
        print(f"[ERROR] Lot size {lot} out of range: min={si.volume_min}, max={si.volume_max}")
        return

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
            return
    else:
        print("[WARN] Could not derive per-$ value reliably; skipping pre-trade breach veto (live breaches still enforced).")

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

    last_res = None
    for fm in _filling_mode_candidates():
        req = dict(request_base, type_filling=fm)
        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[OK] ORDER PLACED (fill_mode={fm}): ticket={res.order}, price={entry_price}, lot={lot}")
            pos = _get_position(symbol)
            if pos:
                filled = pos.price_open
                desired_sl = compute_sl_price(symbol, action_type, filled, sl_usd_distance)
                if desired_sl:
                    mod = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "position": pos.ticket,
                        "sl": desired_sl,
                        "magic": magic_number,
                        "comment": "Set SL post-fill",
                    }
                    mod_res = mt5.order_send(mod)
                    if mod_res is not None and mod_res.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"[OK] SL SET to {desired_sl}")
                    else:
                        print(f"[WARN] SL set failed: {getattr(mod_res,'retcode',None)}, {getattr(mod_res,'comment',None)}")
            else:
                print("[WARN] Position not found after fill; cannot set SL.")
            return
        last_res = res

    print(f"[ERROR] ORDER FAILED across all filling modes.")
    if last_res:
        print(f"        retcode={last_res.retcode}, comment={last_res.comment}")
    else:
        print(f"        order_send returned None. last_error={mt5.last_error()}")

# ============================================================
# MAIN LOOP
# ============================================================
last_candle_time = None
current_session_start = None
current_session_end   = None
saw_candle_after_session_start = False

print("[BOOT] PropRules engine active. Phase:", current_phase,
      " | Targets → P1:", f"${prop.phase1_target:.0f}", " P2:", f"${prop.phase2_target:.0f}",
      " | Limits → Daily:", f"${prop.daily_limit:.0f}", " Overall:", f"${prop.overall_limit:.0f}")

# === ONE-OFF: optionally email yesterday→now daily report with CSV on startup ===
if FORCE_DAILY_REPORT_ON_START:
    force_daily_report_now()
# === END ONE-OFF ===


while True:
    if prop.enforce_breaches():
        time.sleep(1)
        continue

    while True:
        now_local = datetime.now(local_tz)
        prop.enforce_breaches()
        if in_session(now_local):
            current_session_start, current_session_end = current_session_bounds(now_local)
            saw_candle_after_session_start = False
            log(f"[+] IN SESSION: {current_session_start.strftime('%H:%M:%S %Z')}–{current_session_end.strftime('%H:%M:%S %Z')}")
            break
        time.sleep(1)

    next_close = countdown_to_next_5m(local_tz, until_dt=current_session_end, prop_obj=prop)

    if datetime.now(local_tz) >= current_session_end:
        log("[INFO] SESSION ENDED BEFORE NEXT BAR CLOSE.")
        if auto_close_at_session_end:
            pos = _get_position(mt5_symbol)
            if pos:
                log("[ACTION] CLOSING OPEN POSITION AT SESSION END…")
                _close_position_ticket(pos)
        continue

    retry_timeout = timedelta(seconds=20)
    start_time = datetime.now(local_tz)
    df = None
    prev_last = last_candle_time

    while True:
        df = fetch_oanda_candles(symbol=oanda_symbol, granularity="M5", count=num_candles)
        if df is None or df.empty:
            time.sleep(2)
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
            log("[SKIP] Just-closed candle already fetched. Waiting to the next close for a brand-new timestamp...")
            df = None
            break

        last_candle_time = latest_candle_time
        log(f"[OK] New 5-min candle: {latest_candle_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        break

    if df is None or df.empty:
        continue

    # Print raw OANDA candles (last 10)
    try:
        raw_to_show = df[['open','high','low','close','volume']].copy()
        raw_to_show.index = raw_to_show.index.strftime('%Y-%m-%d %H:%M')
        print("\n= = = = =   RAW OANDA CANDLESTICK DATA (LAST 10 CANDLES)   = = = = =")
        print(raw_to_show.tail(10))
    except Exception as e:
        log(f"[DEBUG] Raw candle print failed: {e}")

    # Compute indicators & print HA table (last 10)
    tr = calculate_indicators(df, useHeikinAshi=use_heikin_ashi, atrPeriod=atr_period, atrMult=atr_mult)
    latest = tr.iloc[-1]
    try:
        debug_df = tr[['ha_c','ha_open','ha_high','ha_low','dir','buy_signal','sell_signal']].copy()
        debug_df['signal'] = debug_df.apply(lambda row: 'BUY' if row['buy_signal'] else ('SELL' if row['sell_signal'] else ''), axis=1)
        debug_df.index = debug_df.index.strftime('%Y-%m-%d %H:%M')
        print("\n= = = = =   LAST 10 HEIKIN-ASHI CANDLES WITH SIGNALS  = = = = =")
        print(debug_df[['ha_c','ha_open','ha_high','ha_low','dir','signal']].tail(10))
    except Exception as e:
        log(f"[DEBUG] HA table print failed: {e}")

    # Determine signal
    signal = None
    if latest['buy_signal']:
        signal = 'BUY'
    elif latest['sell_signal']:
        signal = 'SELL'

    # Profit target gate
    hit, gain, target = prop.profit_target_hit()
    if hit:
        print(f"[TARGET] Phase {prop.phase} target hit: Gain ${gain:.2f} ≥ ${target:.2f}. Halting new trades.")
        if alerts_enabled and alerter:
            alerter.phase_passed(prop.phase, gain, target)
        time.sleep(1)
        continue

    # Position state
    position = _get_position(mt5_symbol)
    open_side = None
    if position:
        open_side = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'
        print(f"[INFO] OPEN POSITION: {open_side}, vol: {position.volume}, entry: {position.price_open}")
    else:
        print("[INFO] No open position currently.")

    if datetime.now(local_tz) >= current_session_end:
        log("[INFO] SESSION ENDED POST-CALC.")
        if auto_close_at_session_end and position:
            _close_position_ticket(position)
        continue

    # Execute per signal (with prop guards)
    if signal == 'BUY':
        if open_side == 'SELL' and position:
            _close_position_ticket(position)
        if open_side != 'BUY':
            print_daily_risk_diag()
            print("[EXEC] BUY signal → sending order (auto-sized).")
            send_order(mt5_symbol, mt5.ORDER_TYPE_BUY)

    elif signal == 'SELL':
        if open_side == 'BUY' and position:
            _close_position_ticket(position)
        if open_side != 'SELL':
            print_daily_risk_diag()
            print("[EXEC] SELL signal → sending order (auto-sized).")
            send_order(mt5_symbol, mt5.ORDER_TYPE_SELL)

    if datetime.now(local_tz) >= current_session_end:
        log("[INFO] SESSION JUST ENDED.")
        if auto_close_at_session_end:
            position = _get_position(mt5_symbol)
            if position:
                _close_position_ticket(position)
        continue
