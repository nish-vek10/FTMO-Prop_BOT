"""
Chandelier Exit Strategy for XAUUSD — FTMO-Style 2-Step Prop Rules (Dual TF)
Heikin-Ashi + Chandelier | XAUUSD | M5 + H1

Key changes vs prior:
- TP1 only: close 60%, then move SL -> BE (+buffer), leave 40% runner until opposite signal
- Dual timeframes running simultaneously (M5 + H1), each with its own:
    • Heikin-Ashi toggle, ATR period/mult
    • SL distance, TP1 distance, TP1 fraction
    • Risk percent (of CURRENT BALANCE)
    • Sessions
    • Magic number, order comments, pending/retry state, per-position management state
- Risk sizing uses CURRENT ACCOUNT BALANCE (not initial). M5: 0.25%, H1: 0.50%
- Clean, explicit *_M5 / *_H1 variables, clear logs, guarded retries

Tested structure-wise to mirror your prior working flow.
"""

import os
import sys
import time
import math
from datetime import datetime, timedelta
import pandas as pd
from pytz import timezone, UTC

import requests
import MetaTrader5 as mt5

from email.message import EmailMessage
from pathlib import Path

# ========== USER CONFIG: MARKETS & PATHS ==========

mt5_symbol           = "XAUUSD"
oanda_symbol         = "XAU_USD"
num_candles_M5       = 500
num_candles_H1       = 500

# OANDA API
oanda_token          = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
oanda_api_base       = "https://api-fxpractice.oanda.com/v3"

# Terminal + email (unchanged)
report_dir           = r"C:\Users\anish\OneDrive\Desktop\Anish\A - EAs Reports"
smtp_host            = "smtp.gmail.com"
smtp_port            = 587
smtp_user            = "anishv2610@gmail.com"
smtp_pass            = "lpignmmkhgymwlpi"
email_to             = ["ea.prop0925@gmail.com"]
FORCE_DAILY_REPORT_ON_START = False

# MT5 login
mt5_login            = 52548998
mt5_password         = "rEwFf!e@73da8j"
mt5_server           = "ICMarketsSC-Demo"
mt5_terminal_path    = r"C:\MT5\EA_XAU-MultiTF-5M+1H\terminal64.exe"

# Local/logging TZ
local_tz             = timezone('Europe/London')

# ========== GLOBAL ENGINE SETTINGS ==========
slippage             = 10
alerts_enabled       = True
email_enabled        = True
RETRY_EVERY_SECS     = 15

# ========== PROP RULES (initial-account based as before) ==========
account_size_usd     = 100_000.00
phase1_target_pct    = 10.0
phase2_target_pct    = 5.0
max_daily_loss_pct   = 5.0
max_overall_loss_pct = 10.0

prop_reset_tz        = timezone('Europe/Prague')
prop_reset_hour      = 0
prop_reset_minute    = 0
current_phase        = 1
auto_close_on_breach = True

# ========== TIMEFRAME-SPECIFIC CONFIGS ==========

# ---- M5 CONFIG ----
tag_M5                  = "M5"
magic_M5                = 114477
use_heikin_ashi_M5      = True
atr_period_M5           = 1
atr_mult_M5             = 1.85
sl_usd_distance_M5      = 5.5        # $ distance
tp1_usd_distance_M5     = 7.5        # $ distance for first (and only) partial
tp1_fraction_M5         = 0.60       # close 60%, leave 40% runner
breakeven_buffer_usd_M5 = 0.20       # BE buffer
risk_pct_of_balance_M5  = 0.25       # % of CURRENT balance per trade

# Separate sessions for M5 (edit as you like)
session_windows_M5 = [
    ("06:00", "12:00"),
    ("13:00", "18:00"),
]
require_post_start_candle_M5 = True
auto_close_at_session_end_M5 = True

# ---- H1 CONFIG ----
tag_H1                  = "H1"
magic_H1                = 114478
use_heikin_ashi_H1      = True
atr_period_H1           = 1
atr_mult_H1             = 1.85
sl_usd_distance_H1      = 20.0       # $ distance
tp1_usd_distance_H1     = 25.0       # $ distance for first (and only) partial
tp1_fraction_H1         = 0.60       # close 60%, leave 40% runner
breakeven_buffer_usd_H1 = 0.50       # BE buffer
risk_pct_of_balance_H1  = 0.50       # % of CURRENT balance per trade

# Separate sessions for H1 (placeholder; edit as you like)
# session_windows_H1 = [
#     ("00:00", "23:59"),
# ]

session_windows_H1 = None
require_post_start_candle_H1 = False     # optional: avoids waiting for a "first after open"
auto_close_at_session_end_H1 = False     # optional: don't auto-close at an artificial "end"

# ========== LOGGING ==========
def log(msg: str):
    ts = datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts} | {msg}", flush=True)

# ========== BOOT: MT5 + REPORT DIR ==========
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
    print(f"[INFO] Approx. min stop distance enforced by broker ≈ ${min_sl_usd:.2f}")
else:
    print(f"[ERROR] Unable to fetch symbol info for {mt5_symbol}")

Path(report_dir).mkdir(parents=True, exist_ok=True)

# ========== EMAIL SMOKE TEST ==========
def _email_smoke_test():
    msg = EmailMessage()
    msg['Subject'] = 'SMTP test — PropEA - Multi Timeframe Report'
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

# ========== OANDA FETCH ==========
def fetch_oanda_candles(granularity: str, count: int):
    """
    Fetch candle data from OANDA REST API using requests.get.
    Returns a pandas DataFrame indexed by local_tz time.
    granularity: "M5" or "H1" (OANDA codes)
    """
    url = f"{oanda_api_base}/instruments/{oanda_symbol}/candles"
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
    df = pd.DataFrame(data).set_index("time")
    return df

# ========== ALERTER ==========
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
        subject = "MULTI TF BREACH — Max Daily Loss"
        body = (f"*** DAILY BREACH ***\n"
                f"Closed today: ${closed:,.2f}\nOpen PnL: ${openp:,.2f}\n"
                f"Daily sum (C+O): ${closed+openp:,.2f} <= -${limit:,.2f}\n"
                f"Reset TZ: {tzname}\n")
        self._send_email(subject, body)

    def breach_overall(self, equity: float, min_equity: float):
        if self._overall_breach_sent:
            return
        self._overall_breach_sent = True
        subject = "MULTI TF BREACH — Max Overall Loss"
        body = (f"*** OVERALL BREACH ***\n"
                f"Equity: ${equity:,.2f}\nRequired minimum equity: ${min_equity:,.2f}\n")
        self._send_email(subject, body)

    def phase_passed(self, phase: int, gain: float, target: float):
        if phase in self._phase_sent:
            return
        self._phase_sent.add(phase)
        subject = f"MULTI TF Phase {phase} PASSED"
        body = (f"Congratulations — Phase {phase} target reached!\n"
                f"Gain: ${gain:,.2f} >= Target: ${target:,.2f}\n")
        self._send_email(subject, body)

    def daily_report(self, csv_path: str, net_pnl: float, date_label: str, summary: str = None):
        subject = f"MULTI TF Daily Report — {date_label}"
        body = (summary or "") + f"\nNet PnL (Closed+Open at send time): ${net_pnl:,.2f}\nFile: {csv_path}\n"
        self._send_email(subject, body, attachments=[csv_path] if csv_path else None)

# ========== PROP RULES ==========
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
            out_path = os.path.join(report_dir, f'MULTI-TF_daily_{day_str}.csv')
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
                    _close_position_ticket(p, magic=int(getattr(p, "magic", 0)),
                                           tag=("M5" if int(getattr(p,"magic",0))==magic_M5 else "H1"))
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

# ========== UTILS ==========
def force_daily_report_now():
    try:
        now_tz = datetime.now(prop_reset_tz)
        start_tz = now_tz - timedelta(days=1)
        print(f"[REPORT] Forcing daily report window: {start_tz.strftime('%Y-%m-%d %H:%M %Z')} → {now_tz.strftime('%Y-%m-%d %H:%M %Z')}")
        prop._send_daily_report_for_window(start_tz, now_tz)
    except Exception as e:
        print(f"[REPORT] Startup forced daily report failed: {e}")

def _round_to_step(value, step):
    steps = math.floor(value / step)
    return max(steps * step, 0.0)

def _make_dt(base_dt, hhmm, tz):
    h, m = map(int, hhmm.split(":"))
    return base_dt.astimezone(tz).replace(hour=h, minute=m, second=0, microsecond=0)

def _in_session(now_local, session_windows):
    if session_windows is None:
        return True

    for start, end in session_windows:
        start_dt = _make_dt(now_local, start, local_tz)
        end_dt   = _make_dt(now_local, end,   local_tz)
        if start_dt <= now_local < end_dt:
            return True
    return False

def _current_session_bounds(now_local, session_windows):
    if session_windows is None:
        return (None, None)

    for start, end in session_windows:
        start_dt = _make_dt(now_local, start, local_tz)
        end_dt   = _make_dt(now_local, end,   local_tz)
        if start_dt <= now_local < end_dt:
            return start_dt, end_dt
    return None, None

def _next_close_time(now, granularity: str):
    base = now.replace(second=0, microsecond=0)
    if granularity == "M5":
        mins_to_add = (5 - (base.minute % 5)) % 5
        target = base + timedelta(minutes=mins_to_add)
        if target <= now:
            target += timedelta(minutes=5)
        return target
    elif granularity == "H1":
        target = base.replace(minute=0) + timedelta(hours=1)
        if target <= now:
            target += timedelta(hours=1)
        return target
    else:
        return base + timedelta(minutes=1)  # safe default

def _prev_close_time(now, granularity: str):
    base = now.replace(second=0, microsecond=0)
    if granularity == "M5":
        minute = (base.minute // 5) * 5
        return base.replace(minute=minute)
    elif granularity == "H1":
        return base.replace(minute=0)
    else:
        return base

def _price_side_touched(tick, target_price, order_type):
    # order_type is mt5.ORDER_TYPE_BUY/SELL for the OPENED position side
    if order_type == mt5.ORDER_TYPE_BUY:
        return tick.bid >= target_price
    return tick.ask <= target_price

# ========== INDICATORS ==========
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
        if pd.isna(row['c_prev']): return row['h'] - row['l']
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

# ========== POSITION / ORDER HELPERS ==========
def _get_position(symbol, magic):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    for p in positions:
        if int(getattr(p, "magic", 0)) == int(magic):
            return p
    return None

def _round_volume_to_step(si, vol):
    vol = math.floor(vol / si.volume_step) * si.volume_step
    return max(si.volume_min, min(vol, si.volume_max))

def _close_position_ticket(position, magic, tag):
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
        "magic": magic,
        "comment": f"PropRules Close {tag}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Failed to close position ({tag}): {getattr(res, 'retcode', None)}, {getattr(res, 'comment', None)}")
    else:
        print(f"[OK] POSITION CLOSED ({tag}): {res}")

def compute_sl_price(symbol, action_type, entry_price, sl_usd):
    si = mt5.symbol_info(symbol)
    if si is None: return None
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

def compute_tp_price(symbol, action_type, entry_price, tp_usd):
    si = mt5.symbol_info(symbol)
    if si is None: return None
    digits = si.digits
    if action_type == mt5.ORDER_TYPE_BUY:
        tp = entry_price + float(tp_usd)
    else:
        tp = entry_price - float(tp_usd)
    return round(tp, digits)

def _partial_close(position, fraction, magic, tag):
    si = mt5.symbol_info(position.symbol)
    if si is None: return False
    part_vol_raw = position.volume * float(fraction)
    part_vol = _round_volume_to_step(si, part_vol_raw)

    remain_vol = round(position.volume - part_vol, 8)
    if remain_vol > 0 and remain_vol < si.volume_min:
        min_keep = si.volume_min
        part_vol = _round_volume_to_step(si, position.volume - min_keep)
        remain_vol = round(position.volume - part_vol, 8)
    if part_vol <= 0: return False

    opp_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None: return False
    close_price = tick.bid if opp_type == mt5.ORDER_TYPE_SELL else tick.ask

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": part_vol,
        "type": opp_type,
        "position": position.ticket,
        "price": close_price,
        "deviation": slippage,
        "magic": magic,
        "comment": f"PartialTP60-{tag}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[OK] Partial {tag} closed {part_vol} from ticket {position.ticket}")
        return True
    print(f"[WARN] Partial close ({tag}) failed: {getattr(res,'retcode',None)} {getattr(res,'comment',None)}")
    return False

def _move_sl_to_breakeven(position, buffer_usd, magic, tag):
    symbol = position.symbol
    si = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if si is None or tick is None: return False

    entry = float(position.price_open)
    point = si.point
    stops_pts = getattr(si, "trade_stops_level", 0) or 0
    min_dist = stops_pts * point

    spread = (tick.ask - tick.bid)
    be_buf = max(float(buffer_usd), spread)

    if position.type == mt5.ORDER_TYPE_BUY:
        desired_sl = round(entry + be_buf, si.digits)
        max_allowed_sl = tick.bid - min_dist if min_dist > 0 else tick.bid
        if desired_sl <= max_allowed_sl and desired_sl < tick.bid:
            mod = {"action": mt5.TRADE_ACTION_SLTP, "symbol": symbol, "position": position.ticket,
                   "sl": desired_sl, "magic": magic, "comment": f"SL->BE-{tag}"}
            res = mt5.order_send(mod)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[OK] SL moved to BE+buffer at {desired_sl} ({tag} long).")
                return True
            print(f"[WARN] BE SL set failed ({tag}): {getattr(res,'retcode',None)} {getattr(res,'comment',None)}")
            return False
        return False
    else:
        desired_sl = round(entry - be_buf, si.digits)
        min_allowed_sl = tick.ask + min_dist if min_dist > 0 else tick.ask
        if desired_sl >= min_allowed_sl and desired_sl > tick.ask:
            mod = {"action": mt5.TRADE_ACTION_SLTP, "symbol": symbol, "position": position.ticket,
                   "sl": desired_sl, "magic": magic, "comment": f"SL->BE-{tag}"}
            res = mt5.order_send(mod)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[OK] SL moved to BE+buffer at {desired_sl} ({tag} short).")
                return True
            print(f"[WARN] BE SL set failed ({tag}): {getattr(res,'retcode',None)} {getattr(res,'comment',None)}")
            return False
        return False

# ========== RISK / SIZING (CURRENT BALANCE) ==========
def _value_per_1usd_per_lot(si):
    if getattr(si, "trade_tick_value", 0) and getattr(si, "trade_tick_size", 0):
        return float(si.trade_tick_value) / float(si.trade_tick_size)
    if getattr(si, "tick_value", 0) and getattr(si, "tick_size", 0):
        return float(si.tick_value) / float(si.tick_size)
    if getattr(si, "tick_value", 0) and getattr(si, "point", 0):
        return float(si.tick_value) / float(si.point)
    if getattr(si, "trade_contract_size", 0):
        return float(si.trade_contract_size)
    return None

def compute_lot_for_risk_dynamic_balance(symbol, sl_usd, risk_pct_of_balance: float):
    si = mt5.symbol_info(symbol)
    acc = mt5.account_info()
    if si is None or acc is None:
        print("[ERROR] Sizing: symbol or account info not available.")
        return None, 0.0

    balance = float(acc.balance)
    risk_amount = balance * (float(risk_pct_of_balance) / 100.0)
    if risk_amount <= 0:
        return None, 0.0

    v = _value_per_1usd_per_lot(si)
    if not v:
        print("[ERROR] Cannot derive $ value per $1 move for 1 lot; missing tick fields.")
        return None, 0.0

    risk_per_lot = v * float(sl_usd)
    if risk_per_lot <= 0:
        return None, 0.0

    raw_lot = risk_amount / risk_per_lot
    lot = _round_to_step(raw_lot, si.volume_step)
    lot = max(si.volume_min, min(lot, si.volume_max))
    return lot, risk_amount

# ========== ORDER SEND ==========
def send_order(symbol, action_type, *, magic, tag, sl_usd, risk_pct_of_balance):
    if prop.enforce_breaches():
        print(f"[BLOCK] ({tag}) Prop rule breached — order blocked.")
        return False

    if not mt5.symbol_select(symbol, True):
        print(f"[ERROR] ({tag}) Failed to select symbol {symbol}")
        return False
    si = mt5.symbol_info(symbol)
    if si is None:
        print(f"[ERROR] ({tag}) Symbol info not found for {symbol}")
        return False
    if hasattr(si, "trade_allowed") and not si.trade_allowed:
        print(f"[ERROR] ({tag}) Trading is not allowed for {symbol}")
        return False

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[ERROR] ({tag}) Failed to get tick data for {symbol}")
        return False
    entry_price = tick.ask if action_type == mt5.ORDER_TYPE_BUY else tick.bid

    sl_price_snapshot = compute_sl_price(symbol, action_type, entry_price, sl_usd)
    if sl_price_snapshot is None:
        print(f"[ERROR] ({tag}) Could not compute SL price.")
        return False
    sl_distance = abs(entry_price - sl_price_snapshot)

    lot, risk_amount = compute_lot_for_risk_dynamic_balance(symbol, sl_distance, risk_pct_of_balance)
    if lot is None or lot <= 0:
        print(f"[ERROR] ({tag}) Computed lot is invalid; aborting order.")
        return False
    print(f"[RISK] ({tag}) Risk: {risk_pct_of_balance:.2f}% of BAL = ${risk_amount:,.2f} | SL dist: ${sl_distance:.2f} | Lot: {lot}")

    v_per_1usd = _value_per_1usd_per_lot(si)
    if v_per_1usd:
        worst_case_loss = v_per_1usd * sl_distance * float(lot)
        if prop.would_breach_with_order(worst_case_loss):
            print(f"[BLOCK] ({tag}) Order would breach prop rules (worst-case SL) — aborted.")
            return False
    else:
        print(f"[WARN] ({tag}) Could not derive per-$ value reliably; skipping pre-trade veto.")

    req_base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": action_type,
        "price": entry_price,
        "deviation": slippage,
        "magic": magic,
        "comment": f"CEBot-Prop-{tag}",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    last_res = None
    # Try several filling modes
    modes = []
    if hasattr(mt5, "ORDER_FILLING_RETURN"): modes.append(mt5.ORDER_FILLING_RETURN)
    if hasattr(mt5, "ORDER_FILLING_FOK"):    modes.append(mt5.ORDER_FILLING_FOK)
    if hasattr(mt5, "ORDER_FILLING_IOC"):    modes.append(mt5.ORDER_FILLING_IOC)
    if not modes: modes = [mt5.ORDER_FILLING_IOC]

    for fm in modes:
        req = dict(req_base, type_filling=fm)
        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[OK] ORDER PLACED ({tag}, fill_mode={fm}): ticket={res.order}, price={entry_price}, lot={lot}")
            pos = _get_position(symbol, magic)
            if pos:
                filled = pos.price_open
                desired_sl = compute_sl_price(symbol, action_type, filled, sl_usd)
                if desired_sl:
                    mod = {"action": mt5.TRADE_ACTION_SLTP, "symbol": symbol, "position": pos.ticket,
                           "sl": desired_sl, "magic": magic, "comment": f"Set SL post-fill {tag}"}
                    mod_res = mt5.order_send(mod)
                    if mod_res is not None and mod_res.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"[OK] SL SET ({tag}) to {desired_sl}")
                    else:
                        print(f"[WARN] SL set failed ({tag}): {getattr(mod_res,'retcode',None)}, {getattr(mod_res,'comment',None)}")
            else:
                print(f"[WARN] Position ({tag}) not found after fill; cannot set SL.")
            return True
        last_res = res

    print(f"[ERROR] ORDER FAILED ({tag}) across all filling modes.")
    if last_res:
        print(f"        retcode={last_res.retcode}, comment={last_res.comment}")
    else:
        print(f"        order_send returned None. last_error={mt5.last_error()}")
    return False

# ========== PER-TF STATE ==========
pending_M5  = {"signal": None, "since": None, "last_retry": None}
pending_H1  = {"signal": None, "since": None, "last_retry": None}
pos_state_M5 = {}  # ticket -> {"partial60_done": bool, "moved_to_be": bool}
pos_state_H1 = {}

last_candle_time_M5 = None
last_candle_time_H1 = None
saw_after_open_M5   = False
saw_after_open_H1   = False
sess_start_M5 = None; sess_end_M5 = None
sess_start_H1 = None; sess_end_H1 = None

# ========== MANAGEMENT (TP1 60% + SL->BE, NO TP2) ==========
def _maybe_manage_open_position(tag, magic, tp1_distance, tp1_fraction, be_buffer):
    if prop.enforce_breaches():
        return

    position = _get_position(mt5_symbol, magic)
    if not position:
        return

    tick = mt5.symbol_info_tick(position.symbol)
    si   = mt5.symbol_info(position.symbol)
    if tick is None or si is None:
        return

    state_store = pos_state_M5 if tag == tag_M5 else pos_state_H1
    ticket = position.ticket
    state = state_store.get(ticket, {"partial60_done": False, "moved_to_be": False})

    tp1_price = compute_tp_price(position.symbol, position.type, position.price_open, tp1_distance)
    if tp1_price is None:
        return

    tp1_hit = _price_side_touched(tick, tp1_price, position.type)

    if not state["partial60_done"] and tp1_hit:
        ok = _partial_close(position, tp1_fraction, magic, tag)
        if ok:
            state["partial60_done"] = True
            state_store[ticket] = state
            time.sleep(0.30)
            position = _get_position(mt5_symbol, magic)
            if not position:
                return
            tick = mt5.symbol_info_tick(position.symbol)
            si   = mt5.symbol_info(position.symbol)
            if tick is None or si is None:
                return

    if state["partial60_done"] and not state["moved_to_be"]:
        if _move_sl_to_breakeven(position, be_buffer, magic, tag):
            state["moved_to_be"] = True
            state_store[ticket] = state
    # runner remains; closed by opposite signal

# ========== EXECUTION / RETRY ==========
def _attempt_execution(desired_side, *, tag, magic, sl_usd, risk_pct_of_balance):
    if prop.enforce_breaches():
        return False

    hit, _, _ = prop.profit_target_hit()
    if hit:
        return False

    position = _get_position(mt5_symbol, magic)
    open_side = None
    if position:
        open_side = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'

    if open_side == desired_side:
        return True

    if open_side and open_side != desired_side:
        _close_position_ticket(position, magic, tag)
        time.sleep(0.5)

    order_type = mt5.ORDER_TYPE_BUY if desired_side == 'BUY' else mt5.ORDER_TYPE_SELL
    return bool(send_order(mt5_symbol, order_type, magic=magic, tag=tag, sl_usd=sl_usd, risk_pct_of_balance=risk_pct_of_balance))

def _maybe_retry(pending, *, tag, magic, sl_usd, risk_pct_of_balance, session_windows):
    if not pending["signal"]:
        return
    if prop.enforce_breaches():
        return
    if prop.profit_target_hit()[0]:
        return

    now_local = datetime.now(local_tz)
    if not _in_session(now_local, session_windows):
        return
    if pending["last_retry"] and (now_local - pending["last_retry"]).total_seconds() < RETRY_EVERY_SECS:
        return

    print(f"[RETRY] Pending {pending['signal']} — attempting execution… ({tag})")
    ok = _attempt_execution(pending["signal"], tag=tag, magic=magic, sl_usd=sl_usd, risk_pct_of_balance=risk_pct_of_balance)
    pending["last_retry"] = now_local
    if ok:
        print(f"[OK] Pending {pending['signal']} executed. ({tag})")
        pending["signal"] = None
        pending["since"]  = None
        pending["last_retry"] = None

# ========== PER-TF BAR PROCESSOR ==========
def process_tf(granularity, *, tag, magic, use_heikin, atr_period, atr_mult,
               sl_usd, tp1_usd, tp1_fraction, be_buffer, risk_pct_of_balance,
               num_candles, require_post_start_candle, session_windows,
               last_candle_time_ref: str, saw_after_open_ref: str, pending: dict):
    """
    Runs when a new bar for this TF is due. Fetches candles, computes signals, manages position,
    places/queues orders, respects prop + sessions.
    last_candle_time_ref, saw_after_open_ref: names of globals to update
    """
    global last_candle_time_M5, last_candle_time_H1, saw_after_open_M5, saw_after_open_H1
    # session bounds (computed outside heartbeat when entering session)
    now_local = datetime.now(local_tz)
    if not _in_session(now_local, session_windows):
        return

    df = fetch_oanda_candles(granularity, num_candles)
    if df is None or df.empty:
        log(f"[{tag}] No data from OANDA, skipping.")
        return

    latest_candle_time = df.index[-1]
    prev_time = (last_candle_time_M5 if last_candle_time_ref == "M5" else last_candle_time_H1)
    if prev_time is not None and latest_candle_time <= prev_time:
        log(f"[{tag}] No new candle yet (latest <= last processed).")
        return

    if last_candle_time_ref == "M5": last_candle_time_M5 = latest_candle_time
    else: last_candle_time_H1 = latest_candle_time
    log(f"[{tag}] New {granularity} candle: {latest_candle_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Enforce "first post session start" if requested
    if require_post_start_candle:
        if last_candle_time_ref == "M5":
            start_dt, _ = _current_session_bounds(now_local, session_windows)
            if not saw_after_open_M5 and start_dt and latest_candle_time <= start_dt:
                log(f"[{tag}] Waiting for first post-session candle…")
                return
            saw_after_open_M5 = True
        else:
            start_dt, _ = _current_session_bounds(now_local, session_windows)
            if not saw_after_open_H1 and start_dt and latest_candle_time <= start_dt:
                log(f"[{tag}] Waiting for first post-session candle…")
                return
            saw_after_open_H1 = True

    # Print last 10 raw candles
    try:
        raw_to_show = df[['open','high','low','close','volume']].copy()
        raw_to_show.index = raw_to_show.index.strftime('%Y-%m-%d %H:%M')
        # print(f"\n===== RAW OANDA {granularity} DATA ({tag}) — last 10 =====")
        # print(raw_to_show.tail(10))
    except Exception as e:
        log(f"[{tag}] Raw candle print failed: {e}")

    # Indicators
    tr = calculate_indicators(df, useHeikinAshi=use_heikin, atrPeriod=atr_period, atrMult=atr_mult)
    latest = tr.iloc[-1]
    try:
        debug_df = tr[['ha_c','ha_open','ha_high','ha_low','dir','buy_signal','sell_signal']].copy()
        debug_df['signal'] = debug_df.apply(lambda row: 'BUY' if row['buy_signal'] else ('SELL' if row['sell_signal'] else ''), axis=1)
        debug_df.index = debug_df.index.strftime('%Y-%m-%d %H:%M')
        # print(f"\n===== HA + SIGNALS ({granularity}, {tag}) — last 10 =====")
        # print(debug_df[['ha_c','ha_open','ha_high','ha_low','dir','signal']].tail(10))
    except Exception as e:
        log(f"[{tag}] HA table print failed: {e}")

    # Signals
    signal = 'BUY' if bool(latest['buy_signal']) else ('SELL' if bool(latest['sell_signal']) else None)
    prev_signal = None
    if len(tr) >= 2:
        prev = tr.iloc[-2]
        if prev['buy_signal']: prev_signal = 'BUY'
        elif prev['sell_signal']: prev_signal = 'SELL'

    # Position info
    position = _get_position(mt5_symbol, magic)
    open_side = None
    if position:
        open_side = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'
        print(f"[INFO-{tag}] OPEN POSITION: {open_side}, vol: {position.volume}, entry: {position.price_open}")
        # Seed per-ticket state if new
        state_store = pos_state_M5 if tag == tag_M5 else pos_state_H1
        if position.ticket not in state_store:
            tp1_preview = compute_tp_price(position.symbol, position.type, position.price_open, tp1_usd)
            print(f"[INFO-{tag}] Manage ticket {position.ticket}: entry={position.price_open}, TP1(60%)={tp1_preview}")
            state_store[position.ticket] = {"partial60_done": False, "moved_to_be": False}
    else:
        print(f"[INFO-{tag}] No open position.")

    # Manage partial + BE intrabar
    _maybe_manage_open_position(tag, magic, tp1_usd, tp1_fraction, be_buffer)

    # Profit target gate
    hit, gain, target = prop.profit_target_hit()
    if hit:
        print(f"[TARGET-{tag}] Phase {prop.phase} target hit: Gain ${gain:.2f} ≥ ${target:.2f}. Halting new trades.")
        if alerts_enabled and alerter:
            alerter.phase_passed(prop.phase, gain, target)
        pending["signal"] = None; pending["since"] = None; pending["last_retry"] = None
        time.sleep(1)
        return

    # Execute with pending/backfill/override
    if signal:
        if pending["signal"] and signal != pending["signal"]:
            print(f"[CANCELLED-{tag}] Live signal {signal} overrides pending {pending['signal']}.")
            ok = _attempt_execution(signal, tag=tag, magic=magic, sl_usd=sl_usd, risk_pct_of_balance=risk_pct_of_balance)
            if ok:
                pending["signal"] = None; pending["since"] = None; pending["last_retry"] = None
            else:
                pending["signal"] = signal
                if pending["since"] is None: pending["since"] = datetime.now(local_tz)
        else:
            if open_side != signal:
                print(f"[TRADE-{tag}] New signal={signal} | open={open_side or 'NONE'}")
                ok = _attempt_execution(signal, tag=tag, magic=magic, sl_usd=sl_usd, risk_pct_of_balance=risk_pct_of_balance)
                if ok:
                    pending["signal"] = None; pending["since"] = None; pending["last_retry"] = None
                else:
                    if not pending["signal"]:
                        pending["signal"] = signal
                        pending["since"]  = datetime.now(local_tz)
                        print(f"[PENDING-{tag}] Queued {signal} (will retry).")
    else:
        if prev_signal and open_side != prev_signal and not pending["signal"]:
            print(f"[BACKFILL-{tag}] Previous bar had {prev_signal} — attempting execution now.")
            ok = _attempt_execution(prev_signal, tag=tag, magic=magic, sl_usd=sl_usd, risk_pct_of_balance=risk_pct_of_balance)
            if ok:
                pending["signal"] = None; pending["since"] = None; pending["last_retry"] = None
            else:
                pending["signal"] = prev_signal
                pending["since"]  = datetime.now(local_tz)
                print(f"[PENDING-{tag}] Queued {prev_signal} from previous bar (will retry).")
        else:
            print(f"[INFO-{tag}] No actionable signal.")

# ========== BOOT BANNER ==========
print("[BOOT] PropRules engine active. Phase:", current_phase,
      " | Targets → P1:", f"${prop.phase1_target:.0f}", " P2:", f"${prop.phase2_target:.0f}",
      " | Limits → Daily:", f"${prop.daily_limit:.0f}", " Overall:", f"${prop.overall_limit:.0f}")

if FORCE_DAILY_REPORT_ON_START:
    force_daily_report_now()

# ========== HEARTBEAT LOOP ==========
try:
    while True:
        # Hard stop if breached; also clear queued pendings
        if prop.enforce_breaches():
            pending_M5.update({"signal": None, "since": None, "last_retry": None})
            pending_H1.update({"signal": None, "since": None, "last_retry": None})
            time.sleep(1)
            continue

        # Ensure we're inside at least one TF session before starting waits
        while True:
            now_local = datetime.now(local_tz)
            prop.enforce_breaches()
            in_m5 = _in_session(now_local, session_windows_M5)
            in_h1 = _in_session(now_local, session_windows_H1)
            if in_m5 or in_h1:
                if in_m5:
                    sess_start_M5, sess_end_M5 = _current_session_bounds(now_local, session_windows_M5)
                    saw_after_open_M5 = False
                    if sess_start_M5 and sess_end_M5:
                        log(f"[+] IN SESSION {tag_M5}: {sess_start_M5.strftime('%H:%M:%S %Z')}–{sess_end_M5.strftime('%H:%M:%S %Z')}")
                    else:
                        log(f"[+] IN SESSION {tag_M5}: always-on")

                if in_h1:
                    sess_start_H1, sess_end_H1 = _current_session_bounds(now_local, session_windows_H1)
                    saw_after_open_H1 = False
                    if sess_start_H1 and sess_end_H1:
                        log(f"[+] IN SESSION {tag_H1}: {sess_start_H1.strftime('%H:%M:%S %Z')}–{sess_end_H1.strftime('%H:%M:%S %Z')}")
                    else:
                        log(f"[+] IN SESSION {tag_H1}: always-on")
                break
            time.sleep(1)

        # Wait to the NEAREST of next M5 and next H1 closes (that still lies within that TF's session)
        now = datetime.now(local_tz)
        t_m5 = _next_close_time(now, "M5") if _in_session(now, session_windows_M5) else None
        t_h1 = _next_close_time(now, "H1") if _in_session(now, session_windows_H1) else None

        # Clip by session end if needed
        if t_m5 and sess_end_M5 and t_m5 > sess_end_M5: t_m5 = sess_end_M5
        if t_h1 and sess_end_H1 and t_h1 > sess_end_H1: t_h1 = sess_end_H1

        targets = [t for t in [t_m5, t_h1] if t is not None]
        if not targets:
            # No active sessions right now; spin a bit
            time.sleep(1)
            continue
        target_next = min(targets)

        log(f"[*] Waiting until {target_next.strftime('%H:%M:%S %Z')} (nearest TF close)…")
        while True:
            if prop.enforce_breaches():
                break
            # manage open positions and retry pendings for BOTH TFs while waiting
            _maybe_manage_open_position(tag_M5, magic_M5, tp1_usd_distance_M5, tp1_fraction_M5, breakeven_buffer_usd_M5)
            _maybe_manage_open_position(tag_H1, magic_H1, tp1_usd_distance_H1, tp1_fraction_H1, breakeven_buffer_usd_H1)

            _maybe_retry(pending_M5, tag=tag_M5, magic=magic_M5, sl_usd=sl_usd_distance_M5,
                         risk_pct_of_balance=risk_pct_of_balance_M5, session_windows=session_windows_M5)
            _maybe_retry(pending_H1, tag=tag_H1, magic=magic_H1, sl_usd=sl_usd_distance_H1,
                         risk_pct_of_balance=risk_pct_of_balance_H1, session_windows=session_windows_H1)

            now = datetime.now(local_tz)
            if (target_next - now).total_seconds() <= 0:
                break
            time.sleep(0.25)
        time.sleep(2)  # allow API to finalize the bar

        # PROCESS TFs WHOSE BAR JUST CLOSED (and still inside their sessions)
        now = datetime.now(local_tz)

        if _in_session(now, session_windows_M5):
            # Extra guard: if session ended just now and required
            if sess_end_M5 and now >= sess_end_M5:
                log(f"[INFO] {tag_M5} session ended.")
                if auto_close_at_session_end_M5:
                    pos = _get_position(mt5_symbol, magic_M5)
                    if pos: _close_position_ticket(pos, magic_M5, tag_M5)
                pending_M5.update({"signal": None, "since": None, "last_retry": None})
            else:
                process_tf("M5",
                    tag=tag_M5, magic=magic_M5, use_heikin=use_heikin_ashi_M5,
                    atr_period=atr_period_M5, atr_mult=atr_mult_M5,
                    sl_usd=sl_usd_distance_M5, tp1_usd=tp1_usd_distance_M5, tp1_fraction=tp1_fraction_M5,
                    be_buffer=breakeven_buffer_usd_M5, risk_pct_of_balance=risk_pct_of_balance_M5,
                    num_candles=num_candles_M5, require_post_start_candle=require_post_start_candle_M5,
                    session_windows=session_windows_M5,
                    last_candle_time_ref="M5", saw_after_open_ref="M5", pending=pending_M5
                )

        if _in_session(now, session_windows_H1):
            if sess_end_H1 and now >= sess_end_H1:
                log(f"[INFO] {tag_H1} session ended.")
                if auto_close_at_session_end_H1:
                    pos = _get_position(mt5_symbol, magic_H1)
                    if pos: _close_position_ticket(pos, magic_H1, tag_H1)
                pending_H1.update({"signal": None, "since": None, "last_retry": None})
            else:
                process_tf("H1",
                    tag=tag_H1, magic=magic_H1, use_heikin=use_heikin_ashi_H1,
                    atr_period=atr_period_H1, atr_mult=atr_mult_H1,
                    sl_usd=sl_usd_distance_H1, tp1_usd=tp1_usd_distance_H1, tp1_fraction=tp1_fraction_H1,
                    be_buffer=breakeven_buffer_usd_H1, risk_pct_of_balance=risk_pct_of_balance_H1,
                    num_candles=num_candles_H1, require_post_start_candle=require_post_start_candle_H1,
                    session_windows=session_windows_H1,
                    last_candle_time_ref="H1", saw_after_open_ref="H1", pending=pending_H1
                )

except KeyboardInterrupt:
    log("[INFO] Stopped by user (CTRL-C).")
finally:
    try:
        mt5.shutdown()
        log("[INFO] MT5 shutdown complete.")
    except Exception as e:
        log(f"[WARN] MT5 shutdown error: {e}")
