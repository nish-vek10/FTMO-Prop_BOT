#!/usr/bin/env python
"""
CE + Chandelier Bot (XAUUSD M5) — Forward Test (MT5 execution, OANDA candles)

Replicates the backtest logic (as closely as possible in live conditions):

ENTRY (signal engine):
- Uses OANDA M5 candles (mid) → optional Heikin-Ashi
- CE direction flip signals (atr_period=1, atr_mult=1.85):
    buy_signal  when dir flips -1 -> +1
    sell_signal when dir flips +1 -> -1
- Execute at market after bar close (closest practical match to "next open" in backtest)

EXIT / MANAGEMENT:
- Chandelier stop for trailing:
    atr_period=20, atr_init_mult=1.0, chan_lookback=220
- Trailing SL updated on each new bar close (favorable-only)
- TP optional: tp_r * SL_dist (TP_R=1.5) unless DISABLE_TP=True
- Flip on opposite signal: close existing position then open new one

Risk / sizing:
- Default risk = 0.25% (changeable)
- dynamic_risk toggle:
    False → % of INITIAL_BALANCE
    True  → % of current equity
- Notional cap toggle (matches your backtest realism):
    MAX_NOTIONAL_USD = 500_000

Logging:
- Console logs + JSONL log file
- NO SPAM: SL updates only logged if moved by >= SL_LOG_MIN_DELTA_USD

Requirements:
- MetaTrader5 package installed
- requests, pandas, numpy

IMPORTANT:
- This bot uses OANDA candles for signals and MT5 ticks for execution.
  Live fills will not match backtest "next open" exactly; this is expected.
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import MetaTrader5 as mt5


# ============================================================
# PROJECT ROOT (auto-detected from this file location)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# ========================= CONFIG ============================
# ============================================================

# ---- Symbols / Timeframe ----
MT5_SYMBOL = "XAUUSD"
OANDA_SYMBOL = "XAU_USD"
OANDA_GRANULARITY = "M5"
OANDA_CANDLE_COUNT = 500

# ---- Strategy toggles ----
USE_HEIKIN_ASHI = True

# CE entry engine (fixed)
CE_ATR_PERIOD = 1
CE_ATR_MULT = 1.85

# Chandelier exit (champ parameters)
CH_ATR_PERIOD = 20
CH_ATR_MULT = 1.0
CH_LOOKBACK = 220

TP_R = 1.5
DISABLE_TP = False

# ---- Realism guards (same spirit as backtest) ----
MIN_SL_DIST_USD = 0.50
STOP_BUFFER_USD = 2.00  # backtest realism (min executable stop distance)

USE_NOTIONAL_CAP = True
MAX_NOTIONAL_USD = 500_000.0

# ---- Risk ----
INITIAL_BALANCE_FOR_STATIC = 100_000.0
RISK_PCT_PER_TRADE = 0.50
DYNAMIC_RISK = False        # False = % of INITIAL_BALANCE_FOR_STATIC ; True = % of equity

# ---- Execution ----
MAGIC_NUMBER = 77720180
DEVIATION_POINTS = 50            # MT5 deviation (points); tune for broker
ORDER_FILLING = mt5.ORDER_FILLING_IOC
ORDER_TIME_TYPE = mt5.ORDER_TIME_GTC

# ---- Logging ----
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

JSONL_LOG_PATH = LOG_DIR / "CE_bot_v2_NEW.jsonl"

# Only log trailing SL changes if change >= this (price units, XAU USD)
SL_LOG_MIN_DELTA_USD = 0.5

# Polling
POLL_SLEEP_SECS = 1.0  # while waiting for a new candle


# ---- MT5 login ----
MT5_LOGIN = 52683668
MT5_PASSWORD = "TnRAHT@71f6R3!"
MT5_SERVER = "ICMarketsSC-Demo"
MT5_TERMINAL_PATH = r"C:\MT5\EA-Prop_FTMO\Bot_v2\terminal64.exe"


# ---- OANDA (fill via env if possible) ----
OANDA_API_BASE = "https://api-fxpractice.oanda.com/v3"
OANDA_TOKEN = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"


# ============================================================
# ======================= LOG HELPERS =========================
# ============================================================

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log_event(event: str, **fields):
    payload = {"ts_utc": now_utc_iso(), "event": event, **fields}
    print(f"{payload['ts_utc']} | {event} | " + " | ".join(f"{k}={v}" for k, v in fields.items()), flush=True)
    try:
        with open(JSONL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as e:
        print(f"{now_utc_iso()} | LOG_WRITE_ERROR | {e}", flush=True)


# ============================================================
# ======================= OANDA FETCH =========================
# ============================================================

def fetch_oanda_candles(symbol: str, granularity: str, count: int) -> Optional[pd.DataFrame]:
    if not OANDA_TOKEN:
        log_event("FATAL", msg="Missing OANDA_TOKEN. Set env var OANDA_TOKEN.")
        return None

    url = f"{OANDA_API_BASE}/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {OANDA_TOKEN}"}
    params = {"granularity": granularity, "count": int(count), "price": "M"}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=(8, 20))
    except requests.RequestException as e:
        log_event("OANDA_ERROR", err=str(e))
        return None

    if r.status_code != 200:
        log_event("OANDA_ERROR", status=r.status_code, body=r.text[:300])
        return None

    raw = r.json().get("candles", [])
    rows = []
    for c in raw:
        if not c.get("complete", False):
            continue
        t = pd.to_datetime(c["time"], utc=True)
        mid = c["mid"]
        rows.append((t, float(mid["o"]), float(mid["h"]), float(mid["l"]), float(mid["c"]), int(c.get("volume", 0))))

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["time"]).sort_values("time").set_index("time")
    return df


# ============================================================
# ====================== INDICATORS ===========================
# ============================================================

def heikin_ashi_arrays(O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray):
    n = len(O)
    ha_close = (O + H + L + C) / 4.0
    ha_open = np.empty(n, dtype=float)
    ha_open[0] = (O[0] + C[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum.reduce([H, ha_open, ha_close])
    ha_low = np.minimum.reduce([L, ha_open, ha_close])
    return ha_open, ha_high, ha_low, ha_close

def rma_atr_from_hlc(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    c_prev = np.roll(c, 1)
    c_prev[0] = np.nan
    tr = np.empty_like(c, dtype=float)
    tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c_prev[i]), abs(l[i] - c_prev[i]))
    if n <= 1:
        return tr
    out = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= n:
        seed = float(np.nanmean(tr[:n]))
        out[n - 1] = seed
        alpha = 1.0 / n
        for i in range(n, len(tr)):
            out[i] = out[i - 1] + alpha * (tr[i] - out[i - 1])
    return pd.Series(out).ffill().bfill().to_numpy(dtype=float)

def compute_ce_signals(O, H, L, C, use_ha: bool, atr_period: int, atr_mult: float):
    if use_ha:
        _, h, l, c = heikin_ashi_arrays(O, H, L, C)
    else:
        h, l, c = H, L, C

    n = int(max(1, atr_period))
    atr = rma_atr_from_hlc(h, l, c, n)
    atr_val = atr_mult * atr

    hh = pd.Series(h).rolling(window=n, min_periods=n).max().to_numpy()
    ll = pd.Series(l).rolling(window=n, min_periods=n).min().to_numpy()

    long_stop = hh - atr_val
    short_stop = ll + atr_val

    lss = long_stop.copy()
    sss = short_stop.copy()

    for i in range(1, len(c)):
        long_prev = lss[i - 1] if np.isfinite(lss[i - 1]) else long_stop[i]
        short_prev = sss[i - 1] if np.isfinite(sss[i - 1]) else short_stop[i]
        lss[i] = max(long_stop[i], long_prev) if c[i - 1] > long_prev else long_stop[i]
        sss[i] = min(short_stop[i], short_prev) if c[i - 1] < short_prev else short_stop[i]

    dir_vals = np.ones(len(c), dtype=int)
    for i in range(1, len(c)):
        if c[i] > sss[i - 1]:
            dir_vals[i] = 1
        elif c[i] < lss[i - 1]:
            dir_vals[i] = -1
        else:
            dir_vals[i] = dir_vals[i - 1]

    dir_prev = np.roll(dir_vals, 1)
    dir_prev[0] = dir_vals[0]
    buy_signal = (dir_vals == 1) & (dir_prev == -1)
    sell_signal = (dir_vals == -1) & (dir_prev == 1)
    return atr, buy_signal, sell_signal

def chandelier_stop_arrays(O, H, L, C, atr_arr: np.ndarray, lookback: int, mult: float, use_ha: bool):
    if use_ha:
        _, h, l, c = heikin_ashi_arrays(O, H, L, C)
    else:
        h, l, c = H, L, C

    lb = int(max(1, lookback))
    hh = pd.Series(h).rolling(window=lb, min_periods=lb).max().to_numpy()
    ll = pd.Series(l).rolling(window=lb, min_periods=lb).min().to_numpy()

    long_stop = hh - (mult * atr_arr)
    short_stop = ll + (mult * atr_arr)

    lss = long_stop.copy()
    sss = short_stop.copy()

    for i in range(1, len(c)):
        long_prev = lss[i - 1] if np.isfinite(lss[i - 1]) else long_stop[i]
        short_prev = sss[i - 1] if np.isfinite(sss[i - 1]) else short_stop[i]
        lss[i] = max(long_stop[i], long_prev) if c[i - 1] > long_prev else long_stop[i]
        sss[i] = min(short_stop[i], short_prev) if c[i - 1] < short_prev else short_stop[i]

    return lss, sss


# ============================================================
# ===================== CLOSE DETECTOR ========================
# ============================================================

# Track last known position so we can detect closes
_last_pos_ticket: Optional[int] = None
_last_pos_side: Optional[str] = None
_last_pos_sl: Optional[float] = None
_last_pos_tp: Optional[float] = None

# Track trailing SL updates so we can label SL hits as TRAIL_SL when appropriate
_last_trail_by_ticket: dict[int, dict] = {}  # {ticket: {"sl": float, "ts_utc": str}}

def _utc_from_mt5_time(t: int) -> str:
    # MT5 deal time is unix seconds in terminal timezone; treating as UTC is usually fine for logging.
    # If your terminal runs local time, this will still be consistent internally.
    return datetime.fromtimestamp(int(t), tz=timezone.utc).isoformat()

def _deal_reason_name(reason_code: int) -> str:
    # Map common MT5 deal reasons; fall back to code.
    # (These constants exist in MetaTrader5 python package)
    mapping = {}
    if hasattr(mt5, "DEAL_REASON_SL"):
        mapping[int(mt5.DEAL_REASON_SL)] = "SL"
    if hasattr(mt5, "DEAL_REASON_TP"):
        mapping[int(mt5.DEAL_REASON_TP)] = "TP"
    if hasattr(mt5, "DEAL_REASON_SO"):
        mapping[int(mt5.DEAL_REASON_SO)] = "STOP_OUT"
    if hasattr(mt5, "DEAL_REASON_CLIENT"):
        mapping[int(mt5.DEAL_REASON_CLIENT)] = "MANUAL/CLIENT"
    if hasattr(mt5, "DEAL_REASON_EXPERT"):
        mapping[int(mt5.DEAL_REASON_EXPERT)] = "EXPERT"
    if hasattr(mt5, "DEAL_REASON_MOBILE"):
        mapping[int(mt5.DEAL_REASON_MOBILE)] = "MOBILE"
    if hasattr(mt5, "DEAL_REASON_WEB"):
        mapping[int(mt5.DEAL_REASON_WEB)] = "WEB"
    if hasattr(mt5, "DEAL_REASON_ROLLOVER"):
        mapping[int(mt5.DEAL_REASON_ROLLOVER)] = "ROLLOVER"
    if hasattr(mt5, "DEAL_REASON_EXTERNAL_CLIENT"):
        mapping[int(mt5.DEAL_REASON_EXTERNAL_CLIENT)] = "EXTERNAL_CLIENT"

    return mapping.get(int(reason_code), f"OTHER({reason_code})")

def _best_close_deal_for_ticket(ticket: int):
    """
    Fetch recent deals and return the most recent OUT deal for this position ticket.
    """
    # IMPORTANT: MT5 python API is most reliable with naive datetimes (terminal/local time)
    t_to = datetime.now()
    t_from = t_to - timedelta(hours=12)

    deals = mt5.history_deals_get(t_from, t_to)

    if deals is None or len(deals) == 0:
        return None

    # Filter deals for our ticket (position_id usually matches the position ticket)
    # Also filter symbol/magic for safety.
    filtered = []
    for d in deals:
        if getattr(d, "symbol", None) != MT5_SYMBOL:
            continue
        if int(getattr(d, "magic", 0) or 0) != int(MAGIC_NUMBER):
            continue

        pos_id = int(getattr(d, "position_id", 0) or 0)
        order_id = int(getattr(d, "order", 0) or 0)

        # Some brokers link closes by position_id, others by order id.
        if pos_id != int(ticket) and order_id != int(ticket):
            continue

        entry = getattr(d, "entry", None)
        # out types: DEAL_ENTRY_OUT / DEAL_ENTRY_OUT_BY (depending on broker)
        if hasattr(mt5, "DEAL_ENTRY_OUT") and entry == mt5.DEAL_ENTRY_OUT:
            filtered.append(d)
        elif hasattr(mt5, "DEAL_ENTRY_OUT_BY") and entry == mt5.DEAL_ENTRY_OUT_BY:
            filtered.append(d)

    if not filtered:
        return None

    # Return the latest by time
    filtered.sort(key=lambda x: int(getattr(x, "time", 0) or 0))
    return filtered[-1]

def detect_and_log_close(current_pos: Optional[mt5.TradePosition]):
    """
    If we had a position last cycle and now it's gone, pull close deal & log.
    """
    global _last_pos_ticket, _last_pos_side, _last_pos_sl, _last_pos_tp

    # no prior position -> nothing to detect
    if _last_pos_ticket is None:
        return

    # if still open -> nothing to do
    if current_pos is not None and int(getattr(current_pos, "ticket", 0)) == int(_last_pos_ticket):
        return

    # position disappeared OR ticket changed (closed/flip)
    ticket = int(_last_pos_ticket)
    side = _last_pos_side

    d = _best_close_deal_for_ticket(ticket)
    if d is None:
        log_event("CLOSE_OUT", ticket=ticket, side=side, reason="UNKNOWN", msg="No deal found in history window")
        _last_pos_ticket = None
        _last_pos_side = None
        _last_pos_sl = None
        _last_pos_tp = None
        return

    close_time_utc = _utc_from_mt5_time(getattr(d, "time", 0) or 0)
    close_price = float(getattr(d, "price", 0.0) or 0.0)
    profit = float(getattr(d, "profit", 0.0) or 0.0)
    volume = float(getattr(d, "volume", 0.0) or 0.0)
    reason_code = int(getattr(d, "reason", -1) or -1)
    base_reason = _deal_reason_name(reason_code)

    # Promote SL -> TRAIL_SL if we recently moved SL via trailing and it matches
    final_reason = base_reason
    if base_reason == "SL":
        trail = _last_trail_by_ticket.get(ticket)
        if trail:
            tr_sl = float(trail.get("sl", 0.0) or 0.0)
            # if close was near the last trailed SL, label as TRAIL_SL
            if tr_sl > 0 and abs(close_price - tr_sl) <= 1.0:  # 1.0 USD tolerance
                final_reason = "TRAIL_SL"

    slippage = None
    if base_reason in ("SL", "TRAIL_SL"):
        if _last_pos_sl and _last_pos_sl > 0:
            slippage = round(close_price - _last_pos_sl, 3)

    log_event(
        "CLOSE_OUT",
        ticket=ticket,
        side=side,
        close_time_utc=close_time_utc,
        close_price=round(close_price, 3),
        reason=final_reason,
        profit=round(profit, 2),
        volume=volume,
        base_reason=base_reason,
        slippage=slippage,
    )

    # clear last position state (we already logged close)
    _last_pos_ticket = None
    _last_pos_side = None
    _last_pos_sl = None
    _last_pos_tp = None


# ============================================================
# ======================== MT5 HELPERS ========================
# ============================================================

def mt5_init_or_die():
    ok = mt5.initialize(
        login=int(MT5_LOGIN) if str(MT5_LOGIN).isdigit() and int(MT5_LOGIN) != 0 else None,
        password=MT5_PASSWORD if MT5_PASSWORD else None,
        server=MT5_SERVER if MT5_SERVER else None,
        path=MT5_TERMINAL_PATH if MT5_TERMINAL_PATH else None
    )
    if not ok:
        print("[FATAL] MT5 initialize failed:", mt5.last_error())
        sys.exit(1)

    acc = mt5.account_info()
    if not acc:
        print("[FATAL] MT5 account_info failed:", mt5.last_error())
        sys.exit(1)

    if not mt5.symbol_select(MT5_SYMBOL, True):
        print("[FATAL] mt5.symbol_select failed:", mt5.last_error())
        sys.exit(1)

    log_event("MT5_CONNECTED", login=acc.login, balance=acc.balance, equity=acc.equity)

def get_our_position() -> Optional[mt5.TradePosition]:
    pos = mt5.positions_get(symbol=MT5_SYMBOL)
    if not pos:
        return None
    for p in pos:
        if int(getattr(p, "magic", 0)) == int(MAGIC_NUMBER):
            return p
    return None

def close_position(pos: mt5.TradePosition) -> bool:
    tick = mt5.symbol_info_tick(MT5_SYMBOL)
    if not tick:
        return False
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": MT5_SYMBOL,
        "volume": float(pos.volume),
        "type": order_type,
        "position": pos.ticket,
        "price": price,
        "deviation": DEVIATION_POINTS,
        "magic": MAGIC_NUMBER,
        "comment": "CE_CHAN_CLOSE",
        "type_time": ORDER_TIME_TYPE,
        "type_filling": ORDER_FILLING,
    }
    res = mt5.order_send(req)
    ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
    log_event("CLOSE", ok=ok, retcode=getattr(res, "retcode", None), comment=getattr(res, "comment", None), ticket=pos.ticket)
    return ok

def modify_sltp(pos: mt5.TradePosition, sl: Optional[float], tp: Optional[float]) -> Tuple[bool, object]:
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": MT5_SYMBOL,
        "position": pos.ticket,
        "sl": float(sl) if sl is not None else 0.0,
        "tp": float(tp) if tp is not None else 0.0,
        "magic": MAGIC_NUMBER,
        "comment": "CE_CHAN_SLTP",
    }

    res = mt5.order_send(req)
    if res is None:
        return False, res

    rc = int(getattr(res, "retcode", -1) or -1)
    # 10009 = DONE, 10025 = NO_CHANGES
    ok = rc in (10009, 10025)

    return ok, res

def ensure_sltp(pos: mt5.TradePosition, sl: Optional[float], tp: Optional[float], retries: int = 5) -> bool:
    for i in range(retries):
        ok, res = modify_sltp(pos, sl=sl, tp=tp)
        log_event(
            "SET_SLTP_TRY",
            ticket=pos.ticket,
            try_n=i + 1,
            ok=ok,
            sl=sl,
            tp=tp,
            retcode=getattr(res, "retcode", None),
            comment=getattr(res, "comment", None),
            last_error=str(mt5.last_error()),
        )
        if ok:
            return True
        time.sleep(0.35)
    return False

def _round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.floor(x / step) * step

def lots_for_risk(sl_dist_usd: float, entry_price: float) -> Tuple[float, dict]:
    """
    Return (lots, debug_dict) using:
      risk_amount = equity_or_initial * (RISK_PCT_PER_TRADE / 100)
      risk_per_lot = value_per_$1_move_per_lot * sl_dist_usd
      lots = risk_amount / risk_per_lot
    Apply broker min/max/step and (optional) notional cap.
    """

    CONTRACT_SIZE = 100.0

    si = mt5.symbol_info(MT5_SYMBOL)
    acc = mt5.account_info()
    if not si or not acc:
        return 0.0, {"err": "mt5 symbol/account missing"}

    base_equity = float(acc.equity) if DYNAMIC_RISK else float(INITIAL_BALANCE_FOR_STATIC)
    risk_amount = base_equity * (float(RISK_PCT_PER_TRADE) / 100.0)

    # derive value per $1 move per lot
    v_per_1usd = None
    tv = getattr(si, "trade_tick_value", None)
    ts = getattr(si, "trade_tick_size", None)
    if tv not in (None, 0) and ts not in (None, 0):
        v_per_1usd = float(tv) / float(ts)
    else:
        tv2 = getattr(si, "tick_value", None)
        ts2 = getattr(si, "tick_size", None)
        if tv2 not in (None, 0) and ts2 not in (None, 0):
            v_per_1usd = float(tv2) / float(ts2)

    if v_per_1usd in (None, 0):
        cs = getattr(si, "trade_contract_size", None)
        if cs not in (None, 0):
            v_per_1usd = float(cs)
        else:
            # fallback to your backtest contract size assumption
            v_per_1usd = float(CONTRACT_SIZE)

    risk_per_lot = max(1e-9, v_per_1usd * float(sl_dist_usd))
    raw_lots = risk_amount / risk_per_lot

    capped_notional = False
    if USE_NOTIONAL_CAP and MAX_NOTIONAL_USD > 0:

        cs = float(getattr(si, "trade_contract_size", 0.0) or CONTRACT_SIZE)
        max_lots_notional = float(MAX_NOTIONAL_USD) / max(1e-9, float(entry_price) * cs)

        if raw_lots > max_lots_notional:
            raw_lots = max_lots_notional
            capped_notional = True

    lots = float(raw_lots)
    lots = _round_to_step(lots, float(si.volume_step))
    lots = max(float(si.volume_min), min(lots, float(si.volume_max)))

    return lots, {
        "risk_amount": risk_amount,
        "base_equity": base_equity,
        "v_per_1usd": v_per_1usd,
        "raw_lots": raw_lots,
        "lots": lots,
        "capped_notional": capped_notional,
    }

def place_market(side: str, sl: float, tp: Optional[float]) -> bool:
    si = mt5.symbol_info(MT5_SYMBOL)
    tick = mt5.symbol_info_tick(MT5_SYMBOL)
    if not si or not tick:
        return False

    order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    min_stop = max(MIN_SL_DIST_USD, STOP_BUFFER_USD)
    sl_dist = abs(price - sl)
    if sl_dist < min_stop:
        log_event("SKIP_ENTRY_MIN_STOP", side=side, sl_dist=round(sl_dist, 3), min_stop=min_stop)
        return False

    lots, dbg = lots_for_risk(sl_dist_usd=sl_dist, entry_price=price)
    if lots <= 0:
        log_event("ORDER_FAIL", side=side, reason="lots<=0", dbg=dbg)
        return False

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": MT5_SYMBOL,
        "volume": lots,
        "type": order_type,
        "price": price,
        "deviation": DEVIATION_POINTS,
        "magic": MAGIC_NUMBER,
        "comment": "CE_CHAN_ENTRY",
        "type_time": ORDER_TIME_TYPE,
        "type_filling": ORDER_FILLING,
        # attach SL/TP at entry (broker-friendly)
        "sl": float(sl),
        "tp": float(tp) if (tp is not None) else 0.0,
    }

    res = mt5.order_send(req)
    ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
    log_event("ENTRY", ok=ok, side=side, lots=lots, price=price, sl=sl, tp=tp, retcode=getattr(res, "retcode", None), comment=getattr(res, "comment", None), sizing=dbg)

    if not ok:
        return False

    time.sleep(0.3)

    pos = get_our_position()
    if not pos:
        log_event("WARN", msg="position not visible after entry; will retry SLTP later")
        return True

    # Arm close detector immediately after entry (so fast closes still get logged)
    global _last_pos_ticket, _last_pos_side, _last_pos_sl, _last_pos_tp
    _last_pos_ticket = int(pos.ticket)
    _last_pos_side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
    _last_pos_sl = float(getattr(pos, "sl", 0.0) or 0.0)
    _last_pos_tp = float(getattr(pos, "tp", 0.0) or 0.0)
    log_event("DEBUG_CLOSE_ARMED", last_ticket=_last_pos_ticket, last_side=_last_pos_side)

    ok2 = ensure_sltp(pos, sl=sl, tp=tp, retries=5)
    log_event("SET_SLTP", ok=ok2, ticket=pos.ticket, sl=sl, tp=tp)

    return True


# ============================================================
# ======================= BOT CORE ============================
# ============================================================

@dataclass
class SignalInfo:
    bar_time: pd.Timestamp
    signal: Optional[str]           # "BUY" / "SELL" / None
    chand_long: float
    chand_short: float
    atr_ch: float

def build_signals(df: pd.DataFrame) -> Optional[SignalInfo]:
    if df is None or df.empty or len(df) < 50:
        return None

    idx = df.index
    O = df["open"].to_numpy(float)
    H = df["high"].to_numpy(float)
    L = df["low"].to_numpy(float)
    C = df["close"].to_numpy(float)

    # CE signal engine (entry)
    _, buy_sig, sell_sig = compute_ce_signals(O, H, L, C, USE_HEIKIN_ASHI, CE_ATR_PERIOD, CE_ATR_MULT)
    sig = None
    if bool(buy_sig[-1]):
        sig = "BUY"
    elif bool(sell_sig[-1]):
        sig = "SELL"

    # Chandelier ATR (exit ATR period)
    if USE_HEIKIN_ASHI:
        _, h_ha, l_ha, c_ha = heikin_ashi_arrays(O, H, L, C)
        atr_ch = rma_atr_from_hlc(h_ha, l_ha, c_ha, int(CH_ATR_PERIOD))
    else:
        atr_ch = rma_atr_from_hlc(H, L, C, int(CH_ATR_PERIOD))

    lss, sss = chandelier_stop_arrays(O, H, L, C, atr_ch, lookback=int(CH_LOOKBACK), mult=float(CH_ATR_MULT), use_ha=USE_HEIKIN_ASHI)

    return SignalInfo(
        bar_time=idx[-1],
        signal=sig,
        chand_long=float(lss[-1]) if np.isfinite(lss[-1]) else float("nan"),
        chand_short=float(sss[-1]) if np.isfinite(sss[-1]) else float("nan"),
        atr_ch=float(atr_ch[-1]) if np.isfinite(atr_ch[-1]) else 0.0,
    )

def desired_sl_for_entry(side: str, entry_price: float, siginfo: SignalInfo) -> float:
    """
    Backtest-aligned ATR-entry stop:
      dist = max(min_stop, CH_ATR_MULT * atr_ch)
      BUY  -> entry - dist
      SELL -> entry + dist

    Note: CH_ATR_MULT=1.0 and atr_ch is ATR(20) built from HA (if enabled), same as backtest.
    """
    min_stop = max(MIN_SL_DIST_USD, STOP_BUFFER_USD)

    atr_dist = float(CH_ATR_MULT) * float(siginfo.atr_ch or 0.0)
    dist = max(min_stop, atr_dist)

    if side == "BUY":
        return float(entry_price - dist)
    else:
        return float(entry_price + dist)

def tp_for_entry(side: str, entry_price: float, sl_price: float) -> Optional[float]:
    if DISABLE_TP:
        return None

    min_stop = max(MIN_SL_DIST_USD, STOP_BUFFER_USD)

    sl_dist = abs(entry_price - sl_price)
    tp_dist = float(TP_R) * sl_dist

    # If TP would be too close (rare now, but keep parity-safe)
    if tp_dist < min_stop:
        return None

    return entry_price + tp_dist if side == "BUY" else entry_price - tp_dist

def update_trailing_sl(pos: mt5.TradePosition, siginfo: SignalInfo):
    """
    Update SL to the latest chandelier stop (favorable-only).
    No spam logs: only log if SL moves by >= SL_LOG_MIN_DELTA_USD.
    """
    if not pos:
        return
    current_sl = float(getattr(pos, "sl", 0.0) or 0.0)

    side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
    new_sl = None

    if side == "BUY":
        if np.isfinite(siginfo.chand_long):
            new_sl = float(siginfo.chand_long)
            if current_sl > 0:
                new_sl = max(current_sl, new_sl)
    else:
        if np.isfinite(siginfo.chand_short):
            new_sl = float(siginfo.chand_short)
            if current_sl > 0:
                new_sl = min(current_sl, new_sl)

    if new_sl is None:
        return

    # only update if meaningful change
    if current_sl > 0 and abs(new_sl - current_sl) < SL_LOG_MIN_DELTA_USD:
        return

    # don't set invalid SL relative to market
    tick = mt5.symbol_info_tick(MT5_SYMBOL)
    if not tick:
        return
    if side == "BUY" and new_sl >= tick.bid:
        return
    if side == "SELL" and new_sl <= tick.ask:
        return

    # keep TP as-is unless disabled
    tp = float(getattr(pos, "tp", 0.0) or 0.0)
    if DISABLE_TP:
        tp_val = None
    else:
        tp_val = tp if tp > 0 else None

    ok = ensure_sltp(pos, sl=new_sl, tp=tp_val, retries=3)

    if ok:
        _last_trail_by_ticket[int(pos.ticket)] = {"sl": float(new_sl), "ts_utc": now_utc_iso()}
        log_event("TRAIL_SL", ticket=pos.ticket, side=side, old_sl=current_sl, new_sl=new_sl)
    else:
        log_event("TRAIL_SL_FAIL", ticket=pos.ticket, side=side, target_sl=new_sl)


def main_loop():
    global _last_pos_ticket, _last_pos_side, _last_pos_sl, _last_pos_tp
    mt5_init_or_die()
    last_bar_time: Optional[pd.Timestamp] = None

    oanda_fail_streak = 0
    next_fetch_sleep = POLL_SLEEP_SECS

    log_event(
        "BOT_START",
        symbol=MT5_SYMBOL,
        risk_pct=RISK_PCT_PER_TRADE,
        dynamic_risk=DYNAMIC_RISK,
        disable_tp=DISABLE_TP,
        tp_r=TP_R,
        ch_atr_period=CH_ATR_PERIOD,
        ch_atr_mult=CH_ATR_MULT,
        ch_lookback=CH_LOOKBACK,
        ce_atr_period=CE_ATR_PERIOD,
        ce_atr_mult=CE_ATR_MULT,
        stop_buffer=STOP_BUFFER_USD,
        notional_cap=USE_NOTIONAL_CAP,
        max_notional=MAX_NOTIONAL_USD,
    )

    while True:
        df = fetch_oanda_candles(OANDA_SYMBOL, OANDA_GRANULARITY, OANDA_CANDLE_COUNT)
        if df is None or df.empty:
            oanda_fail_streak += 1
            # exponential backoff up to 60s
            next_fetch_sleep = min(60.0, POLL_SLEEP_SECS * (2 ** min(oanda_fail_streak, 6)))
            time.sleep(next_fetch_sleep)
            continue

        # reset on success
        oanda_fail_streak = 0
        next_fetch_sleep = POLL_SLEEP_SECS

        siginfo = build_signals(df)
        if not siginfo:
            time.sleep(POLL_SLEEP_SECS)
            continue

        # only act once per new completed bar
        if last_bar_time is not None and siginfo.bar_time <= last_bar_time:
            time.sleep(POLL_SLEEP_SECS)
            continue

        last_bar_time = siginfo.bar_time

        pos = get_our_position()
        detect_and_log_close(pos)
        open_side = None
        if pos:
            open_side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"

        if pos:
            _last_pos_ticket = int(pos.ticket)
            _last_pos_side = open_side
            _last_pos_sl = float(getattr(pos, "sl", 0.0) or 0.0)
            _last_pos_tp = float(getattr(pos, "tp", 0.0) or 0.0)
            log_event("DEBUG_CLOSE_ARMED", last_ticket=_last_pos_ticket, last_side=_last_pos_side)

        # log bar summary (no spam: once per bar)
        log_event(
            "BAR",
            time=str(siginfo.bar_time),
            signal=siginfo.signal,
            open_side=open_side,
            chand_long=(None if not np.isfinite(siginfo.chand_long) else round(siginfo.chand_long, 3)),
            chand_short=(None if not np.isfinite(siginfo.chand_short) else round(siginfo.chand_short, 3)),
        )

        log_event(
            "DEBUG_STATE",
            pos_found=bool(pos),
            open_side=open_side,
            signal=siginfo.signal,
        )

        # trailing update (no spam)
        if pos:
            update_trailing_sl(pos, siginfo)

        # flip/entry logic
        if siginfo.signal in ("BUY", "SELL"):
            log_event("DEBUG_SIGNAL_BRANCH", desired=siginfo.signal, pos_found=bool(pos), open_side=open_side)
            desired = siginfo.signal

            if pos and open_side != desired:
                # close then open
                close_position(pos)
                time.sleep(0.5)
                pos = None
                open_side = None

            if not pos:
                # compute entry via current tick (live approximation)
                tick = mt5.symbol_info_tick(MT5_SYMBOL)
                if not tick:
                    continue
                entry_px = tick.ask if desired == "BUY" else tick.bid

                sl = desired_sl_for_entry(desired, entry_px, siginfo)
                tp = tp_for_entry(desired, entry_px, sl)

                # --- diagnostics: is SL on the correct side? ---
                sl_valid = None
                if desired == "BUY":
                    sl_valid = (sl < entry_px)
                else:
                    sl_valid = (sl > entry_px)

                log_event(
                    "ENTRY_PRECHECK",
                    side=desired,
                    entry_px=round(entry_px, 3),
                    sl=round(sl, 3),
                    tp=(None if tp is None else round(tp, 3)),
                    sl_valid=bool(sl_valid),
                    sl_dist=round(abs(entry_px - sl), 3),
                )

                min_stop = max(MIN_SL_DIST_USD, STOP_BUFFER_USD)

                # enforce min executable SL dist (backtest parity)
                if abs(entry_px - sl) < min_stop:
                    bt = siginfo.bar_time
                    log_event(
                        "SKIP_ENTRY_MIN_STOP",
                        side=desired,
                        bar_time=str(bt),
                        dow=int(bt.dayofweek),
                        hour=int(bt.hour),
                        sl_dist=round(abs(entry_px - sl), 3),
                        min_stop=min_stop,
                    )
                    continue

                place_market(desired, sl=sl, tp=tp)

        # small sleep to reduce API pressure
        time.sleep(0.2)


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        log_event("BOT_STOP", reason="KeyboardInterrupt")
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass
