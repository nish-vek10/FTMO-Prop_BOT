#!/usr/bin/env python
"""
run_grid_ce_xau_m5.py  (FAST + SAFE + FORMATTED + REALISTIC)

- Uses numpy arrays (fast)
- Caches entry signals + ATR arrays + chandelier stops (fast)
- Adds MIN_SL_DIST_USD guard to stop insane lot sizing (realistic)
- Adds MAX_NOTIONAL_USD cap (realistic leverage / margin constraint)
- Adds ATR-based slippage (adverse) for entry/exit
- Uses mark-to-market (worst-case intrabar) drawdown instead of closed-only DD (realistic)
- Labels TRAIL_STOP when stop is hit in profit
- Adds per-scenario disable_tp flag (TP ON vs TP OFF) and runs both in the grid
- Removes duplicate chandelier rows by not looping unused params
- Exports CSV with proper formatting (no scientific notation)
"""

from __future__ import annotations

import os
import json
import time as time_mod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# =========================
# USER CONFIG
# =========================
PROJECT_ROOT = Path(r"C:\Users\anish\PycharmProjects\PropEAs")
BT_ROOT = PROJECT_ROOT / "backtests"
OUT_ROOT = BT_ROOT / "output"

SYMBOL_OANDA = "XAU_USD"
GRANULARITY = "M5"

START_DATE_UTC = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE_UTC = datetime.now(timezone.utc)

INITIAL_BALANCE = 100_000.0
RISK_PCT_PER_TRADE = 0.5  # percent
DYNAMIC_RISK = False      # True: % of current equity, False: % of initial balance

CONTRACT_SIZE = 100.0

# Costs
USE_SPREAD_COST = True
SPREAD_USD = 0.20

# Slippage (adverse). Set USE_SLIPPAGE=False to disable.
USE_SLIPPAGE = True
SLIPPAGE_ENTRY_ATR_MULT = 0.02
SLIPPAGE_EXIT_ATR_MULT  = 0.02

# Mark-to-market DD (worst-case intrabar). Set False to go back to closed-only DD.
USE_MTM_WORST_DD = True

# ENTRY engine (signal) fixed to match your live bot
USE_HEIKIN_ASHI = True
CE_ATR_PERIOD = 1
CE_ATR_MULT = 1.85

# ---- REALISM GUARDS (CRITICAL) ----
MIN_SL_DIST_USD = 0.50          # if SL distance < this, SKIP the trade
MAX_LOTS_CAP = 50.0             # optional hard cap (still useful as safety)
MAX_NOTIONAL_USD = 500_000.0    # realistic clamp (tune: 250k, 500k, 1M)

# Grid dimension: TP ON vs TP OFF (trend ride)
GRID_DISABLE_TP = [False]

# data cache
DATA_DIR = BT_ROOT / "data_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CANDLES_CACHE_CSV = DATA_DIR / f"candles_{SYMBOL_OANDA}_{GRANULARITY}_2025_to_today.csv"

# OANDA creds via env (DO NOT hardcode tokens)
OANDA_API_BASE = "https://api-fxpractice.oanda.com/v3"
OANDA_TOKEN = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"


# =========================
# GRID RANGES
# =========================
MIN_ATR_PERIOD = 5
MAX_ATR_PERIOD = 20
ATR_PERIOD_STEP = 5

MIN_ATR_INIT_MULT = 1
MAX_ATR_INIT_MULT = 4
ATR_INIT_MULT_STEP = 1

MIN_ATR_TRAIL_MULT = 1
MAX_ATR_TRAIL_MULT = 4
ATR_TRAIL_MULT_STEP = 1

MIN_CHAN_LOOKBACK = 20
MAX_CHAN_LOOKBACK = 180
CHAN_LOOKBACK_STEP = 20

GRID_ATR_PERIOD = list(range(MIN_ATR_PERIOD, MAX_ATR_PERIOD + 1, ATR_PERIOD_STEP))
GRID_ATR_INIT_MULT = [float(x) for x in range(MIN_ATR_INIT_MULT, MAX_ATR_INIT_MULT + 1, ATR_INIT_MULT_STEP)]
GRID_ATR_TRAIL_MULT = [float(x) for x in range(MIN_ATR_TRAIL_MULT, MAX_ATR_TRAIL_MULT + 1, ATR_TRAIL_MULT_STEP)]
GRID_CHAN_LOOKBACK = list(range(MIN_CHAN_LOOKBACK, MAX_CHAN_LOOKBACK + 1, CHAN_LOOKBACK_STEP))

MIN_TP_R = 1.0
MAX_TP_R = 3.0
TP_R_STEP = 0.5
GRID_TP_R = [round(float(x), 10) for x in np.arange(MIN_TP_R, MAX_TP_R + 1e-12, TP_R_STEP)]

MIN_TRAIL_START_R = 0.5
MAX_TRAIL_START_R = 3.0
TRAIL_START_R_STEP = 0.5
GRID_TRAIL_START_R = [round(float(x), 10) for x in np.arange(MIN_TRAIL_START_R, MAX_TRAIL_START_R + 1e-12, TRAIL_START_R_STEP)]

GRID_MAX_HOLD_BARS = [None]

STOP_MODES = [
    # "atr_static",
    "atr_trailing",
    "chandelier",
]


# =========================
# TERMINAL PROGRESS PRINTS
# =========================
PRINT_EVERY = 10

def _fmt_td(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return str(timedelta(seconds=seconds))

def _progress_line(i: int, total: int, t0: float, last_row: dict | None = None) -> str:
    elapsed = time_mod.time() - t0
    rate = i / elapsed if elapsed > 0 else 0.0
    eta = (total - i) / rate if rate > 0 else 0.0
    pct = (i / total) * 100.0 if total else 0.0

    core = f"[GRID] {i}/{total} ({pct:5.1f}%) | elapsed={_fmt_td(elapsed)} | eta={_fmt_td(eta)} | rate={rate:,.2f} sc/s"
    if not last_row:
        return core

    return (
        core
        + f" | noTP={bool(last_row.get('disable_tp', False))}"
        + f" | score={last_row.get('score', 0):.2f}"
        + f" | net%={last_row.get('netpnl_pct', 0):.2f}"
        + f" | dd%={last_row.get('max_dd_pct', 0):.2f}"
        + f" | pf={last_row.get('profit_factor', 0):.2f}"
        + f" | wr={last_row.get('win_rate', 0):.1f}"
        + f" | trades={last_row.get('trades', 0)}"
    )


# =========================
# DATA FETCH
# =========================
def _oanda_headers() -> Dict[str, str]:
    if not OANDA_TOKEN:
        raise RuntimeError("Missing OANDA_TOKEN env var. Set it before running.")
    return {"Authorization": f"Bearer {OANDA_TOKEN}"}

def fetch_oanda_candles_range(
    instrument: str,
    granularity: str,
    start_utc: datetime,
    end_utc: datetime,
    max_per_request: int = 5000,
    sleep_s: float = 0.25,
) -> pd.DataFrame:
    url = f"{OANDA_API_BASE}/instruments/{instrument}/candles"
    headers = _oanda_headers()

    all_rows = []
    cur = start_utc
    chunk_minutes = max_per_request * 5

    while cur < end_utc:
        nxt = min(cur + timedelta(minutes=chunk_minutes), end_utc)
        params = {
            "granularity": granularity,
            "price": "M",
            "from": cur.isoformat().replace("+00:00", "Z"),
            "to": nxt.isoformat().replace("+00:00", "Z"),
        }
        r = requests.get(url, headers=headers, params=params, timeout=(10, 30))
        if r.status_code != 200:
            raise RuntimeError(f"OANDA fetch failed: {r.status_code} {r.text[:400]}")

        candles = r.json().get("candles", [])
        for c in candles:
            if not c.get("complete", False):
                continue
            t = pd.to_datetime(c["time"], utc=True)
            mid = c["mid"]
            all_rows.append((t, float(mid["o"]), float(mid["h"]), float(mid["l"]), float(mid["c"]), int(c.get("volume", 0))))

        print(f"[DATA] {instrument} {granularity}: fetched {cur.date()} -> {nxt.date()} rows={len(all_rows)}")
        cur = nxt
        time_mod.sleep(sleep_s)

    df = pd.DataFrame(all_rows, columns=["time", "open", "high", "low", "close", "volume"])
    if df.empty:
        raise RuntimeError("No candles returned.")
    df = df.drop_duplicates(subset=["time"]).sort_values("time").set_index("time")
    return df

def load_or_fetch_candles() -> pd.DataFrame:
    if CANDLES_CACHE_CSV.exists():
        df = pd.read_csv(CANDLES_CACHE_CSV, parse_dates=["time"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        print(f"[DATA] Loaded cached candles: {CANDLES_CACHE_CSV} rows={len(df)}")
        return df

    df = fetch_oanda_candles_range(SYMBOL_OANDA, GRANULARITY, START_DATE_UTC, END_DATE_UTC)
    df.reset_index().to_csv(CANDLES_CACHE_CSV, index=False)
    print(f"[DATA] Saved candles cache: {CANDLES_CACHE_CSV}")
    return df


# =========================
# INDICATORS (HA/ATR/CE)
# =========================
def calculate_heikin_ashi_arrays(O, H, L, C):
    n = len(O)
    ha_close = (O + H + L + C) / 4.0
    ha_open = np.empty(n, dtype=float)
    ha_open[0] = (O[0] + C[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum.reduce([H, ha_open, ha_close])
    ha_low = np.minimum.reduce([L, ha_open, ha_close])
    return ha_open, ha_high, ha_low, ha_close

def rma_atr_from_hlc(h, l, c, n: int) -> np.ndarray:
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
        _, h, l, c = calculate_heikin_ashi_arrays(O, H, L, C)
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

def build_chandelier_stop_arrays(O, H, L, C, atr_arr: np.ndarray, lookback: int, mult: float, use_ha: bool):
    if use_ha:
        _, h, l, c = calculate_heikin_ashi_arrays(O, H, L, C)
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


# =========================
# BACKTEST CORE (FAST + REALISTIC)
# =========================
@dataclass(frozen=True)
class Scenario:
    stop_mode: str
    atr_period: int
    atr_init_mult: float
    tp_r: float
    trail_start_r: float
    atr_trail_mult: float
    chan_lookback: int
    max_hold_bars: Optional[int]
    dynamic_risk: bool
    disable_tp: bool   # <==== NEW GRID DIMENSION


def _apply_spread(price: float, side: str, is_entry: bool) -> float:
    if not USE_SPREAD_COST:
        return price
    half = SPREAD_USD / 2.0
    if side == "BUY":
        return price + half if is_entry else price - half
    else:
        return price - half if is_entry else price + half

def _apply_slippage(price: float, side: str, is_entry: bool, atr_now: float) -> float:
    if (not USE_SLIPPAGE) or (atr_now <= 0) or (not np.isfinite(atr_now)):
        return price
    k = SLIPPAGE_ENTRY_ATR_MULT if is_entry else SLIPPAGE_EXIT_ATR_MULT
    slip = k * atr_now
    if side == "BUY":
        return price + slip if is_entry else price - slip
    else:
        return price - slip if is_entry else price + slip

def _lots_for_risk(equity: float, sl_dist: float, risk_pct: float) -> tuple[float, bool, bool]:
    risk_amount = equity * (risk_pct / 100.0)
    risk_per_lot = max(1e-9, sl_dist * CONTRACT_SIZE)
    raw_lots = risk_amount / risk_per_lot

    lots = raw_lots
    capped_lots = False
    if MAX_LOTS_CAP is not None:
        lots2 = min(lots, float(MAX_LOTS_CAP))
        capped_lots = lots2 < lots - 1e-12
        lots = lots2

    return max(0.0, lots), capped_lots, False

def _apply_notional_cap(lots: float, entry_price: float) -> tuple[float, bool]:
    if (MAX_NOTIONAL_USD is None) or (MAX_NOTIONAL_USD <= 0):
        return lots, False
    max_lots_notional = float(MAX_NOTIONAL_USD) / max(1e-9, entry_price * CONTRACT_SIZE)
    lots2 = min(lots, max_lots_notional)
    capped = lots2 < lots - 1e-12
    return max(0.0, lots2), capped

def _trade_pnl(side: str, entry: float, exit: float, lots: float) -> float:
    direction = 1.0 if side == "BUY" else -1.0
    return (exit - entry) * direction * CONTRACT_SIZE * lots

def _trade_r(pnl_cash: float, sl_cash: float) -> float:
    return pnl_cash / sl_cash if sl_cash > 0 else 0.0


def run_backtest_fast(
    idx, O, H, L, NEXT_O,
    buy_sig, sell_sig,
    atr_arr: np.ndarray,
    sc: Scenario,
    chand_lss: Optional[np.ndarray] = None,
    chand_sss: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:

    balance = INITIAL_BALANCE
    equity = INITIAL_BALANCE
    peak = INITIAL_BALANCE

    in_pos = False
    side = None
    entry_i = -1
    entry_price = 0.0
    lots = 0.0
    lots_capped = False
    notional_capped = False
    sl_price = None
    tp_price = None
    init_sl_dist = 0.0
    highest = -1e18
    lowest = 1e18

    mae = 0.0
    mfe = 0.0

    skipped_min_sl = 0

    trades = []
    eq_times = []
    eq_vals = []
    dd_vals = []

    n = len(O)

    def mark_equity(i: int, eq_mark: float):
        nonlocal peak
        if eq_mark > peak:
            peak = eq_mark
        dd = (eq_mark - peak) / peak * 100.0 if peak > 0 else 0.0
        eq_times.append(idx[i])
        eq_vals.append(eq_mark)
        dd_vals.append(dd)

    for i in range(n - 1):
        if USE_MTM_WORST_DD and in_pos:
            atr_now = float(atr_arr[i]) if np.isfinite(atr_arr[i]) else 0.0
            if side == "BUY":
                mark_px = _apply_spread(L[i], "BUY", is_entry=False)
                mark_px = _apply_slippage(mark_px, "BUY", is_entry=False, atr_now=atr_now)
            else:
                mark_px = _apply_spread(H[i], "SELL", is_entry=False)
                mark_px = _apply_slippage(mark_px, "SELL", is_entry=False, atr_now=atr_now)
            unreal = _trade_pnl(side, entry_price, mark_px, lots)
            mark_equity(i, balance + unreal)
        else:
            mark_equity(i, equity)

        sig = "BUY" if buy_sig[i] else ("SELL" if sell_sig[i] else None)
        h = H[i]
        l = L[i]
        nxt_o = NEXT_O[i]

        atr_now = float(atr_arr[i]) if np.isfinite(atr_arr[i]) else 0.0

        # ---- manage open trade intrabar ----
        if in_pos:
            if side == "BUY":
                highest = max(highest, h)
                lowest = min(lowest, l)
                mae = min(mae, _trade_pnl(side, entry_price, l, lots))
                mfe = max(mfe, _trade_pnl(side, entry_price, h, lots))
            else:
                highest = max(highest, h)
                lowest = min(lowest, l)
                mae = min(mae, _trade_pnl(side, entry_price, h, lots))
                mfe = max(mfe, _trade_pnl(side, entry_price, l, lots))

            if sc.stop_mode == "atr_trailing":
                if init_sl_dist > 0:
                    moved = (highest - entry_price) if side == "BUY" else (entry_price - lowest)
                    if moved >= sc.trail_start_r * init_sl_dist and atr_now > 0:
                        trail_dist = sc.atr_trail_mult * atr_now
                        if side == "BUY":
                            sl_price = max(sl_price, highest - trail_dist)
                        else:
                            sl_price = min(sl_price, lowest + trail_dist)

            elif sc.stop_mode == "chandelier":
                if chand_lss is not None and chand_sss is not None:
                    if side == "BUY":
                        ch = chand_lss[i]
                        if np.isfinite(ch):
                            sl_price = max(sl_price, ch)
                    else:
                        ch = chand_sss[i]
                        if np.isfinite(ch):
                            sl_price = min(sl_price, ch)

            sl_hit = tp_hit = False
            sl_exec = tp_exec = None

            if sl_price is not None:
                if side == "BUY" and l <= sl_price:
                    sl_hit = True; sl_exec = sl_price
                if side == "SELL" and h >= sl_price:
                    sl_hit = True; sl_exec = sl_price

            if tp_price is not None:
                if side == "BUY" and h >= tp_price:
                    tp_hit = True; tp_exec = tp_price
                if side == "SELL" and l <= tp_price:
                    tp_hit = True; tp_exec = tp_price

            if sl_hit or tp_hit:
                raw_exit = sl_exec if sl_hit else tp_exec
                exit_px = _apply_spread(raw_exit, side, is_entry=False)
                exit_px = _apply_slippage(exit_px, side, is_entry=False, atr_now=atr_now)

                pnl = _trade_pnl(side, entry_price, exit_px, lots)
                sl_cash = init_sl_dist * CONTRACT_SIZE * lots

                reason = "TP" if tp_hit else ("TRAIL_STOP" if pnl > 0 else "SL")

                notional = entry_price * CONTRACT_SIZE * lots

                trades.append((
                    idx[entry_i], idx[i], side, entry_price, exit_px, lots,
                    float(init_sl_dist), bool(lots_capped), bool(notional_capped),
                    float(notional),
                    bool(sc.disable_tp),
                    pnl, _trade_r(pnl, sl_cash), mae, mfe, reason
                ))

                balance += pnl
                equity = balance

                in_pos = False
                side = None
                sl_price = None
                tp_price = None
                init_sl_dist = 0.0
                lots = 0.0
                lots_capped = False
                notional_capped = False
                mae = 0.0
                mfe = 0.0

        # ---- signal acts at next open (flip/entry) ----
        if sig is not None:
            if in_pos and sig != side:
                atr_fill = float(atr_arr[i]) if np.isfinite(atr_arr[i]) else 0.0
                exit_px = _apply_spread(nxt_o, side, is_entry=False)
                exit_px = _apply_slippage(exit_px, side, is_entry=False, atr_now=atr_fill)

                pnl = _trade_pnl(side, entry_price, exit_px, lots)
                sl_cash = init_sl_dist * CONTRACT_SIZE * lots

                notional = entry_price * CONTRACT_SIZE * lots

                trades.append((
                    idx[entry_i], idx[i+1], side, entry_price, exit_px, lots,
                    float(init_sl_dist), bool(lots_capped), bool(notional_capped),
                    float(notional),
                    bool(sc.disable_tp),
                    pnl, _trade_r(pnl, sl_cash), mae, mfe, "FLIP"
                ))

                balance += pnl
                equity = balance

                in_pos = False
                side = None
                sl_price = None
                tp_price = None
                init_sl_dist = 0.0
                lots = 0.0
                lots_capped = False
                notional_capped = False
                mae = 0.0
                mfe = 0.0

            if not in_pos:
                side = sig
                atr_fill = float(atr_arr[i]) if np.isfinite(atr_arr[i]) else 0.0

                entry_px = _apply_spread(nxt_o, side, is_entry=True)
                entry_px = _apply_slippage(entry_px, side, is_entry=True, atr_now=atr_fill)

                atr_entry = atr_fill

                if sc.stop_mode == "chandelier" and chand_lss is not None and chand_sss is not None:
                    ch0 = chand_lss[i] if side == "BUY" else chand_sss[i]
                    if np.isfinite(ch0):
                        init_sl_dist = abs(entry_px - ch0)
                        sl_price = ch0
                    else:
                        init_sl_dist = max(1e-9, sc.atr_init_mult * atr_entry)
                        sl_price = entry_px - init_sl_dist if side == "BUY" else entry_px + init_sl_dist
                else:
                    init_sl_dist = max(1e-9, sc.atr_init_mult * atr_entry)
                    sl_price = entry_px - init_sl_dist if side == "BUY" else entry_px + init_sl_dist

                if init_sl_dist < MIN_SL_DIST_USD:
                    skipped_min_sl += 1
                    side = None
                    continue

                tp_price = None

                # TP (disabled if sc.disable_tp)
                if sc.disable_tp:
                    tp_price = None
                else:
                    if sc.tp_r is not None and sc.tp_r > 0:
                        tp_dist = sc.tp_r * init_sl_dist
                        tp_price = entry_px + tp_dist if side == "BUY" else entry_px - tp_dist
                    else:
                        tp_price = None

                base_equity = equity if sc.dynamic_risk else INITIAL_BALANCE
                lots, lots_capped, _ = _lots_for_risk(base_equity, init_sl_dist, RISK_PCT_PER_TRADE)
                lots, notional_capped = _apply_notional_cap(lots, entry_px)

                if lots <= 0:
                    side = None
                    continue

                in_pos = True
                entry_i = i + 1
                entry_price = entry_px
                highest = entry_px
                lowest = entry_px
                mae = 0.0
                mfe = 0.0

    mark_equity(n - 1, equity)

    trades_df = pd.DataFrame(
        trades,
        columns=[
            "entry_time","exit_time","side","entry","exit","lots",
            "sl_dist_usd","lots_capped","notional_capped","notional_usd",
            "disable_tp",
            "pnl_cash","pnl_r","mae_cash","mfe_cash","exit_reason"
        ]
    )
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True)

    eq_df = pd.DataFrame({"time": pd.to_datetime(eq_times, utc=True), "equity": eq_vals, "dd_pct": dd_vals})

    extra = {"skipped_min_sl": float(skipped_min_sl)}
    return trades_df, eq_df, extra


# =========================
# METRICS + FORMATTING
# =========================
def streak_stats(pnls: np.ndarray) -> Tuple[int, int, float, float]:
    max_w = max_l = 0
    cur_w = cur_l = 0
    best_win_run = 0.0
    worst_loss_run = 0.0
    cur_win_sum = 0.0
    cur_loss_sum = 0.0

    for p in pnls:
        if p > 0:
            cur_w += 1; cur_l = 0
            cur_win_sum += p; cur_loss_sum = 0.0
            max_w = max(max_w, cur_w)
            best_win_run = max(best_win_run, cur_win_sum)
        elif p < 0:
            cur_l += 1; cur_w = 0
            cur_loss_sum += p; cur_win_sum = 0.0
            max_l = max(max_l, cur_l)
            worst_loss_run = min(worst_loss_run, cur_loss_sum)
        else:
            cur_w = cur_l = 0
            cur_win_sum = cur_loss_sum = 0.0

    return max_w, max_l, float(worst_loss_run), float(best_win_run)

def compute_metrics(trades_df: pd.DataFrame, eq_df: pd.DataFrame, extra: Dict[str, float]) -> Dict[str, float]:
    if trades_df.empty:
        return {
            "trades": 0, "winners": 0, "win_rate": 0.0,
            "initialBalance": INITIAL_BALANCE, "finalEquity": INITIAL_BALANCE,
            "netpnl": 0.0, "netpnl_pct": 0.0, "max_dd_pct": 0.0,
            "avgpnl": 0.0, "maxWinsStreak": 0, "maxLossesStreak": 0,
            "max_streak_loss_cash": 0.0, "max_streak_profit_cash": 0.0,
            "profit_factor": 0.0, "expectancy": 0.0,
            "skipped_min_sl": float(extra.get("skipped_min_sl", 0.0)),
            "pct_notional_capped": 0.0,
            "avg_notional": 0.0,
            "p95_notional": 0.0,
            "max_notional": 0.0,
        }

    pnls = trades_df["pnl_cash"].to_numpy(dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    winners = int((pnls > 0).sum())
    trades = int(len(pnls))
    win_rate = winners / trades if trades else 0.0

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.sum()) if len(losses) else 0.0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else (9999.0 if gross_profit > 0 else 0.0)

    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)

    final_eq = float(eq_df["equity"].iloc[-1]) if not eq_df.empty else INITIAL_BALANCE
    netpnl = final_eq - INITIAL_BALANCE
    netpnl_pct = (netpnl / INITIAL_BALANCE) * 100.0
    max_dd_pct = float(eq_df["dd_pct"].min()) if not eq_df.empty else 0.0

    max_w, max_l, worst_loss_run, best_win_run = streak_stats(pnls)

    notional = trades_df["notional_usd"].to_numpy(float)
    notional_capped = trades_df["notional_capped"].astype(bool).to_numpy()

    return {
        "trades": trades,
        "winners": winners,
        "win_rate": win_rate * 100.0,
        "initialBalance": INITIAL_BALANCE,
        "finalEquity": final_eq,
        "netpnl": netpnl,
        "netpnl_pct": netpnl_pct,
        "max_dd_pct": max_dd_pct,
        "avgpnl": float(pnls.mean()),
        "maxWinsStreak": max_w,
        "maxLossesStreak": max_l,
        "max_streak_loss_cash": worst_loss_run,
        "max_streak_profit_cash": best_win_run,
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),

        "n_exit_TP": int((trades_df["exit_reason"] == "TP").sum()),

        "skipped_min_sl": float(extra.get("skipped_min_sl", 0.0)),
        "pct_notional_capped": float(np.mean(notional_capped)) * 100.0 if len(notional_capped) else 0.0,
        "avg_notional": float(np.mean(notional)) if len(notional) else 0.0,
        "p95_notional": float(np.percentile(notional, 95)) if len(notional) else 0.0,
        "max_notional": float(np.max(notional)) if len(notional) else 0.0,
    }

def scenario_score(m: Dict[str, float]) -> float:
    net = m["netpnl_pct"]
    dd = abs(m["max_dd_pct"])
    pf = m["profit_factor"]
    wr = m["win_rate"]
    dd = max(0.1, dd)
    return (net / dd) * (1.0 + min(2.0, pf / 2.0)) * (1.0 + (wr / 100.0))

def format_results_for_csv(res: pd.DataFrame) -> pd.DataFrame:
    out = res.copy()

    def fmt2(x): return "" if pd.isna(x) else f"{float(x):.2f}"
    def fmt4(x): return "" if pd.isna(x) else f"{float(x):.4f}"

    for col in ["win_rate", "avgpnl", "profit_factor", "expectancy"]:
        if col in out.columns:
            out[col] = out[col].apply(fmt4)

    for col in [
        "initialBalance","finalEquity","netpnl",
        "max_streak_loss_cash","max_streak_profit_cash",
        "avg_notional","p95_notional","max_notional"
    ]:
        if col in out.columns:
            out[col] = out[col].apply(fmt2)

    for col in ["netpnl_pct", "max_dd_pct", "score", "pct_notional_capped", "skipped_min_sl"]:
        if col in out.columns:
            out[col] = out[col].apply(fmt2)

    return out


# =========================
# MAIN
# =========================
def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_CE_XAU_M5_GRID"
    run_dir = OUT_ROOT / "grid_runs" / run_id
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    df = load_or_fetch_candles()
    df = df.loc[(df.index >= START_DATE_UTC) & (df.index <= END_DATE_UTC)].copy()
    df.reset_index().to_csv(run_dir / "data" / "candles_used.csv", index=False)
    print(f"[DATA] candles used rows={len(df)} from={df.index.min()} to={df.index.max()}")

    idx = df.index.to_numpy()
    O = df["open"].to_numpy(dtype=float)
    H = df["high"].to_numpy(dtype=float)
    L = df["low"].to_numpy(dtype=float)
    C = df["close"].to_numpy(dtype=float)
    NEXT_O = np.roll(O, -1)

    print("[PRECOMP] Building entry signals (once)...")
    _, buy_sig, sell_sig = compute_ce_signals(O, H, L, C, USE_HEIKIN_ASHI, CE_ATR_PERIOD, CE_ATR_MULT)
    buy_sig = buy_sig.astype(bool)
    sell_sig = sell_sig.astype(bool)

    print("[PRECOMP] Building ATR cache per atr_period...")
    if USE_HEIKIN_ASHI:
        _, h_ha, l_ha, c_ha = calculate_heikin_ashi_arrays(O, H, L, C)
        H0, L0, C0 = h_ha, l_ha, c_ha
    else:
        H0, L0, C0 = H, L, C

    atr_cache_np: Dict[int, np.ndarray] = {}
    for ap in GRID_ATR_PERIOD:
        atr_cache_np[ap] = rma_atr_from_hlc(H0, L0, C0, n=int(ap))
    print(f"[PRECOMP] ATR cache built for periods: {sorted(atr_cache_np.keys())}")

    chand_cache: Dict[Tuple[int, float, int], Tuple[np.ndarray, np.ndarray]] = {}
    if "chandelier" in STOP_MODES:
        print("[PRECOMP] Building chandelier stop cache (unique combos only)...")
        combos = sorted({(ap, init_mult, lb) for ap in GRID_ATR_PERIOD for init_mult in GRID_ATR_INIT_MULT for lb in GRID_CHAN_LOOKBACK})
        for ap, init_mult, lb in combos:
            atr_arr = atr_cache_np[ap]
            lss, sss = build_chandelier_stop_arrays(O, H, L, C, atr_arr, lookback=int(lb), mult=float(init_mult), use_ha=USE_HEIKIN_ASHI)
            chand_cache[(int(ap), float(init_mult), int(lb))] = (lss, sss)
        print(f"[PRECOMP] chandelier cache built combos={len(chand_cache)}")

    cfg = {
        "SYMBOL_OANDA": SYMBOL_OANDA,
        "GRANULARITY": GRANULARITY,
        "START_DATE_UTC": START_DATE_UTC.isoformat(),
        "END_DATE_UTC": END_DATE_UTC.isoformat(),
        "INITIAL_BALANCE": INITIAL_BALANCE,
        "RISK_PCT_PER_TRADE": RISK_PCT_PER_TRADE,
        "DYNAMIC_RISK": DYNAMIC_RISK,
        "CONTRACT_SIZE": CONTRACT_SIZE,
        "USE_SPREAD_COST": USE_SPREAD_COST,
        "SPREAD_USD": SPREAD_USD,
        "USE_SLIPPAGE": USE_SLIPPAGE,
        "SLIPPAGE_ENTRY_ATR_MULT": SLIPPAGE_ENTRY_ATR_MULT,
        "SLIPPAGE_EXIT_ATR_MULT": SLIPPAGE_EXIT_ATR_MULT,
        "USE_MTM_WORST_DD": USE_MTM_WORST_DD,
        "MIN_SL_DIST_USD": MIN_SL_DIST_USD,
        "MAX_LOTS_CAP": MAX_LOTS_CAP,
        "MAX_NOTIONAL_USD": MAX_NOTIONAL_USD,
        "GRID_DISABLE_TP": GRID_DISABLE_TP,
        "ENTRY_ENGINE": {"USE_HEIKIN_ASHI": USE_HEIKIN_ASHI, "CE_ATR_PERIOD": CE_ATR_PERIOD, "CE_ATR_MULT": CE_ATR_MULT},
        "GRID": {
            "STOP_MODES": STOP_MODES,
            "GRID_ATR_PERIOD": GRID_ATR_PERIOD,
            "GRID_ATR_INIT_MULT": GRID_ATR_INIT_MULT,
            "GRID_ATR_TRAIL_MULT": GRID_ATR_TRAIL_MULT,
            "GRID_CHAN_LOOKBACK": GRID_CHAN_LOOKBACK,
            "GRID_TP_R": GRID_TP_R,
            "GRID_TRAIL_START_R": GRID_TRAIL_START_R,
            "GRID_MAX_HOLD_BARS": GRID_MAX_HOLD_BARS,
        },
    }
    (run_dir / "logs" / "run_config.json").write_text(json.dumps(cfg, indent=2))

    scenarios: List[Scenario] = []

    for mode in STOP_MODES:
        if mode == "chandelier":
            for ap in GRID_ATR_PERIOD:
                for init_mult in GRID_ATR_INIT_MULT:
                    for tp_r in GRID_TP_R:
                        for chan_lb in GRID_CHAN_LOOKBACK:
                            for disable_tp in GRID_DISABLE_TP:
                                scenarios.append(
                                    Scenario(
                                        stop_mode="chandelier",
                                        atr_period=int(ap),
                                        atr_init_mult=float(init_mult),
                                        tp_r=float(tp_r),
                                        trail_start_r=0.0,
                                        atr_trail_mult=0.0,
                                        chan_lookback=int(chan_lb),
                                        max_hold_bars=None,
                                        dynamic_risk=DYNAMIC_RISK,
                                        disable_tp=bool(disable_tp),
                                    )
                                )
        else:
            for ap in GRID_ATR_PERIOD:
                for init_mult in GRID_ATR_INIT_MULT:
                    for tp_r in GRID_TP_R:
                        for tsr in GRID_TRAIL_START_R:
                            for trail_mult in GRID_ATR_TRAIL_MULT:
                                for disable_tp in GRID_DISABLE_TP:
                                    scenarios.append(
                                        Scenario(
                                            stop_mode=mode,
                                            atr_period=int(ap),
                                            atr_init_mult=float(init_mult),
                                            tp_r=float(tp_r),
                                            trail_start_r=float(tsr),
                                            atr_trail_mult=float(trail_mult),
                                            chan_lookback=0,
                                            max_hold_bars=None,
                                            dynamic_risk=DYNAMIC_RISK,
                                            disable_tp=bool(disable_tp),
                                        )
                                    )

    print(f"[GRID] scenarios={len(scenarios)}")

    rows = []
    t0 = time_mod.time()
    best_score = -1e18
    best_row = None

    for k, sc in enumerate(scenarios, start=1):
        atr_arr = atr_cache_np[sc.atr_period]

        chand_lss = chand_sss = None
        if sc.stop_mode == "chandelier":
            chand_lss, chand_sss = chand_cache[(sc.atr_period, float(sc.atr_init_mult), int(sc.chan_lookback))]

        trades_df, eq_df, extra = run_backtest_fast(
            idx, O, H, L, NEXT_O,
            buy_sig, sell_sig,
            atr_arr,
            sc,
            chand_lss=chand_lss,
            chand_sss=chand_sss,
        )

        m = compute_metrics(trades_df, eq_df, extra)
        score = scenario_score(m)
        row = {**asdict(sc), **m, "score": score}
        rows.append(row)

        if score > best_score:
            best_score = score
            best_row = row

        if (k == 1) or (k % PRINT_EVERY == 0) or (k == len(scenarios)):
            print(_progress_line(k, len(scenarios), t0, last_row=row))

    res = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    res.to_csv(run_dir / "reports" / "scenarios_all_raw.csv", index=False)

    res_fmt = format_results_for_csv(res)
    res_fmt.to_csv(run_dir / "reports" / "scenarios_all.csv", index=False)

    res_fmt.head(200).to_csv(run_dir / "reports" / "scenarios_top_by_score.csv", index=False)
    res_fmt.sort_values("profit_factor", ascending=False).head(200).to_csv(run_dir / "reports" / "scenarios_top_by_pf.csv", index=False)
    res_fmt.sort_values("netpnl_pct", ascending=False).head(200).to_csv(run_dir / "reports" / "scenarios_top_by_netpct.csv", index=False)

    print(f"[DONE] saved: {run_dir}")
    print("[NOTE] scenarios_all_raw.csv = numeric. scenarios_all.csv = formatted (no sci notation).")


if __name__ == "__main__":
    main()
