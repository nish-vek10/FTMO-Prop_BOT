#!/usr/bin/env python
"""
run_single_ce_xau_m5.py

Run ONE chosen scenario (from grid) and output:
- summary.json
- trades.csv
- equity_curve.csv
- plots:
    equity.equity.png
    equity.dd_pct.png                 (DD line in RED)
    equity.equity_dd.png              (combined, DD line in RED)
    heatmap_pnl_by_dow_hour.png
    heatmap_mae_by_dow_hour.png
    heatmap_winrate_by_dow_hour.png

Supports trading_window filter (exclude trades opened in certain hours)
to test removing "bad hours" seen in heatmaps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(r"C:\Users\anish\PycharmProjects\PropEAs")
BT_ROOT = PROJECT_ROOT / "backtests"
OUT_ROOT = BT_ROOT / "output"

DATA_DIR = BT_ROOT / "data_cache"
SYMBOL_OANDA = "XAU_USD"
GRANULARITY = "M5"
CANDLES_CACHE_CSV = DATA_DIR / f"candles_{SYMBOL_OANDA}_{GRANULARITY}_2025_to_today.csv"


# =========================
# STRATEGY / ACCOUNT CONFIG
# =========================
INITIAL_BALANCE = 100_000.0
RISK_PCT_PER_TRADE = 0.5
DYNAMIC_RISK = False
CONTRACT_SIZE = 100.0

# Costs
USE_SPREAD_COST = True
SPREAD_USD = 0.20

# Slippage (adverse)
USE_SLIPPAGE = True
SLIPPAGE_ENTRY_ATR_MULT = 0.05
SLIPPAGE_EXIT_ATR_MULT  = 0.05

# Mark-to-market DD (worst-case intrabar)
USE_MTM_WORST_DD = True

# Entry engine
USE_HEIKIN_ASHI = True
CE_ATR_PERIOD = 1
CE_ATR_MULT = 1.85

# Guards
MIN_SL_DIST_USD = 0.50
MAX_LOTS_CAP = 50.0
MAX_NOTIONAL_USD = 500_000.0


# =========================
# PICK YOUR SCENARIO HERE
# =========================
STOP_MODE = "chandelier"
ATR_PERIOD = 20
ATR_INIT_MULT = 1.0
TP_R = 1.5
TRAIL_START_R = 0.0
ATR_TRAIL_MULT = 0.0
CHAN_LOOKBACK = 180
MAX_HOLD_BARS = None

# Trend ride mode (no TP)
DISABLE_TP = False


# =========================
# TRADING WINDOW FILTER (UTC)
# =========================
TRADING_WINDOWS_UTC: List[Tuple[str, str]] = [
    # ("07:00", "12:00"),
    # ("13:00", "18:00"),
]


# =========================
# HELPERS
# =========================
def _parse_hhmm(x: str) -> dtime:
    h, m = map(int, x.split(":"))
    return dtime(hour=h, minute=m)

def _in_windows_utc(ts: pd.Timestamp, windows: List[Tuple[str, str]]) -> bool:
    if not windows:
        return True
    t = ts.to_pydatetime().time()
    for a, b in windows:
        ta = _parse_hhmm(a)
        tb = _parse_hhmm(b)
        if ta <= t < tb:
            return True
    return False


def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_df = pd.DataFrame(index=df.index)
    ha_df["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0

    ha_open = np.zeros(len(df), dtype=float)
    ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_df["ha_close"].iloc[i - 1]) / 2.0

    ha_df["ha_open"] = ha_open
    ha_df["ha_high"] = pd.concat([df["high"], ha_df["ha_open"], ha_df["ha_close"]], axis=1).max(axis=1)
    ha_df["ha_low"] = pd.concat([df["low"], ha_df["ha_open"], ha_df["ha_close"]], axis=1).min(axis=1)
    return ha_df


def rma_atr(true_range: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(true_range, np.nan, dtype=float)
    if n <= 1:
        return true_range.astype(float)
    if len(true_range) < n:
        return out
    seed = np.nanmean(true_range[:n])
    out[n - 1] = seed
    alpha = 1.0 / n
    for i in range(n, len(true_range)):
        out[i] = out[i - 1] + alpha * (true_range[i] - out[i - 1])
    return pd.Series(out).ffill().bfill().to_numpy(dtype=float)


def compute_ce_engine(df: pd.DataFrame, use_ha: bool, atr_period: int, atr_mult: float) -> pd.DataFrame:
    if use_ha:
        ha = calculate_heikin_ashi(df)
        h = ha["ha_high"].to_numpy()
        l = ha["ha_low"].to_numpy()
        c = ha["ha_close"].to_numpy()
        o = ha["ha_open"].to_numpy()
    else:
        o = df["open"].to_numpy()
        h = df["high"].to_numpy()
        l = df["low"].to_numpy()
        c = df["close"].to_numpy()

    c_prev = np.roll(c, 1)
    c_prev[0] = np.nan

    tr = np.zeros_like(c, dtype=float)
    for i in range(len(c)):
        if i == 0 or np.isnan(c_prev[i]):
            tr[i] = h[i] - l[i]
        else:
            tr[i] = max(h[i] - l[i], abs(h[i] - c_prev[i]), abs(l[i] - c_prev[i]))

    n = int(max(1, atr_period))
    atr = rma_atr(tr, n) if n > 1 else tr.copy()
    atr_val = atr_mult * atr

    hh = pd.Series(h, index=df.index).rolling(window=n, min_periods=n).max().to_numpy()
    ll = pd.Series(l, index=df.index).rolling(window=n, min_periods=n).min().to_numpy()

    long_stop = hh - atr_val
    short_stop = ll + atr_val

    lss = long_stop.copy()
    sss = short_stop.copy()

    for i in range(1, len(c)):
        long_prev = lss[i - 1] if not np.isnan(lss[i - 1]) else long_stop[i]
        short_prev = sss[i - 1] if not np.isnan(sss[i - 1]) else short_stop[i]

        if c[i - 1] > long_prev:
            lss[i] = max(long_stop[i], long_prev)
        else:
            lss[i] = long_stop[i]

        if c[i - 1] < short_prev:
            sss[i] = min(short_stop[i], short_prev)
        else:
            sss[i] = short_stop[i]

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

    return pd.DataFrame(
        {
            "o": o, "h": h, "l": l, "c": c,
            "atr": atr,
            "dir": dir_vals,
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
        },
        index=df.index,
    )


def build_atr_series(df: pd.DataFrame, atr_period: int) -> pd.Series:
    if USE_HEIKIN_ASHI:
        ha = calculate_heikin_ashi(df)
        h = ha["ha_high"].to_numpy()
        l = ha["ha_low"].to_numpy()
        c = ha["ha_close"].to_numpy()
    else:
        h = df["high"].to_numpy()
        l = df["low"].to_numpy()
        c = df["close"].to_numpy()

    c_prev = np.roll(c, 1)
    c_prev[0] = np.nan
    tr = np.zeros_like(c, dtype=float)
    for i in range(len(c)):
        if i == 0 or np.isnan(c_prev[i]):
            tr[i] = h[i] - l[i]
        else:
            tr[i] = max(h[i] - l[i], abs(h[i] - c_prev[i]), abs(l[i] - c_prev[i]))

    n = int(max(1, atr_period))
    atr = rma_atr(tr, n) if n > 1 else tr.copy()
    return pd.Series(atr, index=df.index)


def build_chandelier_stops(df: pd.DataFrame, atr_period: int, atr_mult: float, lookback: int) -> pd.DataFrame:
    if USE_HEIKIN_ASHI:
        ha = calculate_heikin_ashi(df)
        h = ha["ha_high"].to_numpy(dtype=float)
        l = ha["ha_low"].to_numpy(dtype=float)
        c = ha["ha_close"].to_numpy(dtype=float)
    else:
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)

    c_prev = np.roll(c, 1)
    c_prev[0] = np.nan
    tr = np.zeros_like(c, dtype=float)
    for i in range(len(c)):
        if i == 0 or np.isnan(c_prev[i]):
            tr[i] = h[i] - l[i]
        else:
            tr[i] = max(h[i] - l[i], abs(h[i] - c_prev[i]), abs(l[i] - c_prev[i]))

    n = int(max(1, atr_period))
    atr = rma_atr(tr, n) if n > 1 else tr.copy()

    lb = int(max(1, lookback))
    hh = pd.Series(h, index=df.index).rolling(window=lb, min_periods=lb).max().to_numpy()
    ll = pd.Series(l, index=df.index).rolling(window=lb, min_periods=lb).min().to_numpy()

    long_stop = hh - (atr_mult * atr)
    short_stop = ll + (atr_mult * atr)

    lss = long_stop.copy()
    sss = short_stop.copy()

    for i in range(1, len(c)):
        long_prev = lss[i - 1] if not np.isnan(lss[i - 1]) else long_stop[i]
        short_prev = sss[i - 1] if not np.isnan(sss[i - 1]) else short_stop[i]

        if c[i - 1] > long_prev:
            lss[i] = max(long_stop[i], long_prev)
        else:
            lss[i] = long_stop[i]

        if c[i - 1] < short_prev:
            sss[i] = min(short_stop[i], short_prev)
        else:
            sss[i] = short_stop[i]

    return pd.DataFrame({"atr": atr, "long_stop_smooth": lss, "short_stop_smooth": sss}, index=df.index)


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

def _lots_for_risk(equity: float, sl_dist: float, risk_pct: float) -> tuple[float, bool]:
    risk_amount = equity * (risk_pct / 100.0)
    risk_per_lot = max(1e-9, sl_dist * CONTRACT_SIZE)
    raw_lots = risk_amount / risk_per_lot

    lots = raw_lots
    capped = False
    if MAX_LOTS_CAP is not None:
        lots2 = min(lots, float(MAX_LOTS_CAP))
        capped = lots2 < lots - 1e-12
        lots = lots2

    return max(0.0, lots), capped

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

def streak_stats(pnls: np.ndarray) -> tuple[int, int, float, float]:
    """
    Returns:
      max_wins_streak (int),
      max_losses_streak (int),
      worst_loss_run_cash (float, negative),
      best_win_run_cash (float, positive)
    """
    max_w = max_l = 0
    cur_w = cur_l = 0

    best_win_run = 0.0
    worst_loss_run = 0.0

    cur_win_sum = 0.0
    cur_loss_sum = 0.0

    for p in pnls:
        if p > 0:
            cur_w += 1
            cur_l = 0

            cur_win_sum += p
            cur_loss_sum = 0.0

            max_w = max(max_w, cur_w)
            best_win_run = max(best_win_run, cur_win_sum)

        elif p < 0:
            cur_l += 1
            cur_w = 0

            cur_loss_sum += p
            cur_win_sum = 0.0

            max_l = max(max_l, cur_l)
            worst_loss_run = min(worst_loss_run, cur_loss_sum)

        else:
            cur_w = cur_l = 0
            cur_win_sum = 0.0
            cur_loss_sum = 0.0

    return max_w, max_l, float(worst_loss_run), float(best_win_run)


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


def run_one(df: pd.DataFrame, eng_entry: pd.DataFrame, sc: Scenario) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    balance = INITIAL_BALANCE
    equity = INITIAL_BALANCE
    peak = INITIAL_BALANCE

    sc_atr = build_atr_series(df, sc.atr_period)
    chan = None
    if sc.stop_mode == "chandelier":
        chan = build_chandelier_stops(df, sc.atr_period, sc.atr_init_mult, sc.chan_lookback)

    in_pos = False
    side = None
    entry_time = None
    entry_price = None
    lots = 0.0
    lots_capped = False
    notional_capped = False

    sl_price = None
    tp_price = None
    init_sl_dist = None

    mae = 0.0
    mfe = 0.0
    highest = None
    lowest = None

    skipped_min_sl = 0

    trades = []
    eq_rows = []

    buy_sig = eng_entry["buy_signal"].to_numpy(dtype=bool)
    sell_sig = eng_entry["sell_signal"].to_numpy(dtype=bool)

    for i in range(len(df) - 1):
        t = df.index[i]
        row = df.iloc[i]
        o, h, l, c = float(row.open), float(row.high), float(row.low), float(row.close)
        nxt_o = float(df.iloc[i + 1].open)
        nxt_t = df.index[i + 1]

        # mark-to-market worst dd
        if USE_MTM_WORST_DD and in_pos:
            atr_now = float(sc_atr.iloc[i]) if np.isfinite(float(sc_atr.iloc[i])) else 0.0
            if side == "BUY":
                mark_px = _apply_spread(l, "BUY", is_entry=False)
                mark_px = _apply_slippage(mark_px, "BUY", is_entry=False, atr_now=atr_now)
            else:
                mark_px = _apply_spread(h, "SELL", is_entry=False)
                mark_px = _apply_slippage(mark_px, "SELL", is_entry=False, atr_now=atr_now)
            unreal = _trade_pnl(side, entry_price, mark_px, lots)
            eq_mark = balance + unreal
        else:
            eq_mark = equity

        peak = max(peak, eq_mark)
        dd_pct = (eq_mark - peak) / peak * 100.0 if peak > 0 else 0.0
        eq_rows.append((t, eq_mark, dd_pct))

        sig = "BUY" if buy_sig[i] else ("SELL" if sell_sig[i] else None)

        # ---- manage open trade ----
        if in_pos:
            atr_now = float(sc_atr.iloc[i]) if np.isfinite(float(sc_atr.iloc[i])) else 0.0

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
                if init_sl_dist and init_sl_dist > 0 and atr_now > 0:
                    moved = (highest - entry_price) if side == "BUY" else (entry_price - lowest)
                    if moved >= sc.trail_start_r * init_sl_dist:
                        trail_dist = sc.atr_trail_mult * atr_now
                        if side == "BUY":
                            sl_price = max(sl_price, highest - trail_dist)
                        else:
                            sl_price = min(sl_price, lowest + trail_dist)

            elif sc.stop_mode == "chandelier" and chan is not None:
                if side == "BUY":
                    ch = float(chan.iloc[i].long_stop_smooth)
                    if np.isfinite(ch):
                        sl_price = max(sl_price, ch)
                else:
                    ch = float(chan.iloc[i].short_stop_smooth)
                    if np.isfinite(ch):
                        sl_price = min(sl_price, ch)

            # SL/TP hits on this bar (worst-case if both hit -> SL first)
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
                sl_cash = (init_sl_dist or 0.0) * CONTRACT_SIZE * lots

                reason = "TP"
                if sl_hit:
                    reason = "TRAIL_STOP" if pnl > 0 else "SL"

                notional = entry_price * CONTRACT_SIZE * lots

                trades.append((
                    entry_time, t, side, entry_price, exit_px, lots,
                    float(init_sl_dist or 0.0), bool(lots_capped), bool(notional_capped),
                    float(notional),
                    pnl, _trade_r(pnl, sl_cash), mae, mfe, reason
                ))

                balance += pnl
                equity = balance

                in_pos = False
                side = None
                entry_time = None
                entry_price = None
                lots = 0.0
                lots_capped = False
                notional_capped = False
                sl_price = None
                tp_price = None
                init_sl_dist = None
                highest = None
                lowest = None
                mae = 0.0
                mfe = 0.0

        # ---- signal acts at next open ----
        if sig is not None:
            # flip
            if in_pos and sig != side:
                atr_fill = float(sc_atr.iloc[i]) if np.isfinite(float(sc_atr.iloc[i])) else 0.0
                exit_px = _apply_spread(nxt_o, side, is_entry=False)
                exit_px = _apply_slippage(exit_px, side, is_entry=False, atr_now=atr_fill)

                pnl = _trade_pnl(side, entry_price, exit_px, lots)
                sl_cash = (init_sl_dist or 0.0) * CONTRACT_SIZE * lots

                notional = entry_price * CONTRACT_SIZE * lots

                trades.append((
                    entry_time, nxt_t, side, entry_price, exit_px, lots,
                    float(init_sl_dist or 0.0), bool(lots_capped), bool(notional_capped),
                    float(notional),
                    pnl, _trade_r(pnl, sl_cash), mae, mfe, "FLIP"
                ))

                balance += pnl
                equity = balance

                in_pos = False
                side = None
                entry_time = None
                entry_price = None
                lots = 0.0
                lots_capped = False
                notional_capped = False
                sl_price = None
                tp_price = None
                init_sl_dist = None
                highest = None
                lowest = None
                mae = 0.0
                mfe = 0.0

            # entry
            if not in_pos:
                if not _in_windows_utc(nxt_t, TRADING_WINDOWS_UTC):
                    continue

                side = sig
                atr_fill = float(sc_atr.iloc[i]) if np.isfinite(float(sc_atr.iloc[i])) else 0.0

                entry_px = _apply_spread(nxt_o, side, is_entry=True)
                entry_px = _apply_slippage(entry_px, side, is_entry=True, atr_now=atr_fill)

                atr_entry = atr_fill

                if sc.stop_mode in ("atr_static", "atr_trailing", "atr_static_time"):
                    init_sl_dist = max(1e-9, sc.atr_init_mult * atr_entry)
                    sl_price = entry_px - init_sl_dist if side == "BUY" else entry_px + init_sl_dist

                elif sc.stop_mode == "chandelier":
                    if chan is None:
                        raise RuntimeError("Chandelier stops not built.")
                    if side == "BUY":
                        ch = float(chan.iloc[i].long_stop_smooth)
                        if np.isnan(ch):
                            init_sl_dist = max(1e-9, sc.atr_init_mult * atr_entry)
                            sl_price = entry_px - init_sl_dist
                        else:
                            sl_price = ch
                            init_sl_dist = abs(entry_px - sl_price)
                    else:
                        ch = float(chan.iloc[i].short_stop_smooth)
                        if np.isnan(ch):
                            init_sl_dist = max(1e-9, sc.atr_init_mult * atr_entry)
                            sl_price = entry_px + init_sl_dist
                        else:
                            sl_price = ch
                            init_sl_dist = abs(entry_px - sl_price)
                else:
                    raise ValueError(f"Unknown stop_mode: {sc.stop_mode}")

                if init_sl_dist < MIN_SL_DIST_USD:
                    skipped_min_sl += 1
                    side = None
                    continue

                # TP (optional): if DISABLE_TP=True, we never set a TP (trend ride)
                if DISABLE_TP:
                    tp_price = None
                else:
                    if sc.tp_r is not None and sc.tp_r > 0:
                        tp_dist = sc.tp_r * init_sl_dist
                        tp_price = entry_px + tp_dist if side == "BUY" else entry_px - tp_dist
                    else:
                        tp_price = None

                base_eq = equity if sc.dynamic_risk else INITIAL_BALANCE
                lots, lots_capped = _lots_for_risk(base_eq, init_sl_dist, RISK_PCT_PER_TRADE)

                lots, notional_capped = _apply_notional_cap(lots, entry_px)

                if lots <= 0:
                    side = None
                    continue

                in_pos = True
                entry_time = nxt_t
                entry_price = entry_px
                highest = entry_price
                lowest = entry_price
                mae = 0.0
                mfe = 0.0

    # final row
    eq_rows.append((df.index[-1], equity, (equity - peak) / peak * 100 if peak > 0 else 0.0))

    trades_df = pd.DataFrame(
        trades,
        columns=[
            "entry_time","exit_time","side","entry","exit","lots",
            "sl_dist_usd","lots_capped","notional_capped","notional_usd",
            "pnl_cash","pnl_r","mae_cash","mfe_cash","exit_reason"
        ],
    )
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True)

    eq_df = pd.DataFrame(eq_rows, columns=["time","equity","dd_pct"])
    eq_df["time"] = pd.to_datetime(eq_df["time"], utc=True)
    return trades_df, eq_df, skipped_min_sl


def compute_summary(trades_df: pd.DataFrame, eq_df: pd.DataFrame, skipped_min_sl: int) -> dict:
    if trades_df.empty:
        return {
            "trades": 0,
            "finalEquity": INITIAL_BALANCE,
            "netpnl": 0.0,
            "netpnl_pct": 0.0,
            "skipped_min_sl": int(skipped_min_sl),
        }

    pnls = trades_df["pnl_cash"].to_numpy(float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    max_w, max_l, worst_loss_run, best_win_run = streak_stats(pnls)

    winners = int((pnls > 0).sum())
    trades = int(len(pnls))
    win_rate = winners / trades * 100.0 if trades else 0.0

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.sum()) if len(losses) else 0.0
    pf = gross_profit / abs(gross_loss) if gross_loss < 0 else (9999.0 if gross_profit > 0 else 0.0)

    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    expectancy = (win_rate/100.0)*avg_win + (1.0-win_rate/100.0)*avg_loss

    final_eq = float(eq_df["equity"].iloc[-1])
    net = final_eq - INITIAL_BALANCE
    net_pct = net / INITIAL_BALANCE * 100.0
    max_dd_pct = float(eq_df["dd_pct"].min())

    lots = trades_df["lots"].to_numpy(float)
    sls = trades_df["sl_dist_usd"].to_numpy(float)
    capped_lots = trades_df["lots_capped"].astype(bool).to_numpy()
    capped_notional = trades_df["notional_capped"].astype(bool).to_numpy()
    notional = trades_df["notional_usd"].to_numpy(float)
    rvals = trades_df["pnl_r"].to_numpy(float)

    def pct(x: float) -> float:
        return float(x) * 100.0

    return {
        "trades": trades,
        "winners": winners,
        "win_rate": win_rate,
        "initialBalance": INITIAL_BALANCE,
        "finalEquity": final_eq,
        "netpnl": net,
        "netpnl_pct": net_pct,
        "max_dd_pct": max_dd_pct,
        "avgpnl": float(pnls.mean()),
        "profit_factor": float(pf),
        "expectancy": float(expectancy),

        "maxWinsStreak": int(max_w),
        "maxLossesStreak": int(max_l),
        "max_streak_loss_cash": float(worst_loss_run),
        "max_streak_profit_cash": float(best_win_run),

        "skipped_min_sl": int(skipped_min_sl),

        "avg_lots": float(np.mean(lots)) if len(lots) else 0.0,
        "p95_lots": float(np.percentile(lots, 95)) if len(lots) else 0.0,
        "max_lots": float(np.max(lots)) if len(lots) else 0.0,
        "pct_lots_capped": pct(np.mean(capped_lots)) if len(capped_lots) else 0.0,

        "pct_notional_capped": pct(np.mean(capped_notional)) if len(capped_notional) else 0.0,
        "avg_notional_usd": float(np.mean(notional)) if len(notional) else 0.0,
        "p95_notional_usd": float(np.percentile(notional, 95)) if len(notional) else 0.0,
        "max_notional_usd": float(np.max(notional)) if len(notional) else 0.0,

        "avg_sl_dist_usd": float(np.mean(sls)) if len(sls) else 0.0,
        "p10_sl_dist_usd": float(np.percentile(sls, 10)) if len(sls) else 0.0,
        "min_sl_dist_usd": float(np.min(sls)) if len(sls) else 0.0,

        "avg_R": float(np.mean(rvals)) if len(rvals) else 0.0,
        "median_R": float(np.median(rvals)) if len(rvals) else 0.0,
        "cum_R": float(np.sum(rvals)) if len(rvals) else 0.0,
    }


def plot_equity_dd(eq_df: pd.DataFrame, outpath: Path, title: str):
    t = eq_df["time"]
    eq = eq_df["equity"]
    dd = eq_df["dd_pct"]

    plt.figure(figsize=(12, 6))
    plt.plot(t, eq, linewidth=2.0)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity (MTM if enabled)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath.with_suffix(".equity.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(12, 3.8))
    plt.plot(t, dd, linewidth=2.0, color="red")
    plt.title(title + " — Drawdown % (MTM if enabled)")
    plt.xlabel("Time")
    plt.ylabel("Drawdown (%)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath.with_suffix(".dd_pct.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(12, 7.5))
    ax1 = plt.gca()
    ax1.plot(t, eq, linewidth=2.2)
    ax1.set_ylabel("Equity")
    ax1.grid(True, alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(t, dd, linewidth=1.8, color="red")
    ax2.set_ylabel("Drawdown %")
    plt.title(title + " — Equity + Drawdown%")
    plt.tight_layout()
    plt.savefig(outpath.with_suffix(".equity_dd.png"), dpi=180)
    plt.close()


def _heatmap_by_dow_hour(values: pd.Series, times: pd.Series, agg: str = "mean") -> pd.DataFrame:
    df = pd.DataFrame({"time": times, "v": values})
    df["dow"] = df["time"].dt.dayofweek
    df["hour"] = df["time"].dt.hour
    pivot = df.pivot_table(index="dow", columns="hour", values="v", aggfunc=agg)
    return pivot.reindex(index=range(7), columns=range(24))


def plot_heatmap(pivot: pd.DataFrame, outpath: Path, title: str, mode: str):
    data = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(14, 5.5))
    ax = plt.gca()

    if mode == "pnl":
        vmax = np.nanpercentile(np.abs(data), 95) if np.isfinite(data).any() else 1.0
        vmin = -vmax
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
        cblabel = "Avg PnL ($)"
        fmt = "{:.0f}"
    elif mode == "mae":
        vmin = np.nanpercentile(data, 5) if np.isfinite(data).any() else -1.0
        vmax = 0.0
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
        cblabel = "Avg MAE ($) (0 best)"
        fmt = "{:.0f}"
    elif mode == "winrate":
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
        cblabel = "Win Rate (0-1)"
        fmt = "{:.0%}"
    else:
        raise ValueError("mode must be pnl or mae or winrate")

    ax.set_title(title)
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Day of Week")
    ax.set_xticks(range(24))
    ax.set_xticklabels([str(i) for i in range(24)], fontsize=8)
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], fontsize=9)

    for r in range(7):
        for c in range(24):
            v = data[r, c]
            if np.isfinite(v):
                ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=7, alpha=0.60)

    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(cblabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def main():
    if not CANDLES_CACHE_CSV.exists():
        raise RuntimeError(f"Missing candles cache: {CANDLES_CACHE_CSV}. Run the grid script once (it downloads).")

    df = pd.read_csv(CANDLES_CACHE_CSV, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()

    eng_entry = compute_ce_engine(df, USE_HEIKIN_ASHI, CE_ATR_PERIOD, CE_ATR_MULT)

    sc = Scenario(
        stop_mode=STOP_MODE,
        atr_period=int(ATR_PERIOD),
        atr_init_mult=float(ATR_INIT_MULT),
        tp_r=float(TP_R) if TP_R is not None else None,
        trail_start_r=float(TRAIL_START_R),
        atr_trail_mult=float(ATR_TRAIL_MULT),
        chan_lookback=int(CHAN_LOOKBACK),
        max_hold_bars=MAX_HOLD_BARS,
        dynamic_risk=DYNAMIC_RISK,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_CE_XAU_M5_SINGLE"
    run_dir = OUT_ROOT / "single_runs" / run_id
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    trades_df, eq_df, skipped_min_sl = run_one(df, eng_entry, sc)

    summary = compute_summary(trades_df, eq_df, skipped_min_sl=skipped_min_sl)
    meta = {
        "scenario": asdict(sc),
        "summary": summary,
        "guards": {
            "MIN_SL_DIST_USD": MIN_SL_DIST_USD,
            "MAX_LOTS_CAP": MAX_LOTS_CAP,
            "MAX_NOTIONAL_USD": MAX_NOTIONAL_USD,
            "DISABLE_TP": DISABLE_TP,
            "N_EXIT_TP": int((trades_df["exit_reason"] == "TP").sum()),
        },
        "costs": {
            "USE_SPREAD_COST": USE_SPREAD_COST,
            "SPREAD_USD": SPREAD_USD,
            "USE_SLIPPAGE": USE_SLIPPAGE,
            "SLIPPAGE_ENTRY_ATR_MULT": SLIPPAGE_ENTRY_ATR_MULT,
            "SLIPPAGE_EXIT_ATR_MULT": SLIPPAGE_EXIT_ATR_MULT,
            "USE_MTM_WORST_DD": USE_MTM_WORST_DD,
        },
        "entry_engine": {"atr_period": CE_ATR_PERIOD, "atr_mult": CE_ATR_MULT, "use_heikin_ashi": USE_HEIKIN_ASHI},
        "trading_windows_utc": TRADING_WINDOWS_UTC,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    (run_dir / "reports" / "summary.json").write_text(json.dumps(meta, indent=2))
    trades_df.to_csv(run_dir / "reports" / "trades.csv", index=False)
    eq_df.to_csv(run_dir / "reports" / "equity_curve.csv", index=False)

    title = (
        f"CE XAU M5 | {sc.stop_mode} | atr_p={sc.atr_period} "
        f"| init={sc.atr_init_mult} | tpR={sc.tp_r} "
        f"| trail={sc.trail_start_r}R/{sc.atr_trail_mult}ATR "
        f"| chan_lb={sc.chan_lookback} | dyn={sc.dynamic_risk}"
    )
    plot_equity_dd(eq_df, run_dir / "plots" / "equity", title)

    if not trades_df.empty:
        pnl_pivot = _heatmap_by_dow_hour(trades_df["pnl_cash"], trades_df["entry_time"], agg="mean")
        plot_heatmap(pnl_pivot, run_dir / "plots" / "heatmap_pnl_by_dow_hour.png",
                     "Avg PnL by Day/Hour (Entry Time UTC)", mode="pnl")

        mae_pivot = _heatmap_by_dow_hour(trades_df["mae_cash"], trades_df["entry_time"], agg="mean")
        plot_heatmap(mae_pivot, run_dir / "plots" / "heatmap_mae_by_dow_hour.png",
                     "Avg MAE (Max DD in Cash) by Day/Hour (Entry Time UTC)", mode="mae")

        win_flag = (trades_df["pnl_cash"] > 0).astype(float)
        wr_pivot = _heatmap_by_dow_hour(win_flag, trades_df["entry_time"], agg="mean")
        plot_heatmap(wr_pivot, run_dir / "plots" / "heatmap_winrate_by_dow_hour.png",
                     "Win Rate by Day/Hour (Entry Time UTC)", mode="winrate")

    print(f"[DONE] {run_dir}")
    print("[FILES] reports/summary.json, reports/trades.csv, reports/equity_curve.csv, plots/*.png")


if __name__ == "__main__":
    main()
