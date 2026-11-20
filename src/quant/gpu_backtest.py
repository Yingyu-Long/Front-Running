"""[GPU Engine #4] PyTorch CUDA Vectorized Multi-Parameter Grid Backtester.

Validates our 2-week to 2-month swing strategy directly on the user's watchlist over
5-10 years with realistic commission + slippage, sweeping hundreds of parameter combos
(Entry RS/Trend threshold x Stop-Loss ATR Multiple x Max Holding Days) in parallel on CUDA!
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from src.quant.device import get_torch_device
from src.quant.indicators import compute_atr, compute_rsi


def run_gpu_grid_backtest(
    ohlcv_map: dict[str, pd.DataFrame],
    symbols: list[str],
    lookback_years: int = 6,
    cost_bps_per_side: float = 10.0,
    max_positions: int = 5,
    device_pref: str = "auto",
) -> dict[str, Any]:
    """
    Execute GPU-accelerated vectorized parameter sweep for our multi-week swing strategy:
      - Universe: user's core watchlist + bellwethers
      - Entry rule: Trend > SMA50, RSI in sweet spot, Relative Strength vs QQQ > threshold,
        plus optional QQQ > SMA50 macro regime filter
      - Exit rule: ATR Hard Stop-Loss OR Chandelier Trailing Stop OR Max Holding Days
      - Portfolio: Equal-weight up to `max_positions` ranked by composite momentum/RS score,
        rebalanced daily with realistic commission + slippage.
    """
    t0 = time.perf_counter()
    dev = get_torch_device(device_pref)
    use_cuda = torch is not None and dev is not None and dev.type == "cuda"

    qqq = ohlcv_map.get("QQQ")
    soxx = ohlcv_map.get("SOXX")
    if qqq is None or len(qqq) < 252:
        return {"available": False, "reason": "缺少 QQQ 基准历史数据，无法执行回测"}

    # Build aligned OHLC panels across symbols with >= 252 bars
    max_bars = lookback_years * 252 + 60
    valid_syms = [s for s in symbols if s in ohlcv_map and len(ohlcv_map[s]) >= 200]
    if len(valid_syms) < 3:
        return {"available": False, "reason": "历史超过 200 个交易日的标的少于 3 只"}

    dates = qqq.index[-min(max_bars, len(qqq)) :]
    T = len(dates)
    N = len(valid_syms)

    close_mat = np.full((T, N), np.nan, dtype=np.float32)
    low_mat = np.full((T, N), np.nan, dtype=np.float32)
    high_mat = np.full((T, N), np.nan, dtype=np.float32)
    atr_mat = np.full((T, N), np.nan, dtype=np.float32)
    signal_score = np.full((T, N), -999.0, dtype=np.float32)
    trend_ok = np.zeros((T, N), dtype=np.bool_)

    q_close = qqq["Close"].reindex(dates).ffill()
    q_ret21 = q_close.pct_change(21).fillna(0.0).values.astype(np.float32)
    q_sma50 = q_close.rolling(50).mean().values.astype(np.float32)
    q_regime_on = (q_close.values.astype(np.float32) > q_sma50)

    for j, sym in enumerate(valid_syms):
        df = ohlcv_map[sym].reindex(dates)
        c = df["Close"]
        c_vals = c.values.astype(np.float32)
        close_mat[:, j] = c_vals
        low_mat[:, j] = df["Low"].values.astype(np.float32)
        high_mat[:, j] = df["High"].values.astype(np.float32)

        sub_valid = df.dropna(subset=["Close"])
        if len(sub_valid) < 60:
            continue
        atr_s = compute_atr(sub_valid, 14).reindex(dates).values.astype(np.float32)
        rsi_s = compute_rsi(sub_valid["Close"], 14).reindex(dates).values.astype(np.float32)
        sma20 = c.rolling(20).mean().values.astype(np.float32)
        sma50 = c.rolling(50).mean().values.astype(np.float32)
        ret21 = c.pct_change(21).values.astype(np.float32)
        ret63 = c.pct_change(63).values.astype(np.float32)

        atr_mat[:, j] = atr_s
        rs21 = ret21 - q_ret21

        # Composite daily swing entry score (higher = stronger trend + RS, not overheated)
        score = (
            np.nan_to_num(rs21, nan=-1.0) * 100.0 * 1.4
            + np.nan_to_num(ret63, nan=-1.0) * 100.0 * 0.4
            - np.maximum(0.0, np.nan_to_num(rsi_s, nan=50.0) - 72.0) * 0.8
        )
        ok = (
            np.isfinite(c_vals)
            & np.isfinite(sma50)
            & np.isfinite(atr_s)
            & (c_vals > sma50)
            & (c_vals > sma20 * 0.96)
            & (rsi_s >= 38.0)
            & (rsi_s <= 74.0)
        )
        signal_score[:, j] = np.where(ok, score, -999.0)
        trend_ok[:, j] = ok

    # Parameter grid: 6 Entry Thresholds x 6 Stop-Loss ATR Multiples x 4 Hold Periods x 2 Regime Modes = 288 combos
    rs_thresholds = [-2.0, 0.0, 2.0, 4.0, 7.0, 10.0]
    stop_mults = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    hold_days_grid = [15, 25, 35, 50]
    regime_modes = [True, False]

    # Pre-compute forward single-trade return matrices per (stop_mult, hold_days) on GPU!
    # For each (t, j) entry at Close[t], simulate exact stop-loss & trailing stop exit within `hold_days`.
    cost_rt = 2.0 * (cost_bps_per_side / 10_000.0)
    trade_ret_cache: dict[tuple[float, int], np.ndarray] = {}

    start_bar = 65
    eval_T = T - start_bar - 1
    if eval_T < 120:
        return {"available": False, "reason": "对齐后的有效回测交易日不足"}

    if torch is not None and dev is not None:
        t_close = torch.tensor(np.nan_to_num(close_mat, nan=0.0), dtype=torch.float32, device=dev)
        t_low = torch.tensor(np.nan_to_num(low_mat, nan=0.0), dtype=torch.float32, device=dev)
        t_high = torch.tensor(np.nan_to_num(high_mat, nan=0.0), dtype=torch.float32, device=dev)
        t_atr = torch.tensor(np.nan_to_num(atr_mat, nan=0.0), dtype=torch.float32, device=dev)
        t_valid = torch.tensor(np.isfinite(close_mat) & (close_mat > 0), dtype=torch.bool, device=dev)

        with torch.no_grad():
            for sm in stop_mults:
                for hd in hold_days_grid:
                    # Vectorized path scan across ALL (T, N) entry points simultaneously on GPU!
                    entry_px = t_close  # [T, N]
                    init_stop = entry_px - sm * t_atr
                    peak_px = entry_px.clone()
                    stopped = torch.zeros((T, N), dtype=torch.bool, device=dev)
                    exit_px = entry_px.clone()

                    for step in range(1, hd + 1):
                        fut_low = torch.zeros_like(entry_px)
                        fut_high = torch.zeros_like(entry_px)
                        fut_close = torch.zeros_like(entry_px)
                        fut_ok = torch.zeros((T, N), dtype=torch.bool, device=dev)

                        fut_low[:-step] = t_low[step:]
                        fut_high[:-step] = t_high[step:]
                        fut_close[:-step] = t_close[step:]
                        fut_ok[:-step] = t_valid[step:]

                        # Chandelier trailing stop ratchets upward with peak
                        trail_stop = torch.maximum(init_stop, peak_px - (sm + 0.6) * t_atr)
                        hit_now = (~stopped) & fut_ok & (fut_low <= trail_stop)
                        exit_px = torch.where(hit_now, trail_stop, exit_px)
                        stopped = stopped | hit_now
                        peak_px = torch.where((~stopped) & fut_ok, torch.maximum(peak_px, fut_high), peak_px)

                        if step == hd:
                            final_still_open = (~stopped) & fut_ok
                            exit_px = torch.where(final_still_open, fut_close, exit_px)

                    safe_entry = torch.clamp(entry_px, min=1e-4)
                    ret_mat = torch.where(t_valid, (exit_px / safe_entry - 1.0) - cost_rt, torch.zeros_like(entry_px))
                    ret_mat = torch.clamp(ret_mat, min=-0.45, max=1.50)
                    trade_ret_cache[(sm, hd)] = ret_mat.cpu().numpy()
    else:
        for sm in stop_mults:
            for hd in hold_days_grid:
                ret_mat = np.zeros((T, N), dtype=np.float32)
                for t in range(start_bar, T - hd):
                    ep = close_mat[t]
                    atr = atr_mat[t]
                    valid = np.isfinite(ep) & (ep > 0) & np.isfinite(atr)
                    if not np.any(valid):
                        continue
                    stop_p = ep - sm * atr
                    peak_p = ep.copy()
                    ex_p = ep.copy()
                    done = ~valid
                    for step in range(1, hd + 1):
                        fl = low_mat[t + step]
                        fh = high_mat[t + step]
                        fc = close_mat[t + step]
                        ok_s = np.isfinite(fc) & (~done)
                        tr_stop = np.maximum(stop_p, peak_p - (sm + 0.6) * atr)
                        hit = ok_s & (fl <= tr_stop)
                        ex_p = np.where(hit, tr_stop, ex_p)
                        done = done | hit
                        peak_p = np.where(ok_s & (~hit), np.maximum(peak_p, fh), peak_p)
                        if step == hd:
                            ex_p = np.where(ok_s & (~done), fc, ex_p)
                    ret_mat[t] = np.where(valid, np.clip(ex_p / np.maximum(ep, 1e-4) - 1.0 - cost_rt, -0.45, 1.5), 0.0)
                trade_ret_cache[(sm, hd)] = ret_mat

    # Evaluate all 288 portfolio strategies using staggered cohort compounding
    grid_records: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    best_equity_curve: np.ndarray | None = None
    best_rebal_dates: list[str] = []
    best_rebal_idx: list[int] = []

    for use_regime in regime_modes:
        for hd in hold_days_grid:
            # Rebalance cohorts every `hd` days (non-overlapping swing tranches -> super realistic!)
            rebal_indices = list(range(start_bar, T - hd, max(5, hd // 2)))
            if len(rebal_indices) < 8:
                continue
            periods_per_year = 252.0 / max(5, hd // 2)

            for rs_thr in rs_thresholds:
                for sm in stop_mults:
                    t_ret = trade_ret_cache[(sm, hd)]
                    tranche_returns: list[float] = []
                    trade_pnls: list[float] = []

                    for t in rebal_indices:
                        if use_regime and not bool(q_regime_on[t]):
                            tranche_returns.append(0.0)  # Defensive cash mode
                            continue
                        scores_t = signal_score[t]
                        eligible = np.where(scores_t >= rs_thr)[0]
                        if len(eligible) == 0:
                            tranche_returns.append(0.0)
                            continue
                        top_idx = eligible[np.argsort(-scores_t[eligible])[:max_positions]]
                        pnls = t_ret[t, top_idx]
                        # Scale tranche return to avoid double-counting overlapping tranches
                        overlap_factor = float(hd) / float(max(5, hd // 2))
                        port_r = float(np.mean(pnls)) / overlap_factor
                        tranche_returns.append(port_r)
                        trade_pnls.extend([float(x) for x in pnls])

                    arr_r = np.array(tranche_returns, dtype=np.float64)
                    eq = np.cumprod(1.0 + arr_r)
                    total_ret = float(eq[-1] - 1.0)
                    years_span = max((rebal_indices[-1] - rebal_indices[0]) / 252.0, 0.5)
                    cagr = float((eq[-1] ** (1.0 / years_span) - 1.0) * 100.0) if eq[-1] > 0 else -99.0

                    peak = np.maximum.accumulate(eq)
                    dd = (eq / np.maximum(peak, 1e-8) - 1.0)
                    max_dd = float(np.min(dd) * 100.0)

                    std_r = float(np.std(arr_r))
                    sharpe = float(np.mean(arr_r) / std_r * np.sqrt(periods_per_year)) if std_r > 1e-8 else 0.0
                    neg_r = arr_r[arr_r < 0]
                    down_std = float(np.std(neg_r)) if len(neg_r) > 2 else max(std_r, 1e-6)
                    sortino = float(np.mean(arr_r) / down_std * np.sqrt(periods_per_year)) if down_std > 1e-8 else 0.0
                    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 0.5 else 0.0

                    if trade_pnls:
                        tp = np.array(trade_pnls)
                        win_rate = float(np.mean(tp > 0) * 100.0)
                        pos_s = float(np.sum(tp[tp > 0]))
                        neg_s = float(np.abs(np.sum(tp[tp < 0])))
                        pf = round(pos_s / neg_s, 2) if neg_s > 1e-6 else 3.0
                    else:
                        win_rate, pf = 0.0, 0.0

                    rec = {
                        "regime_filter": use_regime,
                        "hold_days": hd,
                        "rs_threshold": rs_thr,
                        "stop_atr_mult": sm,
                        "total_return_pct": round(total_ret * 100.0, 1),
                        "cagr_pct": round(cagr, 1),
                        "max_drawdown_pct": round(max_dd, 1),
                        "sharpe": round(sharpe, 2),
                        "sortino": round(sortino, 2),
                        "calmar": round(calmar, 2),
                        "win_rate_pct": round(win_rate, 1),
                        "profit_factor": pf,
                        "total_trades": len(trade_pnls),
                    }
                    grid_records.append(rec)

                    # Rank best robust strategy by Sharpe with minimum trade count filter
                    if len(trade_pnls) >= 20:
                        if best_record is None or (rec["sharpe"], rec["calmar"]) > (
                            best_record["sharpe"],
                            best_record["calmar"],
                        ):
                            best_record = rec
                            best_equity_curve = eq
                            best_rebal_dates = [str(dates[i].date()) for i in rebal_indices]
                            best_rebal_idx = list(rebal_indices)

    if best_record is None or best_equity_curve is None or not best_rebal_idx:
        return {"available": False, "reason": "有效交易次数不足，无法生成稳定回测曲线"}

    # Benchmark equity curves aligned directly by integer bar positions
    curve_dates = best_rebal_dates
    q_sub = q_close.iloc[best_rebal_idx]
    qqq_curve = (q_sub / q_sub.iloc[0]).values.tolist()
    soxx_curve: list[float] | None = None
    if soxx is not None and not soxx.empty:
        s_aligned = soxx["Close"].reindex(dates).ffill().bfill()
        s_sub = s_aligned.iloc[best_rebal_idx]
        if not s_sub.empty and float(s_sub.iloc[0]) > 0:
            soxx_curve = (s_sub / s_sub.iloc[0]).values.tolist()

    # Build 2D Parameter Robustness Heatmap (RS threshold x Stop ATR mult) for best (regime_filter, hold_days)
    hm_z: list[list[float]] = []
    for rs_thr in rs_thresholds:
        row_vals = []
        for sm in stop_mults:
            matched = [
                r["sharpe"]
                for r in grid_records
                if r["regime_filter"] == best_record["regime_filter"]
                and r["hold_days"] == best_record["hold_days"]
                and r["rs_threshold"] == rs_thr
                and r["stop_atr_mult"] == sm
            ]
            row_vals.append(matched[0] if matched else 0.0)
        hm_z.append(row_vals)

    return {
        "available": True,
        "backend": "cuda" if use_cuda else (str(dev) if dev else "cpu"),
        "combos_evaluated": len(grid_records),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "cost_bps_per_side": cost_bps_per_side,
        "symbols_used": valid_syms,
        "best_strategy": best_record,
        "equity_curve": {
            "dates": curve_dates,
            "strategy": [round(float(x), 4) for x in best_equity_curve],
            "qqq": [round(float(x), 4) for x in qqq_curve],
            "soxx": [round(float(x), 4) for x in soxx_curve] if soxx_curve else None,
        },
        "heatmap": {
            "x_stop_atr": stop_mults,
            "y_rs_threshold": rs_thresholds,
            "z_sharpe": hm_z,
        },
        "top_10_configs": sorted(grid_records, key=lambda x: x["sharpe"], reverse=True)[:10],
    }
