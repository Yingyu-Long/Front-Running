"""RTX 5070 Ti 4-Stage GPU Grid Trading Lab (PyTorch CUDA + SciPy + XGBoost/LightGBM).

Stage 1: Watchlist Grid Suitability Screener (PyTorch CUDA Hurst Exponent, Intraday Range/Liquidity,
         Overnight Gap Risk & 1% Baseline Grid Round-Trip Frequency -> Honest Objective Scoring).
Stage 2: 4D Hyper-Cube Tensor Grid Search (up to 60,000 combos across Arithmetic / Geometric / ATR-Dynamic
         x Spacing x Range Bounds x Martingale/Anti-Martingale Sizing x Stop-Loss) + SciPy 4D Gaussian
         Plateau Filter to eliminate overfit islands.
Stage 3: 10,000-Path Intraday Black-Swan Monte Carlo Stress Test on CUDA (Top 5 plateau configs x
         10,000 paths = 50,000 simulations covering Flash Crash, Slow Bleed, V-Reversal & Jump-Diffusion)
         with strict Grid Break Rate & 95th-percentile Floating Drawdown pass/fail gates.
Stage 4: Walk-Forward XGBoost (CUDA) + LightGBM 5-Day Crash Switch + Live Broker Grid Order Ladder.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover
    gaussian_filter = None  # type: ignore[assignment]

try:
    import xgboost as xgb
except Exception:  # pragma: no cover (ImportError or macOS missing libomp OSError)
    xgb = None  # type: ignore[assignment]

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover (ImportError or macOS missing libomp OSError)
    lgb = None  # type: ignore[assignment]

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    HistGradientBoostingClassifier = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]

from src.quant.device import get_torch_device, release_gpu_memory
from src.quant.indicators import compute_adx, compute_atr, compute_cmf, compute_rsi

logger = logging.getLogger(__name__)


# ==============================================================================
# STAGE 1: WATCHLIST GRID SUITABILITY SCREENER (PyTorch CUDA)
# ==============================================================================
def compute_hurst_exponent_gpu(
    close_arr: np.ndarray,
    dev: Any = None,
) -> tuple[float, float]:
    """
    Compute Generalized Hurst Exponent H(q=2) across multi-scale lags on PyTorch CUDA
    plus Ornstein-Uhlenbeck half-life (trading days).

      H < 0.50: Mean-reverting / oscillating (favorable for grid)
      H ~ 0.50: Geometric Brownian Motion (borderline)
      H > 0.50: Trending / persistent (dangerous for grid)
    """
    arr = np.asarray(close_arr, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    n = len(arr)
    if n < 60:
        return 0.55, 999.0

    log_p = np.log(arr)
    max_lag = min(64, max(12, n // 5))
    lags = [l for l in (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64) if l <= max_lag]
    if len(lags) < 4:
        return 0.55, 999.0

    if torch is not None and dev is not None and dev.type == "cuda":
        t_lp = torch.tensor(log_p, dtype=torch.float64, device=dev)
        variances = []
        for lag in lags:
            diff = t_lp[lag:] - t_lp[:-lag]
            variances.append(float(torch.mean(diff * diff).item()))
    else:
        variances = [float(np.mean((log_p[lag:] - log_p[:-lag]) ** 2)) for lag in lags]

    log_lags = np.log(np.array(lags, dtype=np.float64))
    log_vars = np.log(np.maximum(np.array(variances, dtype=np.float64), 1e-12))
    slope, _ = np.polyfit(log_lags, log_vars, 1)
    hurst = float(np.clip(slope / 2.0, 0.10, 0.95))

    # Estimate Ornstein-Uhlenbeck mean-reversion half-life on last 252 bars
    sub = log_p[-min(252, n) :]
    y = sub[1:] - sub[:-1]
    x = sub[:-1] - np.mean(sub[:-1])
    denom = float(np.dot(x, x))
    beta = float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0
    if beta < -1e-4:
        half_life = float(np.clip(-np.log(2.0) / beta, 1.0, 500.0))
    else:
        half_life = 999.0

    return round(hurst, 3), round(half_life, 1)


def _simulate_baseline_1pct_grid_batch(
    ohlcv_map: dict[str, pd.DataFrame],
    symbols: list[str],
    spacing_pct: float = 0.01,
    band_pct: float = 0.25,
    lookback_bars: int = 504,
    dev: Any = None,
) -> dict[str, dict[str, float]]:
    """
    Parallel PyTorch CUDA baseline 1% geometric grid sweep across ALL watchlist symbols
    simultaneously to measure objective annualized round-trip frequency & in-range ratio.
    """
    valid_syms = [s for s in symbols if s in ohlcv_map and len(ohlcv_map[s]) >= 40]
    if not valid_syms:
        return {}

    max_len = min(lookback_bars, max(len(ohlcv_map[s]) for s in valid_syms))
    S = len(valid_syms)

    low_mat = np.full((max_len, S), np.nan, dtype=np.float32)
    high_mat = np.full((max_len, S), np.nan, dtype=np.float32)
    close_mat = np.full((max_len, S), np.nan, dtype=np.float32)
    bar_counts = np.zeros(S, dtype=np.int32)

    for j, sym in enumerate(valid_syms):
        df = ohlcv_map[sym].tail(max_len)
        m = len(df)
        bar_counts[j] = m
        low_mat[:m, j] = df["Low"].values.astype(np.float32)
        high_mat[:m, j] = df["High"].values.astype(np.float32)
        close_mat[:m, j] = df["Close"].values.astype(np.float32)

    # Anchor each stock's baseline grid at its rolling 60-day median (or initial price)
    ref_px = np.array([close_mat[0, j] for j in range(S)], dtype=np.float32)
    log_step = float(np.log(1.0 + spacing_pct))

    if torch is not None and dev is not None:
        with torch.no_grad():
            t_low = torch.tensor(np.nan_to_num(low_mat, nan=0.0), device=dev, dtype=torch.float32)
            t_high = torch.tensor(np.nan_to_num(high_mat, nan=0.0), device=dev, dtype=torch.float32)
            t_close = torch.tensor(np.nan_to_num(close_mat, nan=0.0), device=dev, dtype=torch.float32)
            t_ok = torch.tensor(np.isfinite(close_mat) & (close_mat > 0), device=dev, dtype=torch.bool)

            t_ref = torch.tensor(ref_px, device=dev, dtype=torch.float32)
            lower_b = t_ref * (1.0 - band_pct)
            upper_b = t_ref * (1.0 + band_pct)
            max_rungs = int(band_pct / spacing_pct)

            last_idx = torch.zeros(S, device=dev, dtype=torch.float32)
            buys = torch.zeros(S, device=dev, dtype=torch.float32)
            sells = torch.zeros(S, device=dev, dtype=torch.float32)
            in_range_bars = torch.zeros(S, device=dev, dtype=torch.float32)

            for i in range(max_len):
                ok = t_ok[i]
                lo = torch.clamp(t_low[i], min=1e-4)
                hi = torch.clamp(t_high[i], min=1e-4)
                cl = torch.clamp(t_close[i], min=1e-4)

                inside = ok & (cl >= lower_b) & (cl <= upper_b)
                in_range_bars = in_range_bars + inside.float()

                # Continuous log-grid index traversed intraday (clamped to grid range)
                idx_lo = torch.clamp(torch.floor(torch.log(lo / t_ref) / log_step), -max_rungs, max_rungs)
                idx_hi = torch.clamp(torch.floor(torch.log(hi / t_ref) / log_step), -max_rungs, max_rungs)
                idx_cl = torch.clamp(torch.floor(torch.log(cl / t_ref) / log_step), -max_rungs, max_rungs)

                # Dip below last_idx triggers buys; rally above triggers sells
                b_cnt = torch.clamp(last_idx - idx_lo, min=0.0)
                s_cnt = torch.clamp(idx_hi - torch.minimum(last_idx, idx_lo), min=0.0)
                buys = torch.where(ok, buys + b_cnt, buys)
                sells = torch.where(ok, sells + s_cnt, sells)
                last_idx = torch.where(ok, idx_cl, last_idx)

            round_trips = torch.minimum(buys, sells).cpu().numpy()
            in_range_np = in_range_bars.cpu().numpy()
    else:
        round_trips = np.zeros(S, dtype=np.float32)
        in_range_np = np.zeros(S, dtype=np.float32)
        max_rungs = int(band_pct / spacing_pct)
        for j in range(S):
            m = int(bar_counts[j])
            r0 = float(ref_px[j])
            lb, ub = r0 * (1.0 - band_pct), r0 * (1.0 + band_pct)
            last_i, b_tot, s_tot, in_r = 0.0, 0.0, 0.0, 0.0
            for i in range(m):
                lo, hi, cl = float(low_mat[i, j]), float(high_mat[i, j]), float(close_mat[i, j])
                if lb <= cl <= ub:
                    in_r += 1.0
                i_lo = float(np.clip(np.floor(np.log(max(lo, 1e-4) / r0) / log_step), -max_rungs, max_rungs))
                i_hi = float(np.clip(np.floor(np.log(max(hi, 1e-4) / r0) / log_step), -max_rungs, max_rungs))
                i_cl = float(np.clip(np.floor(np.log(max(cl, 1e-4) / r0) / log_step), -max_rungs, max_rungs))
                b_tot += max(0.0, last_i - i_lo)
                s_tot += max(0.0, i_hi - min(last_i, i_lo))
                last_i = i_cl
            round_trips[j] = min(b_tot, s_tot)
            in_range_np[j] = in_r

    out: dict[str, dict[str, float]] = {}
    for j, sym in enumerate(valid_syms):
        yrs = max(float(bar_counts[j]) / 252.0, 0.15)
        ann_rt = float(round_trips[j]) / yrs
        in_range_pct = float(in_range_np[j]) / max(float(bar_counts[j]), 1.0) * 100.0
        # Net per round-trip after ~10bps total friction on 1% grid ~ 0.90% per grid unit
        est_grid_yield = ann_rt * (spacing_pct - 0.001) * 100.0 / max(2.0 * band_pct / spacing_pct, 10.0)
        out[sym] = {
            "annual_round_trips_1pct": round(ann_rt, 1),
            "in_range_pct": round(in_range_pct, 1),
            "baseline_grid_yield_pct": round(est_grid_yield, 1),
        }
    return out


def screen_watchlist_for_grid(
    ohlcv_map: dict[str, pd.DataFrame],
    symbols: list[str],
    universe_cfg: dict[str, Any] | None = None,
    device_pref: str = "auto",
) -> dict[str, Any]:
    """
    Stage 1: Objective GPU-accelerated Watchlist Grid Suitability Screener.
    Can honestly conclude that NO watchlist stock is currently ideal for grid trading!
    """
    t0 = time.perf_counter()
    dev = get_torch_device(device_pref)
    use_cuda = torch is not None and dev is not None and dev.type == "cuda"
    u_meta = (universe_cfg or {}).get("universe") or {}

    baseline_map = _simulate_baseline_1pct_grid_batch(
        ohlcv_map, symbols, spacing_pct=0.01, band_pct=0.25, lookback_bars=504, dev=dev
    )

    rows: list[dict[str, Any]] = []
    for sym in symbols:
        df = ohlcv_map.get(sym)
        if df is None or len(df) < 40:
            rows.append(
                {
                    "symbol": sym,
                    "name": str((u_meta.get(sym) or {}).get("name", sym)),
                    "history_bars": 0 if df is None else len(df),
                    "hurst": None,
                    "half_life_days": None,
                    "adx14": None,
                    "intraday_amp_pct": None,
                    "gap_p95_pct": None,
                    "dollar_vol_m": None,
                    "annual_round_trips_1pct": 0.0,
                    "in_range_pct": 0.0,
                    "grid_score": 0.0,
                    "verdict_code": "INSUFFICIENT_DATA",
                    "verdict_en": "⚪ INSUFFICIENT DATA",
                    "verdict_zh": "⚪ 历史数据不足",
                    "reason_en": "Fewer than 40 daily bars; cannot calibrate Hurst or grid range safely.",
                    "reason_zh": "历史交易日少于40天，无法计算赫斯特指数与安全网格区间。",
                }
            )
            continue

        tail_2y = df.tail(min(504, len(df)))
        close = tail_2y["Close"]
        hurst, half_life = compute_hurst_exponent_gpu(close.values, dev=dev)
        adx14 = float(compute_adx(tail_2y, 14).iloc[-1])

        # Intraday amplitude distribution & overnight gap risk
        amp = ((tail_2y["High"] - tail_2y["Low"]) / tail_2y["Close"].replace(0.0, np.nan)).dropna()
        med_amp_pct = float(amp.tail(min(126, len(amp))).median() * 100.0) if not amp.empty else 2.0

        prev_c = tail_2y["Close"].shift(1)
        gaps = ((tail_2y["Open"] / prev_c - 1.0).abs() * 100.0).dropna()
        gap_p95 = float(np.percentile(gaps, 95)) if len(gaps) >= 10 else 2.5

        # Liquidity ($M/day median dollar volume over last 60 days)
        dollar_vol = (tail_2y["Close"] * tail_2y["Volume"]).tail(min(60, len(tail_2y)))
        med_dv_m = float(dollar_vol.median() / 1e6) if not dollar_vol.empty else 0.0

        base = baseline_map.get(sym) or {}
        ann_rt = float(base.get("annual_round_trips_1pct", 0.0))
        in_range = float(base.get("in_range_pct", 0.0))

        # --- Strict, Honest 0-100 Objective Grid Suitability Score ---
        # 1. Hurst Exponent (38 pts max) — Most critical factor!
        if hurst <= 0.42:
            s_hurst = 38.0
        elif hurst <= 0.48:
            s_hurst = 38.0 - (hurst - 0.42) / 0.06 * 9.0
        elif hurst <= 0.53:
            s_hurst = 29.0 - (hurst - 0.48) / 0.05 * 14.0
        elif hurst <= 0.60:
            s_hurst = 15.0 - (hurst - 0.53) / 0.07 * 13.0
        else:
            s_hurst = 0.0

        # 2. Intraday Amplitude & Gap Safety (22 pts max)
        if 1.8 <= med_amp_pct <= 4.2:
            s_amp = 15.0
        elif 1.2 <= med_amp_pct < 1.8 or 4.2 < med_amp_pct <= 5.8:
            s_amp = 10.0
        else:
            s_amp = 4.0

        if gap_p95 <= 2.5:
            s_gap = 7.0
        elif gap_p95 <= 4.5:
            s_gap = 4.0
        else:
            s_gap = 0.0  # Frequent big overnight gaps jump right over grid orders!

        # 3. Liquidity & Slippage Quality (15 pts max)
        if med_dv_m >= 500.0:
            s_liq = 15.0
        elif med_dv_m >= 100.0:
            s_liq = 12.0
        elif med_dv_m >= 25.0:
            s_liq = 7.0
        else:
            s_liq = 1.0  # Illiquid / high slippage

        # 4. Baseline 1% Grid Frequency & In-Range Stability (25 pts max)
        s_freq = min(15.0, ann_rt / 160.0 * 15.0)
        s_range = min(10.0, in_range / 85.0 * 10.0)

        raw_score = s_hurst + s_amp + s_gap + s_liq + s_freq + s_range

        # Hard realism guardrails (so trending stocks NEVER get fake high scores!)
        reasons_en: list[str] = []
        reasons_zh: list[str] = []

        if hurst >= 0.56:
            raw_score = min(raw_score, 44.0)
            reasons_en.append(f"Strong trend (H={hurst:.2f} > 0.56) — grid risks breakout/bag-holding")
            reasons_zh.append(f"强趋势特征(H={hurst:.2f} > 0.56)，单边行情极易破网或卖飞")
        elif hurst < 0.48:
            reasons_en.append(f"Mean-reverting (H={hurst:.2f} < 0.48, ideal oscillation)")
            reasons_zh.append(f"均值回归特征明显(H={hurst:.2f} < 0.48)，天然适合震荡套利")
        else:
            reasons_en.append(f"Near random-walk (H={hurst:.2f}), requires ML Crash Switch")
            reasons_zh.append(f"接近随机游走(H={hurst:.2f})，需搭配 ML 防暴跌开关")

        if adx14 >= 32.0:
            raw_score -= 6.0
            reasons_en.append(f"High current trend ADX={adx14:.0f}")
            reasons_zh.append(f"当前趋势过强(ADX={adx14:.0f})")

        if gap_p95 >= 5.0:
            raw_score -= 6.0
            reasons_en.append(f"Heavy overnight gap risk (P95={gap_p95:.1f}%)")
            reasons_zh.append(f"隔夜跳空风险大(P95={gap_p95:.1f}%)")

        if med_dv_m < 25.0:
            raw_score = min(raw_score, 40.0)
            reasons_en.append("Low liquidity / slippage risk")
            reasons_zh.append("流动性偏低、滑点风险高")

        if in_range < 45.0:
            raw_score -= 8.0
            reasons_en.append(f"Frequent range breakout (in-range {in_range:.0f}%)")
            reasons_zh.append(f"经常突破±25%箱体(区间内占比仅{in_range:.0f}%)")

        final_score = round(max(0.0, min(98.0, raw_score)), 1)

        if final_score >= 72.0:
            v_code = "SUITABLE"
            v_en = "🟢 SUITABLE"
            v_zh = "🟢 适合跑网格"
        elif final_score >= 56.0:
            v_code = "CONDITIONAL"
            v_en = "🟡 CONDITIONAL (USE ML SWITCH)"
            v_zh = "🟡 谨慎可用(须开ML防暴跌)"
        else:
            v_code = "NOT_SUITABLE"
            v_en = "🔴 NOT SUITABLE"
            v_zh = "🔴 不适合跑网格"

        rows.append(
            {
                "symbol": sym,
                "name": str((u_meta.get(sym) or {}).get("name", sym)),
                "history_bars": int(len(df)),
                "hurst": hurst,
                "half_life_days": half_life,
                "adx14": round(adx14, 1),
                "intraday_amp_pct": round(med_amp_pct, 2),
                "gap_p95_pct": round(gap_p95, 2),
                "dollar_vol_m": round(med_dv_m, 1),
                "annual_round_trips_1pct": ann_rt,
                "in_range_pct": in_range,
                "baseline_grid_yield_pct": float(base.get("baseline_grid_yield_pct", 0.0)),
                "grid_score": final_score,
                "verdict_code": v_code,
                "verdict_en": v_en,
                "verdict_zh": v_zh,
                "reason_en": "; ".join(reasons_en),
                "reason_zh": "；".join(reasons_zh),
            }
        )

    rows.sort(key=lambda r: float(r.get("grid_score") or 0.0), reverse=True)
    suitable_cnt = sum(1 for r in rows if r["verdict_code"] == "SUITABLE")
    cond_cnt = sum(1 for r in rows if r["verdict_code"] == "CONDITIONAL")

    if suitable_cnt > 0:
        top_syms = [r["symbol"] for r in rows if r["verdict_code"] == "SUITABLE"][:3]
        summary_en = f"{suitable_cnt} stock(s) objectively qualify for grid trading ({', '.join(top_syms)}). Run Stage 2–4 below before deploying capital."
        summary_zh = f"自选池中有 {suitable_cnt} 只标的客观达到网格标准（{', '.join(top_syms)}），可在下方运行阶段二~四做参数高原与压力测试。"
    elif cond_cnt > 0:
        top_syms = [r["symbol"] for r in rows if r["verdict_code"] == "CONDITIONAL"][:3]
        summary_en = f"⚠️ ZERO ideal pure-grid candidates (your AI/Storage watchlist is heavily trend-driven). {cond_cnt} borderline ticker(s) ({', '.join(top_syms)}) are CONDITIONAL ONLY — must use ATR Dynamic Grid + ML Crash Switch."
        summary_zh = f"⚠️ 客观结论：你当前自选池以强趋势 AI/存储股为主，【没有】天生完美的纯网格标的！仅 {cond_cnt} 只（{', '.join(top_syms)}）勉强有条件可用，必须使用 ATR 动态网格 + ML 防单边暴跌开关。"
    else:
        summary_en = "🚫 HONEST VERDICT: NONE of your current watchlist stocks are suitable for grid trading right now (excessive trend persistence H > 0.55 or gap/breakout risk). Stick to Swing Trend Trading instead of forcing a grid!"
        summary_zh = "🚫 客观结论：你当前 Watchlist 中【没有任何一只股票】适合跑网格（普遍强趋势 H > 0.55 或跳空破网风险过高）。请坚持波段趋势交易，切勿硬跑网格送钱！"

    return {
        "backend": "cuda" if use_cuda else (str(dev) if dev else "cpu"),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "suitable_count": suitable_cnt,
        "conditional_count": cond_cnt,
        "summary_en": summary_en,
        "summary_zh": summary_zh,
        "candidates": rows,
    }


# ==============================================================================
# STAGE 2: 4D HYPER-CUBE GPU TENSOR GRID SEARCH + SCIPY GAUSSIAN PLATEAU FINDER
# ==============================================================================
def _simulate_grid_hypercube_batch_torch(
    *,
    low_arr: np.ndarray,
    high_arr: np.ndarray,
    close_arr: np.ndarray,
    atr_arr: np.ndarray,
    grid_type: str,            # "arithmetic" | "geometric" | "atr_dynamic"
    spacings: np.ndarray,      # [B]
    lower_mults: np.ndarray,   # [B] e.g. 0.75 (25% below anchor)
    upper_mults: np.ndarray,   # [B] e.g. 1.25 (25% above anchor)
    martingale_r: np.ndarray,  # [B] <1 anti-martingale, 1 equal, >1 martingale
    stop_dds: np.ndarray,      # [B] negative float e.g. -0.20 (or -0.99 = no stop)
    fee_rate: float,
    dev: Any,
    chunk_size: int = 4096,
) -> dict[str, np.ndarray]:
    """
    Vectorized PyTorch CUDA grid simulator evaluating B parameter combinations simultaneously
    over T historical OHLC bars (batched in chunks of 4,096 to keep VRAM < 150 MB!).
    """
    B = len(spacings)
    T = len(close_arr)
    anchor_px = float(close_arr[0])

    out_ret = np.zeros(B, dtype=np.float32)
    out_mdd = np.zeros(B, dtype=np.float32)
    out_sharpe = np.zeros(B, dtype=np.float32)
    out_util = np.zeros(B, dtype=np.float32)
    out_trades = np.zeros(B, dtype=np.float32)

    use_torch = torch is not None and dev is not None

    for start in range(0, B, chunk_size):
        end = min(B, start + chunk_size)
        bsz = end - start

        sp_c = spacings[start:end].astype(np.float32)
        lo_m = lower_mults[start:end].astype(np.float32)
        hi_m = upper_mults[start:end].astype(np.float32)
        mg_r = martingale_r[start:end].astype(np.float32)
        st_dd = stop_dds[start:end].astype(np.float32)

        if use_torch:
            with torch.no_grad():
                t_sp = torch.tensor(sp_c, device=dev, dtype=torch.float32)
                t_lb = torch.tensor(lo_m * anchor_px, device=dev, dtype=torch.float32)
                t_ub = torch.tensor(hi_m * anchor_px, device=dev, dtype=torch.float32)
                t_mg = torch.tensor(mg_r, device=dev, dtype=torch.float32)
                t_stop = torch.tensor(st_dd, device=dev, dtype=torch.float32)

                # Initial capital = 1.0 (normalized); start 30% deployed at anchor, 70% cash bullets
                cash = torch.full((bsz,), 0.70, device=dev, dtype=torch.float32)
                shares = torch.full((bsz,), 0.30 / max(anchor_px, 1e-4), device=dev, dtype=torch.float32)
                last_grid_px = torch.full((bsz,), anchor_px, device=dev, dtype=torch.float32)
                depth_lvl = torch.zeros((bsz,), device=dev, dtype=torch.float32)  # steps bought below anchor

                peak_eq = torch.ones((bsz,), device=dev, dtype=torch.float32)
                max_dd = torch.zeros((bsz,), device=dev, dtype=torch.float32)
                prev_eq = torch.ones((bsz,), device=dev, dtype=torch.float32)
                sum_r = torch.zeros((bsz,), device=dev, dtype=torch.float32)
                sum_r2 = torch.zeros((bsz,), device=dev, dtype=torch.float32)
                sum_util = torch.zeros((bsz,), device=dev, dtype=torch.float32)
                trades = torch.zeros((bsz,), device=dev, dtype=torch.float32)
                alive = torch.ones((bsz,), device=dev, dtype=torch.bool)

                # Estimate base number of rungs in lower half to normalize per-rung bullet size
                base_step_abs = (
                    t_sp * anchor_px
                    if grid_type == "arithmetic"
                    else (
                        t_sp * anchor_px
                        if grid_type == "geometric"
                        else t_sp * float(np.nanmedian(atr_arr))
                    )
                )
                n_rungs_est = torch.clamp((anchor_px - t_lb) / torch.clamp(base_step_abs, min=1e-4), min=3.0, max=60.0)
                base_order_frac = 0.70 / n_rungs_est

                for i in range(1, T):
                    lo_i = float(low_arr[i])
                    hi_i = float(high_arr[i])
                    cl_i = float(close_arr[i])
                    atr_i = float(atr_arr[i])

                    if grid_type == "arithmetic":
                        step_px = t_sp * anchor_px
                    elif grid_type == "geometric":
                        step_px = t_sp * last_grid_px
                    else:
                        step_px = torch.clamp(t_sp * atr_i, min=cl_i * 0.004)

                    # Trigger BUY if bar low dropped at least 1 grid step and stays above lower bound
                    buy_px = last_grid_px - step_px
                    can_buy = alive & (lo_i <= buy_px) & (buy_px >= t_lb) & (cash > 0.005)
                    # Number of rungs dipped (cap at 3 fills per daily bar for realistic execution)
                    n_buy = torch.clamp(torch.floor((last_grid_px - torch.maximum(torch.full_like(t_lb, lo_i), t_lb)) / torch.clamp(step_px, min=1e-4)), min=0.0, max=3.0)
                    can_buy = can_buy & (n_buy >= 1.0)

                    # Martingale / Anti-Martingale weight multiplier based on depth below anchor
                    mg_mult = torch.clamp(torch.pow(t_mg, torch.clamp(depth_lvl, min=0.0, max=8.0)), min=0.25, max=3.5)
                    buy_usd = torch.minimum(cash, base_order_frac * mg_mult * n_buy)
                    avg_buy_px = torch.clamp(last_grid_px - 0.5 * step_px * n_buy, min=max(lo_i, 1e-4))
                    add_sh = torch.where(can_buy, buy_usd * (1.0 - fee_rate) / avg_buy_px, torch.zeros_like(shares))

                    cash = torch.where(can_buy, cash - buy_usd, cash)
                    shares = shares + add_sh
                    depth_lvl = torch.where(can_buy, depth_lvl + n_buy, depth_lvl)
                    last_grid_px = torch.where(can_buy, last_grid_px - step_px * n_buy, last_grid_px)
                    trades = trades + torch.where(can_buy, n_buy, torch.zeros_like(trades))

                    # Trigger SELL if bar high rallied at least 1 grid step above last_grid_px
                    sell_px = last_grid_px + step_px
                    can_sell = alive & (hi_i >= sell_px) & (last_grid_px <= t_ub) & (shares > 1e-6)
                    n_sell = torch.clamp(torch.floor((torch.minimum(torch.full_like(t_ub, hi_i), t_ub) - last_grid_px) / torch.clamp(step_px, min=1e-4)), min=0.0, max=3.0)
                    can_sell = can_sell & (n_sell >= 1.0)

                    avg_sell_px = torch.clamp(last_grid_px + 0.5 * step_px * n_sell, max=max(hi_i, 1e-4))
                    sell_target_usd = base_order_frac * n_sell * 1.05
                    sell_sh = torch.minimum(shares, sell_target_usd / torch.clamp(avg_sell_px, min=1e-4))
                    proceeds = torch.where(can_sell, sell_sh * avg_sell_px * (1.0 - fee_rate), torch.zeros_like(cash))

                    shares = torch.where(can_sell, shares - sell_sh, shares)
                    cash = cash + proceeds
                    depth_lvl = torch.where(can_sell, torch.clamp(depth_lvl - n_sell, min=0.0), depth_lvl)
                    last_grid_px = torch.where(can_sell, last_grid_px + step_px * n_sell, last_grid_px)
                    trades = trades + torch.where(can_sell, n_sell, torch.zeros_like(trades))

                    # Mark-to-market equity & hard portfolio drawdown stop-loss
                    eq = cash + shares * cl_i
                    peak_eq = torch.maximum(peak_eq, eq)
                    dd = eq / torch.clamp(peak_eq, min=1e-6) - 1.0
                    max_dd = torch.minimum(max_dd, dd)

                    hit_stop = alive & (dd <= t_stop)
                    cash = torch.where(hit_stop, cash + shares * cl_i * (1.0 - fee_rate), cash)
                    shares = torch.where(hit_stop, torch.zeros_like(shares), shares)
                    alive = alive & (~hit_stop)
                    eq = cash + shares * cl_i

                    dr = eq / torch.clamp(prev_eq, min=1e-6) - 1.0
                    sum_r = sum_r + dr
                    sum_r2 = sum_r2 + dr * dr
                    sum_util = sum_util + (shares * cl_i) / torch.clamp(eq, min=1e-6)
                    prev_eq = eq

                n_steps = float(max(T - 1, 1))
                mean_r = sum_r / n_steps
                var_r = torch.clamp(sum_r2 / n_steps - mean_r * mean_r, min=1e-10)
                sharpe = mean_r / torch.sqrt(var_r) * float(np.sqrt(252.0))

                out_ret[start:end] = ((prev_eq - 1.0) * 100.0).cpu().numpy()
                out_mdd[start:end] = (max_dd * 100.0).cpu().numpy()
                out_sharpe[start:end] = torch.clamp(sharpe, -5.0, 6.0).cpu().numpy()
                out_util[start:end] = (sum_util / n_steps * 100.0).cpu().numpy()
                out_trades[start:end] = trades.cpu().numpy()
        else:
            # Pure NumPy vectorized fallback (same math!)
            lb = lo_m * anchor_px
            ub = hi_m * anchor_px
            cash = np.full(bsz, 0.70, dtype=np.float32)
            shares = np.full(bsz, 0.30 / max(anchor_px, 1e-4), dtype=np.float32)
            last_px = np.full(bsz, anchor_px, dtype=np.float32)
            depth = np.zeros(bsz, dtype=np.float32)
            peak_eq = np.ones(bsz, dtype=np.float32)
            max_dd = np.zeros(bsz, dtype=np.float32)
            prev_eq = np.ones(bsz, dtype=np.float32)
            sum_r = np.zeros(bsz, dtype=np.float32)
            sum_r2 = np.zeros(bsz, dtype=np.float32)
            sum_util = np.zeros(bsz, dtype=np.float32)
            trades = np.zeros(bsz, dtype=np.float32)
            alive = np.ones(bsz, dtype=bool)

            base_step = sp_c * (float(np.nanmedian(atr_arr)) if grid_type == "atr_dynamic" else anchor_px)
            n_rungs = np.clip((anchor_px - lb) / np.maximum(base_step, 1e-4), 3.0, 60.0)
            base_frac = 0.70 / n_rungs

            for i in range(1, T):
                lo_i, hi_i, cl_i, atr_i = float(low_arr[i]), float(high_arr[i]), float(close_arr[i]), float(atr_arr[i])
                if grid_type == "arithmetic":
                    step_px = sp_c * anchor_px
                elif grid_type == "geometric":
                    step_px = sp_c * last_px
                else:
                    step_px = np.maximum(sp_c * atr_i, cl_i * 0.004)

                n_buy = np.clip(np.floor((last_px - np.maximum(lo_i, lb)) / np.maximum(step_px, 1e-4)), 0.0, 3.0)
                can_buy = alive & (n_buy >= 1.0) & (cash > 0.005)
                mg_mult = np.clip(np.power(mg_r, np.clip(depth, 0.0, 8.0)), 0.25, 3.5)
                buy_usd = np.minimum(cash, base_frac * mg_mult * n_buy)
                avg_bp = np.maximum(last_px - 0.5 * step_px * n_buy, max(lo_i, 1e-4))
                cash = np.where(can_buy, cash - buy_usd, cash)
                shares = shares + np.where(can_buy, buy_usd * (1.0 - fee_rate) / avg_bp, 0.0)
                depth = np.where(can_buy, depth + n_buy, depth)
                last_px = np.where(can_buy, last_px - step_px * n_buy, last_px)
                trades += np.where(can_buy, n_buy, 0.0)

                n_sell = np.clip(np.floor((np.minimum(hi_i, ub) - last_px) / np.maximum(step_px, 1e-4)), 0.0, 3.0)
                can_sell = alive & (n_sell >= 1.0) & (shares > 1e-6)
                avg_sp = np.minimum(last_px + 0.5 * step_px * n_sell, max(hi_i, 1e-4))
                sell_sh = np.minimum(shares, (base_frac * n_sell * 1.05) / np.maximum(avg_sp, 1e-4))
                shares = np.where(can_sell, shares - sell_sh, shares)
                cash += np.where(can_sell, sell_sh * avg_sp * (1.0 - fee_rate), 0.0)
                depth = np.where(can_sell, np.maximum(depth - n_sell, 0.0), depth)
                last_px = np.where(can_sell, last_px + step_px * n_sell, last_px)
                trades += np.where(can_sell, n_sell, 0.0)

                eq = cash + shares * cl_i
                peak_eq = np.maximum(peak_eq, eq)
                dd = eq / np.maximum(peak_eq, 1e-6) - 1.0
                max_dd = np.minimum(max_dd, dd)
                hit_stop = alive & (dd <= st_dd)
                cash = np.where(hit_stop, cash + shares * cl_i * (1.0 - fee_rate), cash)
                shares = np.where(hit_stop, 0.0, shares)
                alive &= ~hit_stop
                eq = cash + shares * cl_i

                dr = eq / np.maximum(prev_eq, 1e-6) - 1.0
                sum_r += dr
                sum_r2 += dr * dr
                sum_util += (shares * cl_i) / np.maximum(eq, 1e-6)
                prev_eq = eq

            n_steps = float(max(T - 1, 1))
            mean_r = sum_r / n_steps
            var_r = np.maximum(sum_r2 / n_steps - mean_r * mean_r, 1e-10)
            out_ret[start:end] = (prev_eq - 1.0) * 100.0
            out_mdd[start:end] = max_dd * 100.0
            out_sharpe[start:end] = np.clip(mean_r / np.sqrt(var_r) * np.sqrt(252.0), -5.0, 6.0)
            out_util[start:end] = sum_util / n_steps * 100.0
            out_trades[start:end] = trades

    return {
        "total_return_pct": out_ret,
        "max_drawdown_pct": out_mdd,
        "sharpe": out_sharpe,
        "capital_util_pct": out_util,
        "trades": out_trades,
    }


def run_gpu_4d_grid_hypercube(
    symbol: str,
    df: pd.DataFrame,
    lookback_bars: int = 504,
    fee_bps: float = 5.0,
    high_density_gpu: bool = True,
    device_pref: str = "auto",
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Stage 2: 4D Hyper-Cube Parameter Optimization on RTX 5070 Ti CUDA + SciPy Gaussian Plateau Finder.

    Dimensions per Grid Type (Arithmetic / Geometric / ATR-Dynamic):
      - CUDA High Density: 20 Spacings x 20 Ranges x 10 Martingale Ratios x 5 Stop-Losses = 20,000 / type
        => 60,000 TOTAL parameter combos across all 3 grid types!
      - CPU Fallback:      10 Spacings x 10 Ranges x  6 Martingale Ratios x 4 Stop-Losses =  2,400 / type
        =>  7,200 TOTAL parameter combos (fast even on Mac CPU!).
    """
    t0 = time.perf_counter()
    dev = get_torch_device(device_pref)
    use_cuda = torch is not None and dev is not None and dev.type == "cuda"

    sub = df.tail(min(lookback_bars, len(df))).copy()
    if len(sub) < 60:
        return {"available": False, "reason": "历史数据不足 60 个交易日，无法构建 4D 参数超立方体"}

    low_arr = sub["Low"].values.astype(np.float32)
    high_arr = sub["High"].values.astype(np.float32)
    close_arr = sub["Close"].values.astype(np.float32)
    atr_arr = compute_atr(sub, 14).bfill().fillna(close_arr[0] * 0.025).values.astype(np.float32)
    anchor_px = float(close_arr[0])
    last_px = float(close_arr[-1])
    last_atr = float(atr_arr[-1])

    # Parameter axes tailored to RTX 5070 Ti compute vs CPU fallback
    if use_cuda and high_density_gpu:
        n_sp, n_rg, n_mg, n_st = 20, 20, 10, 5
    else:
        n_sp, n_rg, n_mg, n_st = 10, 10, 6, 4

    range_lower_pcts = np.linspace(0.10, 0.38, n_rg)   # -10% to -38% lower bound
    range_upper_pcts = np.linspace(0.10, 0.38, n_rg)   # +10% to +38% upper bound
    martingale_vals = np.linspace(0.75, 1.35, n_mg)    # <1 Anti-Martingale, 1 Equal, >1 Martingale
    stop_vals = np.array([-0.12, -0.18, -0.25, -0.32, -0.99][:n_st])  # -0.99 = No Hard Stop

    grid_configs = {
        "arithmetic": {
            "label_en": "Arithmetic (Fixed $)",
            "label_zh": "等差网格 (固定金额)",
            "spacings": np.linspace(0.004, 0.036, n_sp),  # % of anchor converted to fixed $
        },
        "geometric": {
            "label_en": "Geometric (Fixed %)",
            "label_zh": "等比网格 (固定百分比)",
            "spacings": np.linspace(0.005, 0.040, n_sp),
        },
        "atr_dynamic": {
            "label_en": "ATR Dynamic (Adaptive)",
            "label_zh": "ATR 动态自适应网格",
            "spacings": np.linspace(0.25, 1.75, n_sp),    # ATR multiples
        },
    }

    type_summaries: list[dict[str, Any]] = []
    all_plateaus: list[dict[str, Any]] = []
    heatmaps_by_type: dict[str, Any] = {}
    total_combos = 0

    for gtype, gmeta in grid_configs.items():
        if progress_cb:
            progress_cb(f"⚡ Stage 2 [{gtype.upper()}]: evaluating {n_sp * n_rg * n_mg * n_st:,} 4D combos on {'RTX 5070 Ti CUDA' if use_cuda else 'CPU'}...")

        sp_axis = gmeta["spacings"]
        # Build exact 4D meshgrid [n_sp, n_rg, n_mg, n_st]
        g_sp, g_rg_idx, g_mg, g_st = np.meshgrid(
            sp_axis, np.arange(n_rg), martingale_vals, stop_vals, indexing="ij"
        )
        g_lo_m = 1.0 - range_lower_pcts[g_rg_idx]
        g_hi_m = 1.0 + range_upper_pcts[g_rg_idx]

        res = _simulate_grid_hypercube_batch_torch(
            low_arr=low_arr,
            high_arr=high_arr,
            close_arr=close_arr,
            atr_arr=atr_arr,
            grid_type=gtype,
            spacings=g_sp.ravel(),
            lower_mults=g_lo_m.ravel(),
            upper_mults=g_hi_m.ravel(),
            martingale_r=g_mg.ravel(),
            stop_dds=g_st.ravel(),
            fee_rate=fee_bps / 10_000.0,
            dev=dev,
        )
        shape4 = (n_sp, n_rg, n_mg, n_st)
        total_combos += int(np.prod(shape4))

        sh4 = res["sharpe"].reshape(shape4)
        ret4 = res["total_return_pct"].reshape(shape4)
        mdd4 = res["max_drawdown_pct"].reshape(shape4)
        util4 = res["capital_util_pct"].reshape(shape4)
        trd4 = res["trades"].reshape(shape4)

        # Objective penalizing zero-activity configs and extreme drawdowns
        obj4 = sh4 + np.clip(ret4 / 60.0, -1.5, 1.5) * 0.35 - np.maximum(0.0, np.abs(mdd4) - 25.0) * 0.04
        obj4 = np.where(trd4 >= 12.0, obj4, obj4 - 1.5)

        # SciPy 4D Gaussian Filter -> Smooths away isolated overfit spikes, reveals broad "Parameter Plateaus"!
        if gaussian_filter is not None:
            smooth4 = gaussian_filter(obj4.astype(np.float64), sigma=(1.1, 1.1, 0.9, 0.7), mode="nearest")
            # Local stability penalty: subtract 0.25 * local neighborhood std
            sq_smooth = gaussian_filter((obj4.astype(np.float64)) ** 2, sigma=(1.1, 1.1, 0.9, 0.7), mode="nearest")
            local_std4 = np.sqrt(np.maximum(sq_smooth - smooth4 ** 2, 0.0))
            plateau_score4 = smooth4 - 0.25 * local_std4
        else:
            smooth4 = obj4.astype(np.float64)
            local_std4 = np.zeros_like(smooth4)
            plateau_score4 = smooth4

        # Pick top 3 distinct plateau champions for this grid type (Manhattan distance >= 3)
        flat_order = np.argsort(-plateau_score4.ravel())
        picked_coords: list[tuple[int, int, int, int]] = []
        for flat_i in flat_order:
            coord = np.unravel_index(int(flat_i), shape4)
            if all(sum(abs(a - b) for a, b in zip(coord, prev)) >= 3 for prev in picked_coords):
                picked_coords.append(tuple(int(c) for c in coord))
                if len(picked_coords) >= 3:
                    break

        best_c = picked_coords[0]
        # 2D slice at best (martingale, stop_loss) for Raw vs Gaussian Plateau visual comparison
        heatmaps_by_type[gtype] = {
            "x_range_pct": [round(float(x * 100.0), 1) for x in range_lower_pcts],
            "y_spacing": [round(float(x * 100.0 if gtype != "atr_dynamic" else x), 2) for x in sp_axis],
            "y_unit": "x ATR" if gtype == "atr_dynamic" else "%",
            "z_raw_sharpe": np.round(sh4[:, :, best_c[2], best_c[3]], 2).tolist(),
            "z_plateau_score": np.round(plateau_score4[:, :, best_c[2], best_c[3]], 2).tolist(),
            "best_slice_martingale": round(float(martingale_vals[best_c[2]]), 2),
            "best_slice_stop_pct": round(float(stop_vals[best_c[3]] * 100.0), 0),
        }

        type_summaries.append(
            {
                "grid_type": gtype,
                "label_en": gmeta["label_en"],
                "label_zh": gmeta["label_zh"],
                "combos": int(np.prod(shape4)),
                "best_plateau_score": round(float(plateau_score4[best_c]), 3),
                "best_sharpe": round(float(sh4[best_c]), 2),
                "best_return_pct": round(float(ret4[best_c]), 1),
                "best_max_dd_pct": round(float(mdd4[best_c]), 1),
                "best_util_pct": round(float(util4[best_c]), 1),
                "best_trades": int(trd4[best_c]),
                "profitable_space_pct": round(float(np.mean(ret4 > 0) * 100.0), 1),
            }
        )

        for coord in picked_coords:
            i_sp, i_rg, i_mg, i_st = coord
            sp_val = float(sp_axis[i_sp])
            rg_pct = float(range_lower_pcts[i_rg])
            mg_val = float(martingale_vals[i_mg])
            st_val = float(stop_vals[i_st])

            sizing_style = (
                "Anti-Martingale" if mg_val < 0.94 else ("Martingale" if mg_val > 1.06 else "Equal Size")
            )
            spacing_display = (
                f"${sp_val * last_px:.2f} ({sp_val * 100:.2f}%)"
                if gtype == "arithmetic"
                else (f"{sp_val * 100:.2f}%" if gtype == "geometric" else f"{sp_val:.2f}x ATR (${sp_val * last_atr:.2f})")
            )

            all_plateaus.append(
                {
                    "grid_type": gtype,
                    "type_label_en": gmeta["label_en"],
                    "type_label_zh": gmeta["label_zh"],
                    "spacing_param": round(sp_val, 5),
                    "spacing_display": spacing_display,
                    "range_pct": round(rg_pct * 100.0, 1),
                    "lower_bound_live": round(last_px * (1.0 - rg_pct), 2),
                    "upper_bound_live": round(last_px * (1.0 + rg_pct), 2),
                    "martingale_r": round(mg_val, 2),
                    "sizing_style": sizing_style,
                    "stop_loss_dd_pct": round(st_val * 100.0, 0) if st_val > -0.90 else None,
                    "stop_loss_raw": float(st_val),
                    "plateau_score": round(float(plateau_score4[coord]), 3),
                    "neighborhood_std": round(float(local_std4[coord]), 3),
                    "sharpe": round(float(sh4[coord]), 2),
                    "total_return_pct": round(float(ret4[coord]), 1),
                    "max_drawdown_pct": round(float(mdd4[coord]), 1),
                    "capital_util_pct": round(float(util4[coord]), 1),
                    "trades": int(trd4[coord]),
                }
            )

    all_plateaus.sort(key=lambda x: x["plateau_score"], reverse=True)
    top_5 = all_plateaus[:5]
    for rank, item in enumerate(top_5, start=1):
        item["rank"] = rank
        item["config_id"] = f"P{rank} · {item['grid_type'].upper()}"

    type_summaries.sort(key=lambda x: x["best_plateau_score"], reverse=True)

    return {
        "available": True,
        "symbol": symbol,
        "backend": "cuda" if use_cuda else (str(dev) if dev else "cpu"),
        "total_combos_evaluated": total_combos,
        "hypercube_shape": f"{n_sp}×{n_rg}×{n_mg}×{n_st} × 3 types",
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "anchor_price_live": round(last_px, 2),
        "live_atr14": round(last_atr, 2),
        "type_comparison": type_summaries,
        "top_5_plateaus": top_5,
        "heatmaps": heatmaps_by_type,
        "winning_type": type_summaries[0]["grid_type"],
    }


# ==============================================================================
# STAGE 3: 10,000-PATH BLACK-SWAN MONTE CARLO STRESS TEST (50,000 SIMULATIONS)
# ==============================================================================
def run_gpu_grid_monte_carlo(
    *,
    symbol: str,
    df: pd.DataFrame,
    top_5_configs: list[dict[str, Any]],
    n_paths: int = 10_000,
    steps_per_day: int = 13,       # 13 x 30-min intraday steps/day x 252d = 3,276 steps/year
    break_rate_limit_pct: float = 5.0,
    fee_bps: float = 5.0,
    device_pref: str = "auto",
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Stage 3: Generate 10,000 parallel-universe 1-year intraday paths on RTX 5070 Ti CUDA
    (mixing Empirical Jump-Diffusion + 15% Flash Crash -30% + 15% Slow Bleed + 10% V-Reversal),
    and stress-test ALL Top-5 Stage-2 parameter sets simultaneously (50,000 full simulations!).
    """
    t0 = time.perf_counter()
    dev = get_torch_device(device_pref)
    use_cuda = torch is not None and dev is not None and dev.type == "cuda"

    if not top_5_configs or df is None or len(df) < 40:
        return {"available": False, "reason": "缺少候选网格参数或历史波动率数据"}

    # Scale smoothly if running on CPU so Mac CPU doesn't stall while 5070 Ti runs full 10,000 x 5 = 50,000!
    paths_eff = int(n_paths) if use_cuda else min(int(n_paths), 1_500)
    spd_eff = int(steps_per_day) if use_cuda else min(int(steps_per_day), 4)
    total_steps = 252 * spd_eff

    rets = np.log(df["Close"] / df["Close"].shift(1)).dropna().tail(504).values
    ann_vol = float(np.clip(np.std(rets) * np.sqrt(252.0), 0.14, 1.15))
    step_vol = ann_vol / np.sqrt(float(total_steps))
    last_px = float(df["Close"].iloc[-1])
    live_atr = float(compute_atr(df, 14).iloc[-1])
    live_atr_pct = max(live_atr / max(last_px, 1e-4), 0.01)
    fee_rate = fee_bps / 10_000.0

    # Regime partition across the 10,000 parallel universes:
    # 0: Empirical Fat-Tail Jump-Diffusion (60%)
    # 1: Flash Crash -30%+ Bear Script (15%)
    # 2: Slow Bleed & Sideways Grind (15%)
    # 3: V-Shape Violent Drop + Rally (10%)
    n_emp = int(paths_eff * 0.60)
    n_crash = int(paths_eff * 0.15)
    n_bleed = int(paths_eff * 0.15)
    n_v = paths_eff - n_emp - n_crash - n_bleed

    scenario_labels = np.concatenate(
        [
            np.zeros(n_emp, dtype=np.int32),
            np.ones(n_crash, dtype=np.int32),
            np.full(n_bleed, 2, dtype=np.int32),
            np.full(n_v, 3, dtype=np.int32),
        ]
    )

    k_cfg = len(top_5_configs)
    # Chunk paths to guarantee tiny VRAM usage (< 500 MB even at 10,000 paths x 5 configs!)
    path_chunk = 2_000 if use_cuda else 500

    all_final_ret = np.zeros((k_cfg, paths_eff), dtype=np.float32)
    all_max_dd = np.zeros((k_cfg, paths_eff), dtype=np.float32)
    all_broke_lower = np.zeros((k_cfg, paths_eff), dtype=np.bool_)
    all_stopped = np.zeros((k_cfg, paths_eff), dtype=np.bool_)
    all_trades = np.zeros((k_cfg, paths_eff), dtype=np.float32)

    # Save 24 downsampled sample paths (daily snapshots) for clean fan-chart plotting
    sample_paths_daily: list[list[float]] = []
    sample_scenarios: list[int] = []

    if progress_cb:
        progress_cb(
            f"🎲 Stage 3: simulating {paths_eff:,} intraday Black-Swan paths × {k_cfg} configs = {paths_eff * k_cfg:,} universes on {'RTX 5070 Ti CUDA' if use_cuda else 'CPU'}..."
        )

    for p_start in range(0, paths_eff, path_chunk):
        p_end = min(paths_eff, p_start + path_chunk)
        P = p_end - p_start
        scen_chunk = scenario_labels[p_start:p_end]

        if torch is not None and dev is not None:
            with torch.no_grad():
                torch.manual_seed(1000 + p_start)
                # Generate correlated heavy-tailed intraday log-returns [total_steps, P]
                z = torch.randn((total_steps, P), device=dev, dtype=torch.float32)
                # Student-t scaling (df=5) for realistic equity fat tails
                chi2 = torch.distributions.Chi2(torch.tensor(5.0, device=dev))
                g = chi2.sample((total_steps, P))
                t_scale = torch.rsqrt(g / 5.0) * float(np.sqrt(3.0 / 5.0))
                lr = z * t_scale * step_vol

                # Inject Merton jumps (approx ~6 jump events/year)
                jump_prob = 6.0 / float(total_steps)
                j_mask = (torch.rand((total_steps, P), device=dev) < jump_prob).float()
                j_size = torch.randn((total_steps, P), device=dev) * (ann_vol * 0.14) - 0.015
                lr = lr + j_mask * j_size

                # Inject explicit Black-Swan regimes per universe slice
                t_scen = torch.tensor(scen_chunk, device=dev, dtype=torch.int32)
                t_idx = torch.arange(total_steps, device=dev, dtype=torch.float32).unsqueeze(1) / float(total_steps)

                # Script 1: Flash crash -35% over first quarter + elevated volatility
                is_crash = (t_scen == 1).unsqueeze(0)
                crash_window = (t_idx >= 0.08) & (t_idx <= 0.30)
                lr = torch.where(is_crash & crash_window, lr * 1.45 - (0.42 / (0.22 * total_steps)), lr)

                # Script 2: Slow bleed -28% grind over 9 months
                is_bleed = (t_scen == 2).unsqueeze(0)
                lr = torch.where(is_bleed, lr * 0.90 - (0.30 / float(total_steps)), lr)

                # Script 3: V-Reversal (-32% first 3 months -> +42% recovery)
                is_v = (t_scen == 3).unsqueeze(0)
                v_down = t_idx <= 0.28
                v_up = (t_idx > 0.28) & (t_idx <= 0.72)
                lr = torch.where(is_v & v_down, lr * 1.35 - (0.38 / (0.28 * total_steps)), lr)
                lr = torch.where(is_v & v_up, lr * 1.15 + (0.42 / (0.44 * total_steps)), lr)

                lr = torch.clamp(lr, min=-0.22, max=0.22)
                px_paths = last_px * torch.exp(torch.cumsum(lr, dim=0))  # [total_steps, P]

                # Capture 24 diverse sample paths from first chunk (daily resolution)
                if p_start == 0:
                    stride = max(1, spd_eff)
                    daily_px = px_paths[::stride, :].cpu().numpy()
                    pick_idx = np.linspace(0, P - 1, min(24, P), dtype=int)
                    for pi in pick_idx:
                        sample_paths_daily.append(np.round(daily_px[:, pi], 2).tolist())
                        sample_scenarios.append(int(scen_chunk[pi]))

                # Evaluate ALL Top-5 parameter sets simultaneously on this [total_steps, P] universe!
                # Tensor layout: [K, P] -> 5 * 2,000 = 10,000 live parallel grid bots on CUDA!
                sp_t = torch.tensor([c["spacing_param"] for c in top_5_configs], device=dev, dtype=torch.float32).unsqueeze(1)
                rg_t = torch.tensor([c["range_pct"] / 100.0 for c in top_5_configs], device=dev, dtype=torch.float32).unsqueeze(1)
                mg_t = torch.tensor([c["martingale_r"] for c in top_5_configs], device=dev, dtype=torch.float32).unsqueeze(1)
                st_t = torch.tensor([c["stop_loss_raw"] for c in top_5_configs], device=dev, dtype=torch.float32).unsqueeze(1)
                is_arith = torch.tensor([c["grid_type"] == "arithmetic" for c in top_5_configs], device=dev).unsqueeze(1)
                is_atr = torch.tensor([c["grid_type"] == "atr_dynamic" for c in top_5_configs], device=dev).unsqueeze(1)

                lb = last_px * (1.0 - rg_t)  # [K, 1]
                ub = last_px * (1.0 + rg_t)  # [K, 1]

                base_step = torch.where(is_atr, sp_t * live_atr, sp_t * last_px)
                n_rungs = torch.clamp((last_px - lb) / torch.clamp(base_step, min=1e-4), min=3.0, max=60.0)
                base_frac = 0.70 / n_rungs

                cash = torch.full((k_cfg, P), 0.70, device=dev, dtype=torch.float32)
                shares = torch.full((k_cfg, P), 0.30 / max(last_px, 1e-4), device=dev, dtype=torch.float32)
                last_gpx = torch.full((k_cfg, P), last_px, device=dev, dtype=torch.float32)
                depth = torch.zeros((k_cfg, P), device=dev, dtype=torch.float32)
                peak_eq = torch.ones((k_cfg, P), device=dev, dtype=torch.float32)
                max_dd = torch.zeros((k_cfg, P), device=dev, dtype=torch.float32)
                broke_low = torch.zeros((k_cfg, P), device=dev, dtype=torch.bool)
                stopped = torch.zeros((k_cfg, P), device=dev, dtype=torch.bool)
                trades = torch.zeros((k_cfg, P), device=dev, dtype=torch.float32)

                for step_i in range(total_steps):
                    px = px_paths[step_i].unsqueeze(0)  # [1, P] broadcasts to [K, P]
                    alive = ~stopped

                    # Compute step size per grid type
                    step_px = torch.where(
                        is_arith,
                        sp_t * last_px,
                        torch.where(is_atr, torch.clamp(sp_t * (px * live_atr_pct), min=px * 0.004), sp_t * last_gpx),
                    )

                    broke_low = broke_low | (px < lb)

                    # Buy trigger
                    can_buy = alive & (px <= last_gpx - step_px) & (px >= lb) & (cash > 0.005)
                    mg_mult = torch.clamp(torch.pow(mg_t, torch.clamp(depth, min=0.0, max=8.0)), min=0.25, max=3.5)
                    buy_usd = torch.minimum(cash, base_frac * mg_mult)
                    cash = torch.where(can_buy, cash - buy_usd, cash)
                    shares = torch.where(can_buy, shares + buy_usd * (1.0 - fee_rate) / torch.clamp(px, min=1e-4), shares)
                    depth = torch.where(can_buy, depth + 1.0, depth)
                    last_gpx = torch.where(can_buy, px, last_gpx)
                    trades = trades + can_buy.float()

                    # Sell trigger
                    can_sell = alive & (px >= last_gpx + step_px) & (px <= ub) & (shares > 1e-6)
                    sell_sh = torch.minimum(shares, (base_frac * 1.05) / torch.clamp(px, min=1e-4))
                    shares = torch.where(can_sell, shares - sell_sh, shares)
                    cash = torch.where(can_sell, cash + sell_sh * px * (1.0 - fee_rate), cash)
                    depth = torch.where(can_sell, torch.clamp(depth - 1.0, min=0.0), depth)
                    last_gpx = torch.where(can_sell, px, last_gpx)
                    trades = trades + can_sell.float()

                    # Portfolio equity & drawdown check (every bar!)
                    eq = cash + shares * px
                    peak_eq = torch.maximum(peak_eq, eq)
                    dd = eq / torch.clamp(peak_eq, min=1e-6) - 1.0
                    max_dd = torch.minimum(max_dd, dd)

                    hit_stop = alive & (dd <= st_t)
                    cash = torch.where(hit_stop, cash + shares * px * (1.0 - fee_rate), cash)
                    shares = torch.where(hit_stop, torch.zeros_like(shares), shares)
                    stopped = stopped | hit_stop

                final_eq = cash + shares * px_paths[-1].unsqueeze(0)
                all_final_ret[:, p_start:p_end] = ((final_eq - 1.0) * 100.0).cpu().numpy()
                all_max_dd[:, p_start:p_end] = (max_dd * 100.0).cpu().numpy()
                all_broke_lower[:, p_start:p_end] = broke_low.cpu().numpy()
                all_stopped[:, p_start:p_end] = stopped.cpu().numpy()
                all_trades[:, p_start:p_end] = trades.cpu().numpy()
        else:
            # NumPy CPU fallback
            rng = np.random.default_rng(1000 + p_start)
            z = rng.standard_normal((total_steps, P)).astype(np.float32)
            g = rng.chisquare(5.0, size=(total_steps, P)).astype(np.float32)
            lr = z / np.sqrt(g / 5.0) * np.sqrt(3.0 / 5.0) * step_vol
            t_idx = (np.arange(total_steps, dtype=np.float32) / float(total_steps))[:, None]
            lr[:, scen_chunk == 1] -= np.where((t_idx >= 0.08) & (t_idx <= 0.30), 0.42 / (0.22 * total_steps), 0.0)
            lr[:, scen_chunk == 2] -= 0.30 / float(total_steps)
            lr[:, scen_chunk == 3] -= np.where(t_idx <= 0.28, 0.38 / (0.28 * total_steps), -0.36 / (0.72 * total_steps))
            lr = np.clip(lr, -0.22, 0.22)
            px_paths = last_px * np.exp(np.cumsum(lr, axis=0))

            if p_start == 0:
                stride = max(1, spd_eff)
                daily_px = px_paths[::stride, :]
                for pi in np.linspace(0, P - 1, min(24, P), dtype=int):
                    sample_paths_daily.append(np.round(daily_px[:, pi], 2).tolist())
                    sample_scenarios.append(int(scen_chunk[pi]))

            for ci, cfg in enumerate(top_5_configs):
                sp = float(cfg["spacing_param"])
                rg = float(cfg["range_pct"]) / 100.0
                mg = float(cfg["martingale_r"])
                st = float(cfg["stop_loss_raw"])
                lb, ub = last_px * (1.0 - rg), last_px * (1.0 + rg)
                base_step = sp * live_atr if cfg["grid_type"] == "atr_dynamic" else sp * last_px
                base_frac = 0.70 / float(np.clip((last_px - lb) / max(base_step, 1e-4), 3.0, 60.0))

                cash = np.full(P, 0.70, dtype=np.float32)
                sh = np.full(P, 0.30 / max(last_px, 1e-4), dtype=np.float32)
                lgpx = np.full(P, last_px, dtype=np.float32)
                dep = np.zeros(P, dtype=np.float32)
                pk = np.ones(P, dtype=np.float32)
                mdd = np.zeros(P, dtype=np.float32)
                brk = np.zeros(P, dtype=bool)
                stp = np.zeros(P, dtype=bool)
                trd = np.zeros(P, dtype=np.float32)

                for step_i in range(total_steps):
                    px = px_paths[step_i]
                    alive = ~stp
                    if cfg["grid_type"] == "arithmetic":
                        step_px = np.full(P, sp * last_px, dtype=np.float32)
                    elif cfg["grid_type"] == "atr_dynamic":
                        step_px = np.maximum(sp * px * live_atr_pct, px * 0.004)
                    else:
                        step_px = sp * lgpx
                    brk |= px < lb
                    cb = alive & (px <= lgpx - step_px) & (px >= lb) & (cash > 0.005)
                    bu = np.minimum(cash, base_frac * np.clip(np.power(mg, np.clip(dep, 0.0, 8.0)), 0.25, 3.5))
                    cash = np.where(cb, cash - bu, cash)
                    sh = np.where(cb, sh + bu * (1.0 - fee_rate) / np.maximum(px, 1e-4), sh)
                    dep = np.where(cb, dep + 1.0, dep)
                    lgpx = np.where(cb, px, lgpx)
                    trd += cb.astype(np.float32)

                    cs = alive & (px >= lgpx + step_px) & (px <= ub) & (sh > 1e-6)
                    ss = np.minimum(sh, (base_frac * 1.05) / np.maximum(px, 1e-4))
                    sh = np.where(cs, sh - ss, sh)
                    cash = np.where(cs, cash + ss * px * (1.0 - fee_rate), cash)
                    dep = np.where(cs, np.maximum(dep - 1.0, 0.0), dep)
                    lgpx = np.where(cs, px, lgpx)
                    trd += cs.astype(np.float32)

                    eq = cash + sh * px
                    pk = np.maximum(pk, eq)
                    dd = eq / np.maximum(pk, 1e-6) - 1.0
                    mdd = np.minimum(mdd, dd)
                    hs = alive & (dd <= st)
                    cash = np.where(hs, cash + sh * px * (1.0 - fee_rate), cash)
                    sh = np.where(hs, 0.0, sh)
                    stp |= hs

                feq = cash + sh * px_paths[-1]
                all_final_ret[ci, p_start:p_end] = (feq - 1.0) * 100.0
                all_max_dd[ci, p_start:p_end] = mdd * 100.0
                all_broke_lower[ci, p_start:p_end] = brk
                all_stopped[ci, p_start:p_end] = stp
                all_trades[ci, p_start:p_end] = trd

    # Aggregate strict pass/fail risk metrics for each of the Top-5 parameter sets
    stress_results: list[dict[str, Any]] = []
    for ci, cfg in enumerate(top_5_configs):
        rets_c = all_final_ret[ci]
        mdd_c = all_max_dd[ci]
        brk_c = all_broke_lower[ci]
        stp_c = all_stopped[ci]
        trd_c = all_trades[ci]

        # Empirical break rate (excluding forced -30% artificial crash scripts -> realistic break rate)
        emp_mask = scenario_labels == 0
        emp_break_pct = float(np.mean(brk_c[emp_mask]) * 100.0) if np.any(emp_mask) else float(np.mean(brk_c) * 100.0)
        all_break_pct = float(np.mean(brk_c) * 100.0)
        stop_out_pct = float(np.mean(stp_c) * 100.0)

        # 95th percentile worst floating drawdown (i.e. 5th percentile of negative mdd_c)
        dd_p95 = float(np.percentile(mdd_c, 5))
        dd_p99 = float(np.percentile(mdd_c, 1))
        med_ret = float(np.median(rets_c))
        win_prob = float(np.mean(rets_c > 0) * 100.0)

        passed = (emp_break_pct <= break_rate_limit_pct) and (dd_p95 >= -35.0)
        # Compact return histogram for best config
        hist_cnt, hist_edges = np.histogram(rets_c, bins=50, range=(-50.0, 60.0))

        stress_results.append(
            {
                "rank": cfg.get("rank", ci + 1),
                "config_id": cfg.get("config_id", f"P{ci + 1}"),
                "grid_type": cfg["grid_type"],
                "spacing_display": cfg["spacing_display"],
                "range_pct": cfg["range_pct"],
                "sizing_style": cfg["sizing_style"],
                "stop_loss_dd_pct": cfg["stop_loss_dd_pct"],
                "empirical_break_rate_pct": round(emp_break_pct, 2),
                "black_swan_break_rate_pct": round(all_break_pct, 2),
                "stop_out_rate_pct": round(stop_out_pct, 2),
                "float_dd_p95_pct": round(dd_p95, 2),
                "float_dd_p99_pct": round(dd_p99, 2),
                "median_1y_return_pct": round(med_ret, 2),
                "prob_profit_pct": round(win_prob, 1),
                "median_annual_trades": int(np.median(trd_c)),
                "passed_gate": bool(passed),
                "gate_badge_en": "✅ PASS" if passed else "❌ FAIL (Break Risk Too High)",
                "gate_badge_zh": "✅ 通过压力测试" if passed else "❌ 淘汰 (破网率超标)",
                "ret_hist": {
                    "counts": [int(x) for x in hist_cnt],
                    "centers": [round(float((hist_edges[i] + hist_edges[i + 1]) / 2.0), 1) for i in range(len(hist_cnt))],
                },
            }
        )

    # Pick best passed config (or lowest empirical break rate if none strictly passed)
    passed_list = [r for r in stress_results if r["passed_gate"]]
    if passed_list:
        champion = max(passed_list, key=lambda x: (x["median_1y_return_pct"], -x["empirical_break_rate_pct"]))
    else:
        champion = min(stress_results, key=lambda x: x["empirical_break_rate_pct"])

    return {
        "available": True,
        "symbol": symbol,
        "backend": "cuda" if use_cuda else (str(dev) if dev else "cpu"),
        "n_paths": paths_eff,
        "steps_per_day": spd_eff,
        "total_simulations": paths_eff * k_cfg,
        "break_rate_limit_pct": break_rate_limit_pct,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "any_passed": bool(passed_list),
        "champion_config_id": champion["config_id"],
        "results": stress_results,
        "sample_paths_daily": sample_paths_daily,
        "sample_scenarios": sample_scenarios,
    }


# ==============================================================================
# STAGE 4: XGBOOST + LIGHTGBM "5-DAY ONE-SIDED CRASH SWITCH" & LIVE ORDER LADDER
# ==============================================================================
CRASH_FEATURES = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_21d",
    "sma5_slope",
    "sma20_slope",
    "sma50_slope",
    "dist_sma20",
    "dist_sma50",
    "rsi14",
    "adx14",
    "atr_ratio_10_50",
    "rvol",
    "down_vol_ratio_5d",
    "cmf20",
    "qqq_ret_5d",
    "soxx_ret_5d",
    "vix_level",
    "vix_chg_5d",
    "tnx_chg_10d",
]


def train_predict_grid_crash_switch(
    *,
    symbol: str,
    ohlcv_map: dict[str, pd.DataFrame],
    horizon_days: int = 5,
    device_pref: str = "auto",
) -> dict[str, Any]:
    """
    Stage 4: Walk-Forward XGBoost (CUDA) + LightGBM binary classifier predicting ONE thing:
      "Will the next 5 trading days enter a sharp one-sided crash/breakdown?"

    Outputs live crash probability, Traffic-Light Grid Switch state, OOS ROC-AUC,
    and historical crash-avoidance diagnostics.
    """
    t0 = time.perf_counter()
    dev = get_torch_device(device_pref)
    use_cuda = torch is not None and dev is not None and dev.type == "cuda"

    df = ohlcv_map.get(symbol)
    if df is None or len(df) < 160:
        return {
            "available": False,
            "reason": "历史数据不足 160 个交易日，暂无法训练 XGBoost/LightGBM 单边暴跌过滤开关",
        }

    c = df["Close"]
    v = df["Volume"].replace(0.0, np.nan).ffill().fillna(1.0)
    ret_1d = c.pct_change()
    sma5 = c.rolling(5).mean()
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    atr10 = compute_atr(df, 10)
    atr50 = compute_atr(df, 50)

    feat = pd.DataFrame(index=df.index)
    feat["ret_1d"] = ret_1d
    feat["ret_3d"] = c.pct_change(3)
    feat["ret_5d"] = c.pct_change(5)
    feat["ret_10d"] = c.pct_change(10)
    feat["ret_21d"] = c.pct_change(21)
    feat["sma5_slope"] = sma5.pct_change(3)
    feat["sma20_slope"] = sma20.pct_change(5)
    feat["sma50_slope"] = sma50.pct_change(10)
    feat["dist_sma20"] = c / sma20 - 1.0
    feat["dist_sma50"] = c / sma50 - 1.0
    feat["rsi14"] = compute_rsi(c, 14) / 100.0
    feat["adx14"] = compute_adx(df, 14) / 100.0
    feat["atr_ratio_10_50"] = atr10 / atr50.replace(0.0, np.nan)
    feat["rvol"] = v / v.rolling(20).mean().replace(0.0, np.nan)

    # 5-day selling pressure volume share
    down_v = np.where(ret_1d < 0, v, 0.0)
    feat["down_vol_ratio_5d"] = (
        pd.Series(down_v, index=df.index).rolling(5).sum() / v.rolling(5).sum().replace(0.0, np.nan)
    )
    feat["cmf20"] = compute_cmf(df, 20)

    # Macro context features
    def _align_ret(sym_m: str, n_d: int, default: float = 0.0) -> pd.Series:
        m_df = ohlcv_map.get(sym_m)
        if m_df is None or m_df.empty:
            return pd.Series(default, index=df.index)
        return m_df["Close"].reindex(df.index).ffill().pct_change(n_d).fillna(default)

    feat["qqq_ret_5d"] = _align_ret("QQQ", 5)
    feat["soxx_ret_5d"] = _align_ret("SOXX", 5)
    vix_df = ohlcv_map.get("^VIX")
    if vix_df is not None and not vix_df.empty:
        vix_s = vix_df["Close"].reindex(df.index).ffill().fillna(18.0)
        feat["vix_level"] = vix_s / 100.0
        feat["vix_chg_5d"] = vix_s.pct_change(5).fillna(0.0)
    else:
        feat["vix_level"] = 0.18
        feat["vix_chg_5d"] = 0.0
    feat["tnx_chg_10d"] = _align_ret("^TNX", 10)

    # Target: Next 5 trading days sharp one-sided breakdown!
    # Dynamic volatility-scaled crash threshold (at least -4.5%, or -1.75 * 5d realized vol)
    vol_5d = ret_1d.rolling(60).std() * np.sqrt(horizon_days)
    crash_thr = -np.maximum(0.045, (vol_5d * 1.65).fillna(0.05))
    fwd_5d = c.shift(-horizon_days) / c - 1.0
    fwd_min_low = df["Low"].iloc[::-1].rolling(horizon_days).min().iloc[::-1].shift(-1) / c - 1.0
    target = ((fwd_5d <= crash_thr) | (fwd_min_low <= crash_thr * 1.25)).astype(np.float32)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    live_X = feat[CRASH_FEATURES].iloc[[-1]].fillna(0.0).values.astype(np.float32)

    data = feat[CRASH_FEATURES].copy()
    data["target"] = target
    data["fwd_5d"] = fwd_5d
    data = data.dropna()
    if len(data) < 120 or len(np.unique(data["target"].values)) < 2:
        return {"available": False, "reason": "历史样本或暴跌事件样本不足以完成二分类训练"}

    X_all = data[CRASH_FEATURES].values.astype(np.float32)
    y_all = data["target"].values.astype(np.float32)
    fwd_all = data["fwd_5d"].values.astype(np.float32)

    def _fit_pred_pair(X_tr: np.ndarray, y_tr: np.ndarray, X_ev: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        pos_w = float((len(y_tr) - np.sum(y_tr)) / max(np.sum(y_tr), 1.0))
        pos_w = min(pos_w, 6.0)
        p_xgb = np.full(len(X_ev), float(np.mean(y_tr)), dtype=np.float32)
        p_lgb = np.full(len(X_ev), float(np.mean(y_tr)), dtype=np.float32)
        f_imp = None

        if xgb is not None and len(np.unique(y_tr)) > 1:
            params = {
                "n_estimators": 180,
                "max_depth": 4,
                "learning_rate": 0.04,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": pos_w,
                "reg_lambda": 3.0,
                "tree_method": "hist",
                "device": "cuda" if use_cuda else "cpu",
                "eval_metric": "logloss",
                "random_state": 42,
            }
            try:
                m_x = xgb.XGBClassifier(**params)
                m_x.fit(X_tr, y_tr, verbose=False)
            except Exception:
                params["device"] = "cpu"
                m_x = xgb.XGBClassifier(**params)
                m_x.fit(X_tr, y_tr, verbose=False)
            p_xgb = m_x.predict_proba(X_ev)[:, 1].astype(np.float32)
            f_imp = getattr(m_x, "feature_importances_", None)

        if lgb is not None and len(np.unique(y_tr)) > 1:
            try:
                m_l = lgb.LGBMClassifier(
                    n_estimators=180,
                    max_depth=4,
                    num_leaves=15,
                    learning_rate=0.04,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=pos_w,
                    reg_lambda=3.0,
                    random_state=42,
                    verbosity=-1,
                )
                m_l.fit(X_tr, y_tr)
                p_lgb = m_l.predict_proba(X_ev)[:, 1].astype(np.float32)
            except Exception as exc:
                logger.debug("LightGBM fit error: %s", exc)
                p_lgb = p_xgb
        elif HistGradientBoostingClassifier is not None and len(np.unique(y_tr)) > 1:
            # Built-in LightGBM-style histogram GBDT fallback (no macOS libomp dependency)
            try:
                sample_w = np.where(y_tr > 0.5, pos_w, 1.0)
                m_h = HistGradientBoostingClassifier(
                    max_iter=160,
                    max_depth=4,
                    learning_rate=0.05,
                    l2_regularization=3.0,
                    random_state=42,
                )
                m_h.fit(X_tr, y_tr, sample_weight=sample_w)
                p_lgb = m_h.predict_proba(X_ev)[:, 1].astype(np.float32)
                if xgb is None:
                    p_xgb = p_lgb
            except Exception:
                p_lgb = p_xgb
        else:
            p_lgb = p_xgb

        return p_xgb, p_lgb, f_imp

    # Strict Purged Expanding Walk-Forward Validation (4 folds, embargo=10 bars)
    n_obs = len(X_all)
    bounds = np.linspace(int(n_obs * 0.45), n_obs, 5, dtype=int)
    oos_p, oos_y, oos_fwd = [], [], []
    for fi in range(4):
        va_s, va_e = int(bounds[fi]), int(bounds[fi + 1])
        tr_e = max(50, va_s - 10)
        if va_e <= va_s + 10 or len(np.unique(y_all[:tr_e])) < 2:
            continue
        px_v, pl_v, _ = _fit_pred_pair(X_all[:tr_e], y_all[:tr_e], X_all[va_s:va_e])
        oos_p.append(0.55 * px_v + 0.45 * pl_v)
        oos_y.append(y_all[va_s:va_e])
        oos_fwd.append(fwd_all[va_s:va_e])

    oos_auc = None
    crash_recall = None
    avoided_avg_drop_pct = None
    if oos_p:
        ap = np.concatenate(oos_p)
        ay = np.concatenate(oos_y)
        af = np.concatenate(oos_fwd)
        if roc_auc_score is not None and len(np.unique(ay)) > 1:
            oos_auc = round(float(roc_auc_score(ay, ap)), 3)
        red_mask = ap >= 0.50
        if np.any(ay > 0.5):
            crash_recall = round(float(np.mean(red_mask[ay > 0.5]) * 100.0), 1)
        if np.any(red_mask):
            avoided_avg_drop_pct = round(float(np.mean(af[red_mask]) * 100.0), 2)

    # Full-history fit -> Live Today's 5-Day Crash Risk Probability
    live_x_p, live_l_p, imp = _fit_pred_pair(X_all, y_all, live_X)
    prob_xgb = float(live_x_p[0]) * 100.0
    prob_lgb = float(live_l_p[0]) * 100.0
    prob_ens = 0.55 * prob_xgb + 0.45 * prob_lgb

    if prob_ens >= 55.0:
        state_code = "PAUSE_RED"
        badge_en = "🔴 RED LIGHT · PAUSE GRID BUYS"
        badge_zh = "🔴 单边暴跌红灯 · 暂停网格买入"
        directive_en = "High 5-day breakdown probability. Freeze all lower buy orders, preserve cash bullets, and wait for stabilization before re-enabling."
        directive_zh = "模型预警未来 5 天存在较高单边下跌风险！立即暂停下方网格买单、保留现金子弹，等跌势企稳转绿灯后再恢复。"
    elif prob_ens >= 35.0:
        state_code = "CAUTION_AMBER"
        badge_en = "🟡 AMBER CAUTION · HALF GRID SIZE"
        badge_zh = "🟡 波动黄灯警戒 · 减半单格仓位"
        directive_en = "Elevated downside turbulence. Cut new buy-rung order size by 50% and widen spacing slightly."
        directive_zh = "短期下行波动抬升，建议新触发买单仓位减半，避免过早打光子弹。"
    else:
        state_code = "ACTIVE_GREEN"
        badge_en = "🟢 GREEN LIGHT · GRID ACTIVE"
        badge_zh = "🟢 震荡安全绿灯 · 开启网格高抛低吸"
        directive_en = "Low 5-day one-sided crash risk. Safe to place today's buy & sell limit ladder per your champion plateau parameters."
        directive_zh = "未来 5 天单边暴跌风险低，可按最优高原参数挂好今日上下网格限价单。"

    top_feats = []
    if imp is not None and len(imp) == len(CRASH_FEATURES):
        for idx in np.argsort(-imp)[:6]:
            top_feats.append({"feature": CRASH_FEATURES[idx], "importance": round(float(imp[idx]), 4)})

    return {
        "available": True,
        "symbol": symbol,
        "backend": "cuda" if use_cuda else "cpu",
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "crash_prob_5d_pct": round(prob_ens, 1),
        "xgb_prob_pct": round(prob_xgb, 1),
        "lgb_prob_pct": round(prob_lgb, 1),
        "state_code": state_code,
        "badge_en": badge_en,
        "badge_zh": badge_zh,
        "directive_en": directive_en,
        "directive_zh": directive_zh,
        "oos_auc": oos_auc,
        "oos_crash_recall_pct": crash_recall,
        "red_light_avg_5d_ret_pct": avoided_avg_drop_pct,
        "base_crash_rate_pct": round(float(np.mean(y_all) * 100.0), 1),
        "top_features": top_feats,
    }


def build_live_grid_order_ladder(
    *,
    last_price: float,
    live_atr: float,
    config: dict[str, Any],
    grid_capital_usd: float = 20_000.0,
    current_shares_held: float = 0.0,
    ml_state_code: str = "ACTIVE_GREEN",
) -> dict[str, Any]:
    """
    Generate today's concrete pre-market broker limit-order ladder (exact Buy/Sell prices,
    shares per rung, cumulative bullets deployed, and remaining cash reserve).
    """
    gtype = str(config.get("grid_type", "geometric"))
    sp = float(config.get("spacing_param", 0.015))
    rg_pct = float(config.get("range_pct", 20.0)) / 100.0
    mg_r = float(config.get("martingale_r", 1.0))

    lower_b = round(last_price * (1.0 - rg_pct), 2)
    upper_b = round(last_price * (1.0 + rg_pct), 2)

    # Generate BUY rungs downward from last_price -> lower_b (up to 12 actionable levels)
    buy_prices: list[float] = []
    px = last_price
    while len(buy_prices) < 12:
        if gtype == "arithmetic":
            nxt = px - sp * last_price
        elif gtype == "atr_dynamic":
            nxt = px - max(sp * live_atr, last_price * 0.004)
        else:
            nxt = px * (1.0 - sp)
        if nxt < lower_b * 0.995 or nxt <= 0:
            break
        buy_prices.append(round(nxt, 2))
        px = nxt

    # Generate SELL rungs upward from last_price -> upper_b (up to 8 take-profit levels)
    sell_prices: list[float] = []
    px = last_price
    while len(sell_prices) < 8:
        if gtype == "arithmetic":
            nxt = px + sp * last_price
        elif gtype == "atr_dynamic":
            nxt = px + max(sp * live_atr, last_price * 0.004)
        else:
            nxt = px * (1.0 + sp)
        if nxt > upper_b * 1.005:
            break
        sell_prices.append(round(nxt, 2))
        px = nxt

    held_value = current_shares_held * last_price
    remaining_cash = max(0.0, grid_capital_usd - held_value)

    # Size multiplier from ML Crash Switch
    ml_mult = 0.0 if ml_state_code == "PAUSE_RED" else (0.5 if ml_state_code == "CAUTION_AMBER" else 1.0)

    # Allocate remaining cash across buy rungs according to Martingale / Anti-Martingale ratio
    raw_weights = [float(np.clip(mg_r ** idx, 0.25, 3.5)) for idx in range(len(buy_prices))]
    w_sum = sum(raw_weights) if raw_weights else 1.0

    orders: list[dict[str, Any]] = []
    # Upper sell limit orders (highest price first, broker order-book style)
    sell_sh_per_rung = int(current_shares_held / max(len(sell_prices), 1)) if current_shares_held > 0 else 0
    for idx in range(len(sell_prices) - 1, -1, -1):
        p = sell_prices[idx]
        est_sh = sell_sh_per_rung if sell_sh_per_rung > 0 else max(1, int((grid_capital_usd * 0.08) / max(p, 0.01)))
        orders.append(
            {
                "side": "SELL LIMIT",
                "level": f"+{idx + 1}",
                "price": p,
                "dist_pct": round((p / last_price - 1.0) * 100.0, 2),
                "shares": est_sh,
                "notional_usd": round(est_sh * p, 0),
                "status": "ACTIVE TP" if current_shares_held > 0 else "AFTER FILL",
            }
        )

    # Lower buy limit orders
    cum_usd = 0.0
    for idx, p in enumerate(buy_prices):
        alloc_usd = remaining_cash * (raw_weights[idx] / w_sum) * ml_mult
        sh = int(alloc_usd / max(p, 0.01))
        actual_usd = sh * p
        cum_usd += actual_usd
        orders.append(
            {
                "side": "BUY LIMIT",
                "level": f"-{idx + 1}",
                "price": p,
                "dist_pct": round((p / last_price - 1.0) * 100.0, 2),
                "shares": sh,
                "notional_usd": round(actual_usd, 0),
                "status": "⏸️ PAUSED (ML RED)" if ml_mult == 0.0 else ("🟡 HALF (ML AMBER)" if ml_mult < 1.0 else "🟢 READY"),
            }
        )

    stop_dd = config.get("stop_loss_dd_pct")
    hard_stop_px = round(lower_b * 0.96, 2) if stop_dd is None else round(last_price * (1.0 + float(stop_dd) / 100.0), 2)

    return {
        "last_price": round(last_price, 2),
        "lower_bound": lower_b,
        "upper_bound": upper_b,
        "hard_stop_price": hard_stop_px,
        "grid_capital_usd": round(grid_capital_usd, 0),
        "held_value_usd": round(held_value, 0),
        "inventory_pct": round(held_value / max(grid_capital_usd, 1.0) * 100.0, 1),
        "remaining_cash_usd": round(remaining_cash, 0),
        "buy_rungs_count": len(buy_prices),
        "sell_rungs_count": len(sell_prices),
        "buy_prices": buy_prices,
        "sell_prices": sell_prices,
        "orders": orders,
    }


def run_complete_grid_lab_for_symbol(
    *,
    symbol: str,
    ohlcv_map: dict[str, pd.DataFrame],
    n_mc_paths: int = 10_000,
    steps_per_day: int = 13,
    break_rate_limit_pct: float = 5.0,
    high_density_gpu: bool = True,
    device_pref: str = "auto",
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Run Stages 2 -> 3 -> 4 end-to-end for one selected ticker with automatic VRAM cleanup."""
    df = ohlcv_map.get(symbol)
    if df is None or len(df) < 60:
        return {"available": False, "symbol": symbol, "reason": "Insufficient historical data for Grid Lab."}

    def _p(pct: float, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)

    _p(0.10, f"⚡ Stage 2: 4D Hyper-Cube GPU Grid Search + SciPy Gaussian Plateau on {symbol}...")
    stage2 = run_gpu_4d_grid_hypercube(
        symbol=symbol,
        df=df,
        high_density_gpu=high_density_gpu,
        device_pref=device_pref,
        progress_cb=lambda m: _p(0.32, m),
    )
    if not stage2.get("available"):
        return {"available": False, "symbol": symbol, "reason": stage2.get("reason", "Stage 2 failed")}

    _p(0.55, f"🎲 Stage 3: {n_mc_paths:,}-Path Black-Swan Intraday Monte Carlo Stress Test (×5 configs)...")
    stage3 = run_gpu_grid_monte_carlo(
        symbol=symbol,
        df=df,
        top_5_configs=stage2["top_5_plateaus"],
        n_paths=n_mc_paths,
        steps_per_day=steps_per_day,
        break_rate_limit_pct=break_rate_limit_pct,
        device_pref=device_pref,
        progress_cb=lambda m: _p(0.70, m),
    )

    _p(0.84, f"🧠 Stage 4: Walk-Forward XGBoost (CUDA) + LightGBM 5-Day Crash Switch on {symbol}...")
    stage4 = train_predict_grid_crash_switch(
        symbol=symbol,
        ohlcv_map=ohlcv_map,
        horizon_days=5,
        device_pref=device_pref,
    )

    # Free PyTorch CUDA memory cleanly so 5070 Ti 16GB VRAM stays spotless for Ollama
    release_gpu_memory()

    # Select champion parameter config (Stage 3 winner -> Stage 2 top plateau)
    champ_id = (stage3 or {}).get("champion_config_id")
    champ_cfg = next(
        (c for c in stage2["top_5_plateaus"] if c["config_id"] == champ_id),
        stage2["top_5_plateaus"][0],
    )

    _p(1.0, f"✅ Grid Lab Stages 2–4 completed for {symbol}!")
    return {
        "available": True,
        "symbol": symbol,
        "stage2_hypercube": stage2,
        "stage3_montecarlo": stage3,
        "stage4_ml_switch": stage4,
        "champion_config": champ_cfg,
    }
