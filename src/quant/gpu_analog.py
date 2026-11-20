"""[GPU Engine #1] PyTorch CUDA Vectorized Historical Analog Search Engine.

Searches 10+ years of historical multi-channel price/volume/RS/volatility trajectories
across the entire stock universe using GPU Tensor Core matrix multiplication to compute
empirical 21-day (1M) and 42-day (2M) forward return, win-rate, and MAE/MFE distributions.
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
from src.quant.indicators import compute_rsi


def _build_window_features(
    df: pd.DataFrame,
    qqq_close: pd.Series | None,
    window: int = 30,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]:
    """
    Construct dense multi-channel pattern descriptors for every sliding window of length `window`.

    Channels per window (total dim = 2*window + 6 = 66 for window=30):
      - Normalized 30-day cumulative log-return shape (standardized by window volatility)
      - 30-day log-volume relative profile (vs window mean)
      - Context scalars: 52w drawdown, RSI/100, 21d RS vs QQQ, 63d return, vol regime, up/down vol ratio
    """
    n = len(df)
    if n < window + 5:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), [], {}

    close = df["Close"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)
    vol = np.maximum(df["Volume"].values.astype(np.float64), 1.0)
    dates = [str(d.date()) for d in df.index]

    log_c = np.log(np.maximum(close, 1e-6))
    log_v = np.log(vol)
    ret_1d = np.diff(log_c, prepend=log_c[0])
    rsi_arr = compute_rsi(df["Close"], 14).values.astype(np.float64) / 100.0

    # Rolling 252d high drawdown
    roll_high = pd.Series(high).rolling(252, min_periods=20).max().values
    dd_52w = (close / np.maximum(roll_high, 1e-6)) - 1.0

    # Aligned QQQ 21d return for relative strength
    if qqq_close is not None and not qqq_close.empty:
        q_aligned = qqq_close.reindex(df.index).ffill().bfill().values.astype(np.float64)
        q_log = np.log(np.maximum(q_aligned, 1e-6))
    else:
        q_log = np.zeros_like(log_c)

    feats: list[np.ndarray] = []
    valid_indices: list[int] = []
    valid_dates: list[str] = []

    fwd_21 = np.full(n, np.nan, dtype=np.float64)
    fwd_42 = np.full(n, np.nan, dtype=np.float64)
    mae_21 = np.full(n, np.nan, dtype=np.float64)
    mfe_21 = np.full(n, np.nan, dtype=np.float64)

    for i in range(window - 1, n):
        c_win = log_c[i - window + 1 : i + 1]
        c_shape = c_win - c_win[0]
        vol_scale = max(float(np.std(ret_1d[i - window + 1 : i + 1])) * np.sqrt(window), 0.03)
        c_norm = np.clip(c_shape / vol_scale, -4.0, 4.0)

        v_win = log_v[i - window + 1 : i + 1]
        v_norm = np.clip((v_win - np.mean(v_win)) / max(float(np.std(v_win)), 0.25), -3.0, 3.0) * 0.45

        rs_21 = (
            (log_c[i] - log_c[max(0, i - 21)]) - (q_log[i] - q_log[max(0, i - 21)])
            if i >= 21
            else 0.0
        )
        ret_63 = (log_c[i] - log_c[max(0, i - 63)]) if i >= 63 else float(c_shape[-1])
        rvol_20 = float(np.std(ret_1d[max(0, i - 19) : i + 1]) * np.sqrt(252))
        up_v = float(np.sum(vol[max(0, i - 19) : i + 1][ret_1d[max(0, i - 19) : i + 1] > 0]))
        dn_v = float(np.sum(vol[max(0, i - 19) : i + 1][ret_1d[max(0, i - 19) : i + 1] < 0]))
        ud_ratio = np.tanh((up_v - dn_v) / max(up_v + dn_v, 1.0))

        # Emphasize macro context scalars slightly for regime matching
        ctx = np.array(
            [
                np.clip(dd_52w[i], -0.8, 0.0) * 2.5,
                (rsi_arr[i] - 0.5) * 2.2,
                np.clip(rs_21, -0.4, 0.4) * 3.0,
                np.clip(ret_63, -0.6, 0.6) * 1.5,
                np.clip(rvol_20, 0.1, 1.2) * 1.2,
                ud_ratio * 1.5,
            ],
            dtype=np.float64,
        )

        vec = np.concatenate([c_norm, v_norm, ctx]).astype(np.float32)
        norm_val = float(np.linalg.norm(vec))
        if norm_val > 1e-6:
            vec /= norm_val

        feats.append(vec)
        valid_indices.append(i)
        valid_dates.append(dates[i])

        if i + 21 < n:
            fwd_21[i] = (close[i + 21] / close[i] - 1.0) * 100.0
            win_low = float(np.min(low[i + 1 : i + 22]))
            win_high = float(np.max(high[i + 1 : i + 22]))
            mae_21[i] = (win_low / close[i] - 1.0) * 100.0
            mfe_21[i] = (win_high / close[i] - 1.0) * 100.0
        if i + 42 < n:
            fwd_42[i] = (close[i + 42] / close[i] - 1.0) * 100.0

    idx_arr = np.array(valid_indices, dtype=np.int64)
    outcomes = {
        "fwd_21": fwd_21[idx_arr],
        "fwd_42": fwd_42[idx_arr],
        "mae_21": mae_21[idx_arr],
        "mfe_21": mfe_21[idx_arr],
    }
    return np.vstack(feats), idx_arr, valid_dates, outcomes


def run_gpu_historical_analog_search(
    ohlcv_map: dict[str, pd.DataFrame],
    target_symbols: list[str],
    window: int = 30,
    top_k: int = 60,
    device_pref: str = "auto",
) -> dict[str, Any]:
    """
    Build a cross-universe 10-year pattern database on RTX 5070 Ti CUDA and match
    every stock's current 30-day trajectory against all historical non-overlapping windows.
    """
    t0 = time.perf_counter()
    dev = get_torch_device(device_pref)
    qqq_close = ohlcv_map["QQQ"]["Close"] if "QQQ" in ohlcv_map else None

    db_vecs: list[np.ndarray] = []
    db_symbols: list[str] = []
    db_dates: list[str] = []
    db_fwd21: list[np.ndarray] = []
    db_fwd42: list[np.ndarray] = []
    db_mae21: list[np.ndarray] = []
    db_mfe21: list[np.ndarray] = []

    query_vecs: dict[str, np.ndarray] = {}
    query_dates: dict[str, str] = {}

    for sym in target_symbols:
        df = ohlcv_map.get(sym)
        if df is None or len(df) < window + 5:
            continue
        vecs, _idx, dates, outcomes = _build_window_features(df, qqq_close, window=window)
        if len(vecs) == 0:
            continue
        # Latest window is the live query
        query_vecs[sym] = vecs[-1]
        query_dates[sym] = dates[-1]

        # Historical database excludes the last 45 trading days (prevent outcome leakage & overlap)
        if len(vecs) > 45:
            cutoff = len(vecs) - 45
            # Stride by 3 days to keep analogs diverse & non-duplicate
            keep = np.arange(0, cutoff, 3)
            valid_mask = np.isfinite(outcomes["fwd_21"][keep])
            keep = keep[valid_mask]
            if len(keep) > 0:
                db_vecs.append(vecs[keep])
                db_symbols.extend([sym] * len(keep))
                db_dates.extend([dates[k] for k in keep])
                db_fwd21.append(outcomes["fwd_21"][keep])
                db_fwd42.append(outcomes["fwd_42"][keep])
                db_mae21.append(outcomes["mae_21"][keep])
                db_mfe21.append(outcomes["mfe_21"][keep])

    if not db_vecs or not query_vecs:
        return {
            "backend": str(dev) if dev else "cpu",
            "total_historical_patterns": 0,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "per_symbol": {},
        }

    mat_db = np.vstack(db_vecs).astype(np.float32)  # [N, D]
    arr_fwd21 = np.concatenate(db_fwd21)
    arr_fwd42 = np.concatenate(db_fwd42)
    arr_mae21 = np.concatenate(db_mae21)
    arr_mfe21 = np.concatenate(db_mfe21)
    q_syms = list(query_vecs.keys())
    mat_q = np.vstack([query_vecs[s] for s in q_syms]).astype(np.float32)  # [Q, D]

    k_eff = min(top_k, len(mat_db))
    use_cuda = torch is not None and dev is not None and dev.type == "cuda"

    if torch is not None and dev is not None:
        with torch.no_grad():
            t_db = torch.tensor(mat_db, dtype=torch.float32, device=dev)
            t_q = torch.tensor(mat_q, dtype=torch.float32, device=dev)
            # Cosine similarity matrix [Q, N] via Tensor Core GEMM
            sim = torch.matmul(t_q, t_db.T)
            top_vals, top_idx = torch.topk(sim, k=k_eff, dim=1)
            sim_scores = top_vals.cpu().numpy()
            indices = top_idx.cpu().numpy()
    else:
        sim = mat_q @ mat_db.T
        indices = np.argsort(-sim, axis=1)[:, :k_eff]
        sim_scores = np.take_along_axis(sim, indices, axis=1)

    per_symbol: dict[str, dict[str, Any]] = {}
    for qi, sym in enumerate(q_syms):
        idx = indices[qi]
        sc = sim_scores[qi]
        f21 = arr_fwd21[idx]
        f42 = arr_fwd42[idx]
        f42_valid = f42[np.isfinite(f42)]
        mae = arr_mae21[idx]
        mfe = arr_mfe21[idx]

        # Similarity-weighted win rate & expectancy
        weights = np.clip(sc, 0.05, 1.0)
        weights = weights / np.sum(weights)

        win_rate_21 = float(np.sum(weights[f21 > 0]) * 100.0)
        strong_win_21 = float(np.mean(f21 >= 8.0) * 100.0)
        med_21 = float(np.median(f21))
        mean_21 = float(np.sum(weights * f21))
        p10_21 = float(np.percentile(f21, 10))
        p90_21 = float(np.percentile(f21, 90))

        pos_sum = float(np.sum(f21[f21 > 0]))
        neg_sum = float(np.abs(np.sum(f21[f21 < 0])))
        profit_factor = round(pos_sum / neg_sum, 2) if neg_sum > 1e-6 else 3.0

        win_rate_42 = float(np.mean(f42_valid > 0) * 100.0) if len(f42_valid) > 5 else win_rate_21
        med_42 = float(np.median(f42_valid)) if len(f42_valid) > 5 else med_21 * 1.4

        med_mae = float(np.median(mae))
        med_mfe = float(np.median(mfe))

        # Analog Edge Score (0-100)
        edge_score = 50.0 + (win_rate_21 - 50.0) * 1.1 + np.clip(med_21, -12.0, 12.0) * 1.8
        edge_score += np.clip((profit_factor - 1.2) * 8.0, -10.0, 12.0)
        edge_score = round(max(5.0, min(98.0, edge_score)), 1)

        top_matches = []
        for rank in range(min(5, len(idx))):
            m_i = int(idx[rank])
            top_matches.append(
                {
                    "symbol": db_symbols[m_i],
                    "date": db_dates[m_i],
                    "similarity": round(float(sc[rank]), 3),
                    "fwd_21d_pct": round(float(arr_fwd21[m_i]), 2),
                    "fwd_42d_pct": round(float(arr_fwd42[m_i]), 2)
                    if np.isfinite(arr_fwd42[m_i])
                    else None,
                    "mae_21d_pct": round(float(arr_mae21[m_i]), 2),
                }
            )

        per_symbol[sym] = {
            "query_date": query_dates[sym],
            "matches_used": int(k_eff),
            "avg_similarity": round(float(np.mean(sc[:15])), 3),
            "win_rate_21d_pct": round(win_rate_21, 1),
            "strong_win_21d_pct": round(strong_win_21, 1),
            "median_ret_21d_pct": round(med_21, 2),
            "weighted_mean_21d_pct": round(mean_21, 2),
            "tail_p10_21d_pct": round(p10_21, 2),
            "upside_p90_21d_pct": round(p90_21, 2),
            "profit_factor_21d": profit_factor,
            "win_rate_42d_pct": round(win_rate_42, 1),
            "median_ret_42d_pct": round(med_42, 2),
            "median_mae_21d_pct": round(med_mae, 2),
            "median_mfe_21d_pct": round(med_mfe, 2),
            "analog_score": edge_score,
            "top_matches": top_matches,
        }

    return {
        "backend": "cuda" if use_cuda else (str(dev) if dev else "cpu"),
        "total_historical_patterns": int(len(mat_db)),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "per_symbol": per_symbol,
    }
