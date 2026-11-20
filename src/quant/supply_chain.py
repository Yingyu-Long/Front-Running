"""Supply-Chain Lead-Lag Analysis, Sector Peer Rotation & Pair Spread Z-Score Engine."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from src.quant.device import get_torch_device


def compute_supply_chain_lead_lag(
    ohlcv_map: dict[str, pd.DataFrame],
    edges: list[dict[str, Any]],
    max_lag: int = 10,
    lookback_days: int = 252,
    device_pref: str = "auto",
) -> list[dict[str, Any]]:
    """
    Use GPU-accelerated cross-correlation to quantify whether upstream bellwether
    returns lead downstream stock returns (lags 1..10 trading days), and check
    whether an active upstream move is currently pending transmission!
    """
    dev = get_torch_device(device_pref)
    results: list[dict[str, Any]] = []

    for edge in edges:
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        logic = str(edge.get("logic", ""))
        df_s = ohlcv_map.get(src)
        df_t = ohlcv_map.get(tgt)
        if df_s is None or df_t is None or len(df_s) < 60 or len(df_t) < 60:
            continue

        aligned = pd.concat(
            [df_s["Close"].pct_change().rename("s"), df_t["Close"].pct_change().rename("t")],
            axis=1,
        ).dropna().tail(lookback_days)
        if len(aligned) < 50:
            continue

        s_np = aligned["s"].values.astype(np.float32)
        t_np = aligned["t"].values.astype(np.float32)

        # Compute contemporaneous corr + lagged correlations (src at t-lag -> tgt at t)
        lags = list(range(0, max_lag + 1))
        corrs: list[float] = []
        if torch is not None and dev is not None and dev.type == "cuda":
            s_t = torch.tensor(s_np, device=dev)
            t_t = torch.tensor(t_np, device=dev)
            for lag in lags:
                xs = s_t[:-lag] if lag > 0 else s_t
                yt = t_t[lag:] if lag > 0 else t_t
                xs_c = xs - xs.mean()
                yt_c = yt - yt.mean()
                denom = (xs_c.norm() * yt_c.norm()).clamp(min=1e-8)
                corrs.append(float((xs_c @ yt_c / denom).item()))
        else:
            for lag in lags:
                xs = s_np[:-lag] if lag > 0 else s_np
                yt = t_np[lag:] if lag > 0 else t_np
                c = float(np.corrcoef(xs, yt)[0, 1]) if len(xs) > 5 else 0.0
                corrs.append(0.0 if not np.isfinite(c) else c)

        same_day_corr = corrs[0]
        # Best predictive lag (1..max_lag)
        lagged_corrs = corrs[1:]
        best_idx = int(np.argmax(np.abs(lagged_corrs)))
        best_lag = best_idx + 1
        best_lag_corr = lagged_corrs[best_idx]

        # Recent 10-day return divergence (upstream moved vs downstream lagging?)
        src_10d = float((df_s["Close"].iloc[-1] / df_s["Close"].iloc[-11] - 1.0) * 100.0) if len(df_s) > 11 else 0.0
        tgt_10d = float((df_t["Close"].iloc[-1] / df_t["Close"].iloc[-11] - 1.0) * 100.0) if len(df_t) > 11 else 0.0
        spread_10d = src_10d - tgt_10d

        signal_note = "同步波动"
        if same_day_corr >= 0.45 and spread_10d >= 6.0:
            signal_note = f"🟢 上游 {src} 近10日领先 +{spread_10d:.1f}%，留意 {tgt} 滞后补涨机会"
        elif same_day_corr >= 0.45 and spread_10d <= -6.0:
            signal_note = f"🟠 上游 {src} 近10日明显走弱({spread_10d:.1f}%)，警惕 {tgt} 补跌风险"
        elif abs(best_lag_corr) >= 0.18:
            signal_note = f"📡 存在 {best_lag} 日显著滞后传导效应 (r={best_lag_corr:+.2f})"

        results.append(
            {
                "source": src,
                "target": tgt,
                "logic": logic,
                "same_day_corr": round(same_day_corr, 3),
                "best_lag_days": best_lag,
                "best_lag_corr": round(best_lag_corr, 3),
                "source_10d_pct": round(src_10d, 2),
                "target_10d_pct": round(tgt_10d, 2),
                "divergence_10d_pct": round(spread_10d, 2),
                "signal_note": signal_note,
            }
        )

    return results


def compute_pair_rotation_spreads(
    ohlcv_map: dict[str, pd.DataFrame],
    rotation_pairs: list[dict[str, Any]],
    z_window: int = 60,
) -> list[dict[str, Any]]:
    """
    Compute log-ratio spread Z-Score & 21D momentum spread for intra-sector pairs
    (e.g., STX vs WDC, MU vs SKHY, GOOG vs AMZN, NVDA vs AMD, SOXX vs MCD).
    """
    out: list[dict[str, Any]] = []
    for item in rotation_pairs:
        pair = item.get("pair") or []
        if len(pair) != 2:
            continue
        a, b = str(pair[0]), str(pair[1])
        df_a, df_b = ohlcv_map.get(a), ohlcv_map.get(b)
        if df_a is None or df_b is None or len(df_a) < 35 or len(df_b) < 35:
            continue

        aligned = pd.concat(
            [df_a["Close"].rename("a"), df_b["Close"].rename("b")], axis=1
        ).dropna()
        if len(aligned) < 35:
            continue

        log_ratio = np.log(aligned["a"] / aligned["b"])
        win = min(z_window, len(log_ratio))
        mean_r = float(log_ratio.tail(win).mean())
        std_r = float(log_ratio.tail(win).std())
        curr_r = float(log_ratio.iloc[-1])
        zscore = (curr_r - mean_r) / std_r if std_r > 1e-8 else 0.0

        ret21_a = float((aligned["a"].iloc[-1] / aligned["a"].iloc[-22] - 1.0) * 100.0) if len(aligned) > 22 else 0.0
        ret21_b = float((aligned["b"].iloc[-1] / aligned["b"].iloc[-22] - 1.0) * 100.0) if len(aligned) > 22 else 0.0
        corr_60 = float(
            aligned["a"].pct_change().tail(win).corr(aligned["b"].pct_change().tail(win))
        )

        if zscore >= 1.8:
            interp = f"⚖️ {a} 相对 {b} 短期极端偏贵 (Z={zscore:+.2f})：适合止盈部分 {a} 或换仓关注 {b}"
        elif zscore <= -1.8:
            interp = f"⚖️ {a} 相对 {b} 短期极端超跌 (Z={zscore:+.2f})：若基本面未破，{a} 均值回归赔率更高"
        elif ret21_a > ret21_b + 5.0:
            interp = f"📈 动量占优：{a} 近1月强势领先 {b} ({ret21_a - ret21_b:+.1f}%)"
        elif ret21_b > ret21_a + 5.0:
            interp = f"📈 动量占优：{b} 近1月强势领先 {a} ({ret21_b - ret21_a:+.1f}%)"
        else:
            interp = "均衡区间，无极端价差偏离"

        out.append(
            {
                "pair": f"{a} / {b}",
                "left": a,
                "right": b,
                "label": str(item.get("label", f"{a} vs {b}")),
                "zscore_60d": round(zscore, 2),
                "corr_60d": round(0.0 if not np.isfinite(corr_60) else corr_60, 3),
                "left_21d_pct": round(ret21_a, 2),
                "right_21d_pct": round(ret21_b, 2),
                "interpretation": interp,
            }
        )
    return out


def rank_within_sectors(
    stock_scores: dict[str, dict[str, Any]],
    universe_cfg: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Group and rank all stocks inside each supply-chain sector for '优中选优' rotation."""
    sectors_meta = universe_cfg.get("sectors") or {}
    u_meta = universe_cfg.get("universe") or {}
    grouped: dict[str, list[dict[str, Any]]] = {}

    for sym, data in stock_scores.items():
        if data.get("error"):
            continue
        sec_key = str((u_meta.get(sym) or {}).get("sector", "other"))
        sec_label = str((sectors_meta.get(sec_key) or {}).get("label", sec_key))
        grouped.setdefault(sec_label, []).append(
            {
                "symbol": sym,
                "name": str((u_meta.get(sym) or {}).get("name", sym)),
                "role": str((u_meta.get(sym) or {}).get("role", "watchlist")),
                "composite_score": data.get("composite_score", 50.0),
                "close": data.get("close"),
                "chg_1d_pct": data.get("chg_1d_pct"),
                "ret_21d_pct": data.get("ret_21d_pct"),
                "rs_sector_21d_pct": data.get("rs_sector_21d_pct"),
                "stage_label": data.get("stage_label"),
                "crash_badge": (data.get("crash_analysis") or {}).get("badge", ""),
                "earnings_flag": (data.get("fundamentals") or {}).get("earnings_risk_flag", ""),
            }
        )

    for sec_label in grouped:
        grouped[sec_label].sort(key=lambda x: float(x.get("composite_score", 0.0)), reverse=True)
        for rank_idx, row in enumerate(grouped[sec_label], start=1):
            row["sector_rank"] = rank_idx
    return grouped
