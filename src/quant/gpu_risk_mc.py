"""[GPU Engine #3] PyTorch CUDA 100,000-Path Student-t Correlated Monte Carlo Risk & Stress Engine.

Quantifies fat-tailed portfolio risk (VaR / CVaR), Marginal Risk Contribution (MCR) per holding,
pairwise correlation exposure, and institutional macro/AI stress scenarios on RTX 5070 Ti CUDA.
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

from src.data.market_data import build_aligned_close_matrix
from src.quant.device import get_torch_device


def _shrink_covariance(cov: np.ndarray, shrink: float = 0.12) -> np.ndarray:
    """Apply diagonal shrinkage so Cholesky factorization is guaranteed well-conditioned."""
    diag = np.diag(np.diag(cov))
    shrunk = (1.0 - shrink) * cov + shrink * diag
    # Tiny jitter floor for numerical safety
    eps = max(float(np.mean(np.diag(shrunk))) * 1e-5, 1e-8)
    return shrunk + np.eye(shrunk.shape[0]) * eps


def run_gpu_portfolio_risk_mc(
    ohlcv_map: dict[str, pd.DataFrame],
    portfolio_cfg: dict[str, Any],
    universe_cfg: dict[str, Any],
    n_sims: int = 100_000,
    horizon_days: int = 21,
    df_t: float = 5.0,
    device_pref: str = "auto",
) -> dict[str, Any]:
    """
    Run correlated heavy-tailed (Student-t, df=5) Monte Carlo simulations on RTX 5070 Ti CUDA
    for the user's active holdings + cash reserve.
    """
    t0 = time.perf_counter()
    dev = get_torch_device(device_pref)
    use_cuda = torch is not None and dev is not None and dev.type == "cuda"

    account = portfolio_cfg.get("account") or {}
    holdings = portfolio_cfg.get("holdings") or []
    cash_usd = float(account.get("cash_usd", 0.0))

    # Build current marked-to-market position values
    pos_values: dict[str, float] = {}
    valid_holdings: list[dict[str, Any]] = []
    for h in holdings:
        sym = str(h.get("symbol", "")).upper().strip()
        shares = float(h.get("shares", 0.0))
        if not sym or shares <= 0:
            continue
        df = ohlcv_map.get(sym)
        if df is None or df.empty:
            continue
        last_px = float(df["Close"].iloc[-1])
        val = shares * last_px
        pos_values[sym] = val
        valid_holdings.append({**h, "symbol": sym, "last_price": last_px, "market_value": val})

    equity_val = sum(pos_values.values())
    total_nav = equity_val + cash_usd
    if equity_val <= 0 or total_nav <= 0:
        return {"available": False, "reason": "暂无有效持仓或最新行情"}

    syms = list(pos_values.keys())
    close_mat = build_aligned_close_matrix(ohlcv_map, syms, min_obs=25)
    # Only keep symbols with enough joint history for covariance estimation
    usable_syms = [s for s in syms if s in close_mat.columns]
    if not usable_syms:
        return {"available": False, "reason": "持仓历史重叠天数不足"}

    rets = close_mat[usable_syms].pct_change().dropna().tail(252)
    if len(rets) < 25:
        return {"available": False, "reason": "持仓收益率重叠样本少于25个交易日"}

    # Weights relative to total NAV (including cash buffer)
    weights_nav = np.array([pos_values[s] / total_nav for s in usable_syms], dtype=np.float64)
    weights_eq = np.array([pos_values[s] / max(equity_val, 1e-6) for s in usable_syms], dtype=np.float64)

    mean_daily = rets.mean().values.astype(np.float64)
    # Shrink daily drift heavily toward conservative 0 so risk estimates are never masked by lucky past rallies
    mu_daily = mean_daily * 0.25
    cov_daily = _shrink_covariance(rets.cov().values.astype(np.float64), shrink=0.10)
    corr_mat = rets.corr().fillna(0.0)

    # Analytical Portfolio Volatility & Marginal Risk Contribution (Euler decomposition)
    port_var_daily = float(weights_nav @ cov_daily @ weights_nav)
    port_vol_ann = float(np.sqrt(max(port_var_daily, 0.0) * 252.0) * 100.0)
    mrc = (cov_daily @ weights_nav) / max(np.sqrt(port_var_daily), 1e-8)
    rc_share = (weights_nav * mrc) / max(np.sqrt(port_var_daily), 1e-8) * 100.0

    # Diversification Ratio = weighted avg individual vol / portfolio vol
    indiv_vols = np.sqrt(np.diag(cov_daily))
    weighted_avg_vol = float(weights_nav @ indiv_vols)
    div_ratio = weighted_avg_vol / max(np.sqrt(port_var_daily), 1e-8)

    # GPU Heavy-Tailed Student-t Correlated Path Simulation
    # Scale covariance so Student-t(df) preserves exact empirical variance: scale = sqrt((df - 2) / df)
    chol = np.linalg.cholesky(cov_daily).astype(np.float32)
    t_scale = float(np.sqrt((df_t - 2.0) / df_t))
    n_assets = len(usable_syms)

    # Cap CPU sim count slightly so Mac CPU stays snappy while 5070 Ti CUDA runs full 100,000+ paths
    sims_eff = int(n_sims) if use_cuda else min(int(n_sims), 30_000)

    def _simulate_horizon(h_days: int) -> np.ndarray:
        if torch is not None and dev is not None:
            with torch.no_grad():
                L = torch.tensor(chol * t_scale, dtype=torch.float32, device=dev)
                mu_t = torch.tensor(mu_daily, dtype=torch.float32, device=dev)
                w_t = torch.tensor(weights_nav, dtype=torch.float32, device=dev)
                chi2 = torch.distributions.Chi2(torch.tensor(df_t, device=dev))

                # Stream in chunks if huge to keep VRAM footprint tiny (< 300MB)
                chunk = 25_000
                out_chunks: list[np.ndarray] = []
                for start in range(0, sims_eff, chunk):
                    bsz = min(chunk, sims_eff - start)
                    z = torch.randn((bsz, h_days, n_assets), device=dev, dtype=torch.float32)
                    g = chi2.sample((bsz, h_days, 1))
                    t_mult = torch.rsqrt(g / df_t)
                    # Correlated heavy-tailed daily returns: [bsz, h_days, n_assets]
                    daily_r = torch.matmul(z, L.T) * t_mult + mu_t
                    daily_r = torch.clamp(daily_r, min=-0.45, max=0.60)
                    # Compound asset prices over horizon -> portfolio return with cash = 0% drift
                    cum_asset = torch.prod(1.0 + daily_r, dim=1) - 1.0  # [bsz, n_assets]
                    port_ret = torch.matmul(cum_asset, w_t)
                    out_chunks.append(port_ret.cpu().numpy())
                return np.concatenate(out_chunks)
        else:
            rng = np.random.default_rng(42)
            z = rng.standard_normal((sims_eff, h_days, n_assets)).astype(np.float32)
            g = rng.chisquare(df_t, size=(sims_eff, h_days, 1)).astype(np.float32)
            daily_r = (z @ (chol * t_scale).T) / np.sqrt(g / df_t) + mu_daily.astype(np.float32)
            daily_r = np.clip(daily_r, -0.45, 0.60)
            cum_asset = np.prod(1.0 + daily_r, axis=1) - 1.0
            return (cum_asset @ weights_nav.astype(np.float32)).astype(np.float32)

    sim_21 = _simulate_horizon(horizon_days)
    sim_42 = _simulate_horizon(horizon_days * 2)

    def _risk_stats(arr: np.ndarray) -> dict[str, Any]:
        var95 = float(np.percentile(arr, 5))
        var99 = float(np.percentile(arr, 1))
        cvar95 = float(np.mean(arr[arr <= var95])) if np.any(arr <= var95) else var95
        med = float(np.median(arr))
        p90 = float(np.percentile(arr, 90))
        return {
            "median_return_pct": round(med * 100.0, 2),
            "upside_p90_pct": round(p90 * 100.0, 2),
            "var_95_pct": round(var95 * 100.0, 2),
            "var_95_usd": round(var95 * total_nav, 0),
            "cvar_95_pct": round(cvar95 * 100.0, 2),
            "cvar_95_usd": round(cvar95 * total_nav, 0),
            "var_99_pct": round(var99 * 100.0, 2),
            "var_99_usd": round(var99 * total_nav, 0),
            "prob_loss_gt_10pct": round(float(np.mean(arr <= -0.10) * 100.0), 1),
            "prob_loss_gt_20pct": round(float(np.mean(arr <= -0.20) * 100.0), 1),
            "prob_gain_gt_10pct": round(float(np.mean(arr >= 0.10) * 100.0), 1),
        }

    # Stress Tests using empirical Betas to SOXX / QQQ + sector shocks
    stress_tests = _compute_stress_scenarios(
        ohlcv_map=ohlcv_map,
        usable_syms=usable_syms,
        pos_values=pos_values,
        total_nav=total_nav,
        universe_cfg=universe_cfg,
    )

    # Sector exposure breakdown
    u_meta = universe_cfg.get("universe") or {}
    s_meta = universe_cfg.get("sectors") or {}
    sector_weights: dict[str, float] = {}
    for sym, val in pos_values.items():
        sk = str((u_meta.get(sym) or {}).get("sector", "other"))
        sl = str((s_meta.get(sk) or {}).get("label", sk))
        sector_weights[sl] = round(sector_weights.get(sl, 0.0) + val / total_nav * 100.0, 1)
    if cash_usd > 0:
        sector_weights["现金储备 (Cash)"] = round(cash_usd / total_nav * 100.0, 1)

    risk_breakdown = []
    for i, sym in enumerate(usable_syms):
        risk_breakdown.append(
            {
                "symbol": sym,
                "market_value_usd": round(pos_values[sym], 0),
                "weight_nav_pct": round(weights_nav[i] * 100.0, 1),
                "weight_equity_pct": round(weights_eq[i] * 100.0, 1),
                "indiv_vol_ann_pct": round(float(indiv_vols[i] * np.sqrt(252) * 100.0), 1),
                "risk_contribution_pct": round(float(rc_share[i]), 1),
            }
        )
    risk_breakdown.sort(key=lambda x: x["risk_contribution_pct"], reverse=True)

    # Compact histogram samples (for Plotly chart without storing 100k floats in SQLite)
    hist_counts, hist_edges = np.histogram(sim_21 * 100.0, bins=60, range=(-35.0, 40.0))

    return {
        "available": True,
        "backend": "cuda" if use_cuda else (str(dev) if dev else "cpu"),
        "n_simulations": sims_eff,
        "student_t_df": df_t,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "total_nav_usd": round(total_nav, 2),
        "equity_value_usd": round(equity_val, 2),
        "cash_usd": round(cash_usd, 2),
        "gross_exposure_pct": round(equity_val / total_nav * 100.0, 1),
        "portfolio_vol_ann_pct": round(port_vol_ann, 1),
        "diversification_ratio": round(div_ratio, 2),
        "horizon_21d": _risk_stats(sim_21),
        "horizon_42d": _risk_stats(sim_42),
        "risk_breakdown": risk_breakdown,
        "sector_weights": sector_weights,
        "correlation_matrix": {
            "symbols": usable_syms,
            "values": [[round(float(corr_mat.loc[r, c]), 2) for c in usable_syms] for r in usable_syms],
        },
        "stress_tests": stress_tests,
        "dist_hist_21d": {
            "counts": [int(x) for x in hist_counts],
            "bin_centers": [round(float((hist_edges[i] + hist_edges[i + 1]) / 2.0), 2) for i in range(len(hist_counts))],
        },
    }


def _compute_stress_scenarios(
    *,
    ohlcv_map: dict[str, pd.DataFrame],
    usable_syms: list[str],
    pos_values: dict[str, float],
    total_nav: float,
    universe_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Estimate portfolio drawdown under 4 realistic AI/Semiconductor & Macro shocks."""
    u_meta = universe_cfg.get("universe") or {}
    soxx_ret = ohlcv_map["SOXX"]["Close"].pct_change().dropna() if "SOXX" in ohlcv_map else None
    qqq_ret = ohlcv_map["QQQ"]["Close"].pct_change().dropna() if "QQQ" in ohlcv_map else None

    beta_soxx: dict[str, float] = {}
    beta_qqq: dict[str, float] = {}
    for sym in usable_syms:
        s_ret = ohlcv_map[sym]["Close"].pct_change().dropna()
        beta_soxx[sym] = _estimate_beta(s_ret, soxx_ret, default=0.8)
        beta_qqq[sym] = _estimate_beta(s_ret, qqq_ret, default=1.0)

    scenarios = [
        ("AI / 半导体板块回调 (SOXX -15%)", "AI / Semiconductor Pullback (SOXX -15%)", "soxx", -0.15),
        ("纳斯达克科技杀估值 (QQQ -12%)", "Nasdaq Tech Valuation Compression (QQQ -12%)", "qqq", -0.12),
        (
            "存储/硬件周期急跌 (存储 -22%, 算力 -12%, 云 -6%, 防御 +1%)",
            "Memory / Storage Cycle Shock (Storage -22%, Compute -12%, Cloud -6%, Defensive +1%)",
            "sector_storage",
            -0.22,
        ),
        ("极端系统性科技熊市 (QQQ -25% 冲击)", "Severe Systemic Tech Bear Shock (QQQ -25%)", "qqq", -0.25),
    ]

    results = []
    for name, name_en, mode, shock in scenarios:
        total_pnl = 0.0
        worst_sym = ""
        worst_loss = 0.0
        for sym in usable_syms:
            val = pos_values[sym]
            sec = str((u_meta.get(sym) or {}).get("sector", "other"))
            if mode == "soxx":
                pct = max(-0.55, min(0.15, beta_soxx[sym] * shock))
            elif mode == "qqq":
                pct = max(-0.55, min(0.15, beta_qqq[sym] * shock))
            else:
                if sec == "memory_storage":
                    pct = -0.22
                elif sec == "compute_optical":
                    pct = -0.12
                elif sec == "cloud_ai":
                    pct = -0.06
                elif sec == "defensive":
                    pct = 0.01
                else:
                    pct = -0.10
            loss = val * pct
            total_pnl += loss
            if loss < worst_loss:
                worst_loss = loss
                worst_sym = sym

        results.append(
            {
                "scenario": name,
                "scenario_en": name_en,
                "est_portfolio_impact_pct": round(total_pnl / total_nav * 100.0, 2),
                "est_portfolio_impact_usd": round(total_pnl, 0),
                "largest_drag_symbol": worst_sym,
                "largest_drag_usd": round(worst_loss, 0),
            }
        )
    return results


def _estimate_beta(stock_r: pd.Series, bench_r: pd.Series | None, default: float = 1.0) -> float:
    if bench_r is None or bench_r.empty or stock_r.empty:
        return default
    al = pd.concat([stock_r.rename("s"), bench_r.rename("b")], axis=1).dropna().tail(252)
    if len(al) < 30:
        return default
    var_b = float(al["b"].var())
    if var_b <= 1e-10:
        return default
    beta = float(al["s"].cov(al["b"]) / var_b)
    return float(np.clip(beta, -0.3, 2.5))
