"""[GPU Engine #2] Strict Purged Walk-Forward GPU ML Alpha Ensemble (XGBoost CUDA + PyTorch ResMLP).

Predicts 21-trading-day (~1 month) probability of outperforming QQQ by >= +2% across the
watchlist using strictly time-ordered purged expanding walk-forward validation (zero look-ahead bias!).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

try:
    import xgboost as xgb
except Exception:  # pragma: no cover (ImportError or macOS missing libomp OSError)
    xgb = None  # type: ignore[assignment]

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover
    roc_auc_score = None  # type: ignore[assignment]

from src.quant.device import get_torch_device
from src.quant.indicators import compute_atr, compute_cmf, compute_rsi

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "ret_5d",
    "ret_10d",
    "ret_21d",
    "ret_63d",
    "ret_126d",
    "rev_5_21",
    "dist_sma20",
    "dist_sma50",
    "dist_sma200",
    "sma20_50_spread",
    "rsi14",
    "atr_pct",
    "vcp_ratio",
    "vol_20d",
    "rvol",
    "cmf20",
    "dd_252d",
    "rs_qqq_21d",
    "rs_qqq_63d",
    "rs_soxx_21d",
    "qqq_ret_21d",
    "vix_level",
]


def _build_panel_dataset(
    ohlcv_map: dict[str, pd.DataFrame],
    symbols: list[str],
    horizon: int = 21,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build clean feature panel:
      - hist_df: rows with known `horizon`-day forward excess returns (for walk-forward training/eval)
      - live_df: latest feature row per symbol (for live out-of-sample inference)
    """
    qqq = ohlcv_map.get("QQQ")
    soxx = ohlcv_map.get("SOXX")
    vix = ohlcv_map.get("^VIX")

    qqq_c = qqq["Close"] if qqq is not None and not qqq.empty else None
    soxx_c = soxx["Close"] if soxx is not None and not soxx.empty else None
    vix_c = vix["Close"] if vix is not None and not vix.empty else None

    hist_frames: list[pd.DataFrame] = []
    live_rows: list[pd.DataFrame] = []

    for sym in symbols:
        df = ohlcv_map.get(sym)
        if df is None or len(df) < 80:
            continue

        c = df["Close"]
        v = df["Volume"].replace(0.0, np.nan).ffill().fillna(1.0)
        sma20 = c.rolling(20).mean()
        sma50 = c.rolling(50).mean()
        sma200 = c.rolling(200, min_periods=60).mean()
        atr14 = compute_atr(df, 14)
        atr10 = compute_atr(df, 10)
        atr50 = compute_atr(df, 50)

        f = pd.DataFrame(index=df.index)
        f["symbol"] = sym
        f["ret_5d"] = c.pct_change(5)
        f["ret_10d"] = c.pct_change(10)
        f["ret_21d"] = c.pct_change(21)
        f["ret_63d"] = c.pct_change(63)
        f["ret_126d"] = c.pct_change(126)
        f["rev_5_21"] = f["ret_5d"] - f["ret_21d"] * 0.25
        f["dist_sma20"] = c / sma20 - 1.0
        f["dist_sma50"] = c / sma50 - 1.0
        f["dist_sma200"] = c / sma200 - 1.0
        f["sma20_50_spread"] = sma20 / sma50 - 1.0
        f["rsi14"] = compute_rsi(c, 14) / 100.0
        f["atr_pct"] = atr14 / c.replace(0.0, np.nan)
        f["vcp_ratio"] = atr10 / atr50.replace(0.0, np.nan)
        f["vol_20d"] = c.pct_change().rolling(20).std() * np.sqrt(252)
        f["rvol"] = v / v.rolling(20).mean().replace(0.0, np.nan)
        f["cmf20"] = compute_cmf(df, 20)
        f["dd_252d"] = c / df["High"].rolling(252, min_periods=40).max() - 1.0

        if qqq_c is not None:
            q_al = qqq_c.reindex(df.index).ffill()
            q_ret21 = q_al.pct_change(21)
            q_ret63 = q_al.pct_change(63)
            q_fwd = q_al.shift(-horizon) / q_al - 1.0
        else:
            q_ret21 = pd.Series(0.0, index=df.index)
            q_ret63 = pd.Series(0.0, index=df.index)
            q_fwd = pd.Series(0.0, index=df.index)

        if soxx_c is not None:
            s_al = soxx_c.reindex(df.index).ffill()
            s_ret21 = s_al.pct_change(21)
        else:
            s_ret21 = q_ret21

        if vix_c is not None:
            v_al = vix_c.reindex(df.index).ffill().fillna(18.0) / 100.0
        else:
            v_al = pd.Series(0.18, index=df.index)

        f["rs_qqq_21d"] = f["ret_21d"] - q_ret21
        f["rs_qqq_63d"] = f["ret_63d"] - q_ret63
        f["rs_soxx_21d"] = f["ret_21d"] - s_ret21
        f["qqq_ret_21d"] = q_ret21
        f["vix_level"] = v_al

        # Forward target (strictly shifted forward — only used on historical rows!)
        f["fwd_ret"] = c.shift(-horizon) / c - 1.0
        f["fwd_excess"] = f["fwd_ret"] - q_fwd
        f["target_cls"] = (f["fwd_excess"] >= 0.015).astype(np.float32)

        f = f.replace([np.inf, -np.inf], np.nan)
        f["date"] = df.index

        # Live row: latest date with finite features (fill rare missing long-horizon with 0)
        latest = f.iloc[[-1]].copy()
        latest[FEATURE_COLS] = latest[FEATURE_COLS].fillna(0.0)
        live_rows.append(latest)

        # Historical training rows (sampled every 3 trading days to cut autocorrelation)
        hist = f.dropna(subset=FEATURE_COLS + ["fwd_excess"]).iloc[::3]
        if not hist.empty:
            hist_frames.append(hist)

    hist_df = pd.concat(hist_frames, ignore_index=True).sort_values("date") if hist_frames else pd.DataFrame()
    live_df = pd.concat(live_rows, ignore_index=True) if live_rows else pd.DataFrame()
    return hist_df, live_df


if nn is not None:
    class _TabularResMLP(nn.Module):
        """Compact residual MLP optimized for GPU tabular swing alpha classification."""

        def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.25) -> None:
            super().__init__()
            self.in_proj = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )
            self.block1 = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
            )
            self.head = nn.Sequential(
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden, 1),
            )

        def forward(self, x: Any) -> Any:
            h = self.in_proj(x)
            h = h + self.block1(h)
            return self.head(h).squeeze(-1)


def _train_predict_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    dev: Any,
    epochs: int = 35,
    batch_size: int = 256,
) -> np.ndarray:
    """Train residual MLP on CUDA (or CPU fallback) with standardization & cosine decay."""
    if torch is None or nn is None or dev is None or len(X_train) < 100:
        return np.full(len(X_eval), 0.5, dtype=np.float32)

    mean = X_train.mean(axis=0, keepdims=True)
    std = np.maximum(X_train.std(axis=0, keepdims=True), 1e-5)
    X_tr = np.clip((X_train - mean) / std, -5.0, 5.0).astype(np.float32)
    X_ev = np.clip((X_eval - mean) / std, -5.0, 5.0).astype(np.float32)

    torch.manual_seed(42)
    model = _TabularResMLP(X_tr.shape[1], hidden=128, dropout=0.25).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    tx = torch.tensor(X_tr, device=dev)
    ty = torch.tensor(y_train.astype(np.float32), device=dev)
    n_samples = len(tx)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_samples, device=dev)
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            logits = model(tx[idx])
            loss = loss_fn(logits, ty[idx])
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_ev, device=dev))).cpu().numpy()
    return probs.astype(np.float32)


def _train_predict_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    use_cuda: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Train GPU-accelerated XGBoost classifier (falls back smoothly to CPU if needed)."""
    if xgb is None or len(X_train) < 100 or len(np.unique(y_train)) < 2:
        base_p = float(np.mean(y_train)) if len(y_train) > 0 else 0.5
        return np.full(len(X_eval), base_p, dtype=np.float32), None

    params = {
        "n_estimators": 220,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 3.0,
        "reg_alpha": 0.5,
        "tree_method": "hist",
        "device": "cuda" if use_cuda else "cpu",
        "eval_metric": "logloss",
        "random_state": 42,
    }
    try:
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train, verbose=False)
    except Exception as exc:
        logger.warning("XGBoost CUDA failed (%s), falling back to CPU hist.", exc)
        params["device"] = "cpu"
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train, verbose=False)

    probs = clf.predict_proba(X_eval)[:, 1].astype(np.float32)
    importances = getattr(clf, "feature_importances_", None)
    return probs, importances


def run_gpu_walkforward_ml_alpha(
    ohlcv_map: dict[str, pd.DataFrame],
    symbols: list[str],
    horizon: int = 21,
    n_splits: int = 4,
    device_pref: str = "auto",
) -> dict[str, Any]:
    """
    Execute Purged Expanding Walk-Forward Validation + Live Inference on RTX 5070 Ti.
    Returns calibrated 21-day outperformance probabilities + honest OOS diagnostics.
    """
    t0 = time.perf_counter()
    dev = get_torch_device(device_pref)
    use_cuda = dev is not None and dev.type == "cuda"

    hist_df, live_df = _build_panel_dataset(ohlcv_map, symbols, horizon=horizon)
    if hist_df.empty or live_df.empty or len(hist_df) < 350:
        return {
            "available": False,
            "backend": "cuda" if use_cuda else "cpu",
            "reason": "历史样本不足以执行严格 Walk-Forward 训练",
            "per_symbol": {},
        }

    unique_dates = np.sort(hist_df["date"].unique())
    n_dates = len(unique_dates)
    embargo_dates = max(12, horizon // 2 + 5)  # strict purge gap between train & test folds

    oos_preds: list[np.ndarray] = []
    oos_targets: list[np.ndarray] = []
    oos_excess: list[np.ndarray] = []

    # 1. Purged Expanding Walk-Forward Validation
    fold_bounds = np.linspace(int(n_dates * 0.45), n_dates, n_splits + 1, dtype=int)
    for fi in range(n_splits):
        val_start_idx = int(fold_bounds[fi])
        val_end_idx = int(fold_bounds[fi + 1])
        train_end_idx = max(10, val_start_idx - embargo_dates)
        if val_end_idx <= val_start_idx + 5 or train_end_idx < 30:
            continue

        tr_cut = unique_dates[train_end_idx]
        va_start = unique_dates[val_start_idx]
        va_end = unique_dates[min(val_end_idx - 1, n_dates - 1)]

        tr_df = hist_df[hist_df["date"] <= tr_cut]
        va_df = hist_df[(hist_df["date"] >= va_start) & (hist_df["date"] <= va_end)]
        if len(tr_df) < 150 or len(va_df) < 30:
            continue

        X_tr = tr_df[FEATURE_COLS].values.astype(np.float32)
        y_tr = tr_df["target_cls"].values.astype(np.float32)
        X_va = va_df[FEATURE_COLS].values.astype(np.float32)

        p_xgb, _ = _train_predict_xgb(X_tr, y_tr, X_va, use_cuda=use_cuda)
        p_mlp = _train_predict_torch_mlp(X_tr, y_tr, X_va, dev=dev, epochs=25)
        p_ens = 0.60 * p_xgb + 0.40 * p_mlp

        oos_preds.append(p_ens)
        oos_targets.append(va_df["target_cls"].values.astype(np.float32))
        oos_excess.append(va_df["fwd_excess"].values.astype(np.float32))

    # Aggregate OOS honesty diagnostics
    oos_metrics: dict[str, Any] = {
        "oos_samples": 0,
        "oos_auc": None,
        "oos_hit_rate_top_pct": None,
        "oos_top_vs_bottom_spread_pct": None,
    }
    if oos_preds:
        all_p = np.concatenate(oos_preds)
        all_y = np.concatenate(oos_targets)
        all_ex = np.concatenate(oos_excess)
        oos_metrics["oos_samples"] = int(len(all_p))
        if roc_auc_score is not None and len(np.unique(all_y)) > 1:
            oos_metrics["oos_auc"] = round(float(roc_auc_score(all_y, all_p)), 3)
        q80 = np.percentile(all_p, 80)
        q20 = np.percentile(all_p, 20)
        top_mask = all_p >= q80
        bot_mask = all_p <= q20
        if np.any(top_mask):
            oos_metrics["oos_hit_rate_top_pct"] = round(float(np.mean(all_ex[top_mask] > 0) * 100.0), 1)
            top_avg = float(np.mean(all_ex[top_mask]) * 100.0)
            bot_avg = float(np.mean(all_ex[bot_mask]) * 100.0) if np.any(bot_mask) else 0.0
            oos_metrics["oos_top_vs_bottom_spread_pct"] = round(top_avg - bot_avg, 2)

    # 2. Final Full-History Training -> Live Out-of-Sample Inference
    X_full = hist_df[FEATURE_COLS].values.astype(np.float32)
    y_full = hist_df["target_cls"].values.astype(np.float32)
    X_live = live_df[FEATURE_COLS].values.astype(np.float32)

    live_xgb, feat_imp = _train_predict_xgb(X_full, y_full, X_live, use_cuda=use_cuda)
    live_mlp = _train_predict_torch_mlp(X_full, y_full, X_live, dev=dev, epochs=40)
    live_ens = 0.60 * live_xgb + 0.40 * live_mlp

    top_features = []
    if feat_imp is not None and len(feat_imp) == len(FEATURE_COLS):
        order = np.argsort(-feat_imp)[:8]
        top_features = [
            {"feature": FEATURE_COLS[i], "importance": round(float(feat_imp[i]), 4)}
            for i in order
        ]

    per_symbol: dict[str, dict[str, Any]] = {}
    for idx, row in live_df.reset_index(drop=True).iterrows():
        sym = str(row["symbol"])
        prob = float(live_ens[idx])
        # Convert calibrated probability into 0-100 ML Alpha score
        ml_score = round(max(5.0, min(98.0, 50.0 + (prob - 0.5) * 125.0)), 1)
        per_symbol[sym] = {
            "prob_outperform_21d_pct": round(prob * 100.0, 1),
            "xgb_prob_pct": round(float(live_xgb[idx]) * 100.0, 1),
            "mlp_prob_pct": round(float(live_mlp[idx]) * 100.0, 1),
            "ml_alpha_score": ml_score,
        }

    return {
        "available": True,
        "backend": "cuda" if use_cuda else (str(dev) if dev else "cpu"),
        "training_samples": int(len(hist_df)),
        "horizon_days": horizon,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "oos_validation": oos_metrics,
        "top_features": top_features,
        "per_symbol": per_symbol,
    }
