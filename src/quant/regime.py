"""Macro & Sector Market Regime Engine (大盘与半导体温度计 / 风险仓位开关)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _sma(series: pd.Series, window: int) -> float | None:
    if series is None or len(series) < window:
        return None
    val = float(series.rolling(window).mean().iloc[-1])
    return val if np.isfinite(val) else None


def _pct_change_n(series: pd.Series, n: int) -> float | None:
    if series is None or len(series) <= n:
        return None
    prev = float(series.iloc[-1 - n])
    curr = float(series.iloc[-1])
    if prev <= 0 or not np.isfinite(prev) or not np.isfinite(curr):
        return None
    return (curr / prev - 1.0) * 100.0


def evaluate_market_regime(
    ohlcv_map: dict[str, pd.DataFrame],
    equity_symbols: list[str],
) -> dict[str, Any]:
    """
    Compute multi-pillar Macro & Tech/Semiconductor Regime Score (0-100)
    and translate it directly into an actionable Max Equity Exposure & Cash/MCD guide.
    """
    score = 50.0
    signals: list[str] = []
    metrics: dict[str, Any] = {}

    # 1. SPY & QQQ Trend Structure (+/- 20 pts)
    for idx_sym, weight in (("SPY", 10.0), ("QQQ", 12.0), ("SOXX", 14.0)):
        df = ohlcv_map.get(idx_sym)
        if df is None or len(df) < 50:
            continue
        close = df["Close"]
        last = float(close.iloc[-1])
        sma20 = _sma(close, 20)
        sma50 = _sma(close, 50)
        sma200 = _sma(close, 200)
        ret20 = _pct_change_n(close, 20)
        ret5 = _pct_change_n(close, 5)

        metrics[idx_sym] = {
            "close": round(last, 2),
            "ret_5d_pct": round(ret5, 2) if ret5 is not None else None,
            "ret_20d_pct": round(ret20, 2) if ret20 is not None else None,
            "above_sma20": bool(sma20 and last > sma20),
            "above_sma50": bool(sma50 and last > sma50),
            "above_sma200": bool(sma200 and last > sma200),
            "dist_sma50_pct": round((last / sma50 - 1.0) * 100.0, 2) if sma50 else None,
        }

        sub = 0.0
        if sma20 and last > sma20:
            sub += 0.35
        else:
            sub -= 0.30
        if sma50 and last > sma50:
            sub += 0.40
        else:
            sub -= 0.40
        if sma200:
            sub += 0.25 if last > sma200 else -0.35
        if sma20 and sma50 and sma20 > sma50:
            sub += 0.15
        score += sub * weight

    # 2. Semiconductor Leadership (SOXX / SPY relative strength, +/- 8 pts)
    if "SOXX" in ohlcv_map and "SPY" in ohlcv_map:
        soxx_c = ohlcv_map["SOXX"]["Close"]
        spy_c = ohlcv_map["SPY"]["Close"]
        aligned = pd.concat([soxx_c.rename("soxx"), spy_c.rename("spy")], axis=1).dropna()
        if len(aligned) >= 50:
            ratio = aligned["soxx"] / aligned["spy"]
            ratio_ret20 = _pct_change_n(ratio, 20)
            ratio_sma50 = _sma(ratio, 50)
            curr_ratio = float(ratio.iloc[-1])
            metrics["soxx_vs_spy_20d_pct"] = round(ratio_ret20, 2) if ratio_ret20 is not None else None
            if ratio_sma50 and curr_ratio > ratio_sma50 and (ratio_ret20 or 0) > 0:
                score += 7.0
                signals.append("🟢 半导体相对大盘走强 (SOXX/SPY > 50日线)，科技主线健康")
            elif ratio_sma50 and curr_ratio < ratio_sma50 and (ratio_ret20 or 0) < -3.0:
                score -= 8.0
                signals.append("🔴 半导体明显跑输大盘 (SOXX/SPY 弱势)，AI/存储板块处于逆风期")
            else:
                signals.append("🟡 半导体相对大盘震荡分化，重个股精选而非盲目全仓")

    # 3. Volatility Regime (^VIX, +/- 12 pts)
    vix_df = ohlcv_map.get("^VIX")
    if vix_df is not None and len(vix_df) >= 20:
        vix_last = float(vix_df["Close"].iloc[-1])
        vix_sma20 = _sma(vix_df["Close"], 20) or vix_last
        metrics["VIX"] = round(vix_last, 2)
        metrics["VIX_vs_sma20"] = round(vix_last - vix_sma20, 2)
        if vix_last < 16.0:
            score += 9.0
            signals.append(f"🟢 VIX={vix_last:.1f}（低波动顺风环境，适合趋势持有）")
        elif vix_last < 22.0:
            score += 2.0
            signals.append(f"🟡 VIX={vix_last:.1f}（正常波动区间，控制买点节奏）")
        elif vix_last < 30.0:
            score -= 8.0
            signals.append(f"🟠 VIX={vix_last:.1f}（波动抬升，严控追高与单笔止损）")
        else:
            score -= 13.0
            signals.append(f"🔴 VIX={vix_last:.1f}（高恐慌区，警惕连环杀跌，但同时留意极端情绪见底信号）")

    # 4. 10Y Treasury Yield Pressure (^TNX, +/- 6 pts)
    tnx_df = ohlcv_map.get("^TNX")
    if tnx_df is not None and len(tnx_df) >= 25:
        tnx_last = float(tnx_df["Close"].iloc[-1])
        # Yahoo ^TNX is often quoted as 10x yield (e.g. 42.5 = 4.25%) or direct (4.25)
        tnx_pct = tnx_last / 10.0 if tnx_last > 15.0 else tnx_last
        tnx_prev20 = float(tnx_df["Close"].iloc[-21])
        tnx_prev20_pct = tnx_prev20 / 10.0 if tnx_prev20 > 15.0 else tnx_prev20
        tnx_chg_bps = (tnx_pct - tnx_prev20_pct) * 100.0
        metrics["US10Y_pct"] = round(tnx_pct, 2)
        metrics["US10Y_20d_chg_bps"] = round(tnx_chg_bps, 1)
        if tnx_chg_bps >= 35:
            score -= 6.0
            signals.append(f"🟠 10年期美债利率近20日飙升 +{tnx_chg_bps:.0f}bps（当前 {tnx_pct:.2f}%），压制高估值科技股")
        elif tnx_chg_bps <= -25:
            score += 5.0
            signals.append(f"🟢 10年期美债利率回落 {tnx_chg_bps:.0f}bps（当前 {tnx_pct:.2f}%），利好科技成长估值")

    # 5. Internal Watchlist Breadth (% above 20MA & 50MA, +/- 12 pts)
    above20_cnt, above50_cnt, above200_cnt, valid_cnt = 0, 0, 0, 0
    for sym in equity_symbols:
        df = ohlcv_map.get(sym)
        if df is None or len(df) < 25:
            continue
        c = df["Close"]
        last = float(c.iloc[-1])
        s20 = _sma(c, 20)
        s50 = _sma(c, 50)
        s200 = _sma(c, 200)
        valid_cnt += 1
        if s20 and last > s20:
            above20_cnt += 1
        if s50 and last > s50:
            above50_cnt += 1
        if s200 and last > s200:
            above200_cnt += 1

    if valid_cnt > 0:
        b20 = above20_cnt / valid_cnt * 100.0
        b50 = above50_cnt / valid_cnt * 100.0
        b200 = above200_cnt / valid_cnt * 100.0
        metrics["breadth_above_20ma_pct"] = round(b20, 1)
        metrics["breadth_above_50ma_pct"] = round(b50, 1)
        metrics["breadth_above_200ma_pct"] = round(b200, 1)
        if b50 >= 65:
            score += 9.0
            signals.append(f"🟢 股票池内部宽度强劲：{b50:.0f}% 标的站稳 50 日均线")
        elif b50 <= 30:
            score -= 10.0
            signals.append(f"🔴 股票池内部宽度转弱：仅 {b50:.0f}% 标的在 50 日均线上方（多数处于调整/下跌趋势）")
        else:
            signals.append(f"🟡 股票池宽度中性：{b50:.0f}% 在 50 日线上方，{b20:.0f}% 在 20 日线上方")

    score = round(max(0.0, min(100.0, score)), 1)

    # Translate numeric score into actionable trading desk guidance
    if score >= 75:
        label = "🟢 强势进攻期 (Bull Expansion)"
        suggested_max_exposure_pct = 90
        suggested_defensive_cash_pct = 10
        action_guide = "主线顺风，可积极持有核心强势股，逢缩量回踩 20 日线加仓，放宽移动止盈让利润奔跑。"
    elif score >= 58:
        label = "🟡 温和多头 / 结构分化 (Selective Bull)"
        suggested_max_exposure_pct = 75
        suggested_defensive_cash_pct = 25
        action_guide = "精选高确定性龙头，不追高、只买盈亏比 >= 2:1 的支撑位或财报催化形态，保留 20%~25% 现金/MCD。"
    elif score >= 42:
        label = "🟠 震荡防守 / 高波动拉锯 (Choppy / Distribution)"
        suggested_max_exposure_pct = 55
        suggested_defensive_cash_pct = 45
        action_guide = "市场多空拉锯，严禁重仓接飞刀！优先保留强基本面底仓，减掉破位弱势股，提高现金与 MCD 防御比例。"
    else:
        label = "🔴 逆风避险 / 退潮防御 (Risk-Off / Bearish)"
        suggested_max_exposure_pct = 35
        suggested_defensive_cash_pct = 65
        action_guide = "科技/半导体处于明显逆风，保住本金第一！严格执行止损、禁止盲目补仓，仅在极度超卖且放量止跌后小仓位试探。"

    return {
        "regime_score": score,
        "regime_label": label,
        "suggested_max_exposure_pct": suggested_max_exposure_pct,
        "suggested_defensive_cash_pct": suggested_defensive_cash_pct,
        "action_guide": action_guide,
        "metrics": metrics,
        "signals": signals,
    }
