"""Comprehensive Swing Trading Technical, Volume/Institutional Footprint & Crash Classifier Engine."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=min(period, len(df)), adjust=False).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if len(df) < period + 2:
        return pd.Series(20.0, index=df.index)
    high, low = df["High"], df["Low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = compute_atr(df, period).replace(0.0, np.nan)
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().fillna(20.0)


def compute_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow — measures institutional accumulation vs distribution."""
    hl_range = (df["High"] - df["Low"]).replace(0.0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range
    mfv = mfm.fillna(0.0) * df["Volume"]
    vol_sum = df["Volume"].rolling(period, min_periods=min(5, len(df))).sum().replace(0.0, np.nan)
    return (mfv.rolling(period, min_periods=min(5, len(df))).sum() / vol_sum).fillna(0.0)


def compute_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def detect_weinstein_stage(
    close: pd.Series,
    sma50: float | None,
    sma150: float | None,
    sma200: float | None,
) -> tuple[int, str]:
    """
    Stan Weinstein 4-Stage Cycle Classification (essential for 1-2 month swing trading):
      Stage 1: Basing / Bottoming
      Stage 2: Advancing Uptrend (prime long zone)
      Stage 3: Topping / Distribution
      Stage 4: Declining Downtrend (avoid catching falling knives!)
    """
    last = float(close.iloc[-1])
    if sma150 is None or sma50 is None or len(close) < 170:
        # Short-history fallback using 20/50 SMA
        s20 = float(close.rolling(min(20, len(close))).mean().iloc[-1])
        if sma50 is not None:
            if last > s20 > sma50:
                return 2, "Stage 2 (短周期上升)"
            if last < s20 < sma50:
                return 4, "Stage 4 (短周期下跌)"
        return 1, "Stage 1/新股构筑期"

    s150_series = close.rolling(150).mean()
    slope_20 = (
        (float(s150_series.iloc[-1]) / float(s150_series.iloc[-21]) - 1.0) * 100.0
        if len(s150_series.dropna()) > 21
        else 0.0
    )

    if last > sma150 and sma50 > sma150 and slope_20 > 0.4:
        return 2, "Stage 2 · 主升推进期"
    if last < sma150 and sma50 < sma150 and slope_20 < -0.4:
        return 4, "Stage 4 · 下行趋势期"
    if last < sma50 and last >= sma150 * 0.96 and slope_20 >= -0.4:
        return 3, "Stage 3 · 高位派发/筑顶震荡"
    return 1, "Stage 1 · 底部震荡/构筑期"


def detect_support_resistance(df: pd.DataFrame, lookback: int = 120) -> dict[str, float]:
    """Identify nearest structural swing support & resistance levels for stop-loss & targets."""
    sub = df.tail(min(lookback, len(df)))
    last = float(sub["Close"].iloc[-1])
    lows = sub["Low"].values
    highs = sub["High"].values

    pivot_lows: list[float] = []
    pivot_highs: list[float] = []
    n = len(sub)
    for i in range(2, n - 2):
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i - 2] and lows[i] <= lows[i + 1] and lows[i] <= lows[i + 2]:
            pivot_lows.append(float(lows[i]))
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i - 2] and highs[i] >= highs[i + 1] and highs[i] >= highs[i + 2]:
            pivot_highs.append(float(highs[i]))

    supports = sorted([p for p in pivot_lows if p < last * 0.995], reverse=True)
    resistances = sorted([p for p in pivot_highs if p > last * 1.005])

    recent_20_low = float(sub["Low"].tail(min(20, n)).min())
    recent_60_low = float(sub["Low"].tail(min(60, n)).min())
    recent_20_high = float(sub["High"].tail(min(20, n)).max())
    recent_60_high = float(sub["High"].tail(min(60, n)).max())

    support_1 = supports[0] if supports else min(recent_20_low, last * 0.94)
    support_2 = supports[1] if len(supports) > 1 else min(recent_60_low, support_1 * 0.94)
    resistance_1 = resistances[0] if resistances else max(recent_20_high, last * 1.08)
    resistance_2 = resistances[1] if len(resistances) > 1 else max(recent_60_high, resistance_1 * 1.08)

    return {
        "support_1": round(support_1, 2),
        "support_2": round(support_2, 2),
        "resistance_1": round(resistance_1, 2),
        "resistance_2": round(resistance_2, 2),
    }


def analyze_stock_technicals(
    symbol: str,
    df: pd.DataFrame,
    qqq_df: pd.DataFrame | None = None,
    sector_bench_df: pd.DataFrame | None = None,
    fundamental_score: float = 50.0,
) -> dict[str, Any]:
    """
    Compute full quantitative swing-trading technical, institutional-volume,
    relative-strength, VCP, and Crash-vs-Knife diagnostics for one stock.
    """
    if df is None or df.empty:
        return {"symbol": symbol, "error": "no_data"}

    n = len(df)
    close = df["Close"]
    vol = df["Volume"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if n >= 2 else last
    chg_1d = (last / prev - 1.0) * 100.0 if prev > 0 else 0.0

    def _ret(days: int) -> float | None:
        if n <= days:
            return None
        p = float(close.iloc[-1 - days])
        return round((last / p - 1.0) * 100.0, 2) if p > 0 else None

    sma20 = float(close.rolling(20).mean().iloc[-1]) if n >= 20 else None
    sma50 = float(close.rolling(50).mean().iloc[-1]) if n >= 50 else None
    sma150 = float(close.rolling(150).mean().iloc[-1]) if n >= 150 else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else None

    rsi_series = compute_rsi(close, 14)
    rsi14 = float(rsi_series.iloc[-1])
    atr_series = compute_atr(df, 14)
    atr14 = float(atr_series.iloc[-1]) if np.isfinite(atr_series.iloc[-1]) else last * 0.03
    atr_pct = (atr14 / last * 100.0) if last > 0 else 3.0
    adx14 = float(compute_adx(df, 14).iloc[-1])
    cmf20 = float(compute_cmf(df, 20).iloc[-1])
    macd_l, macd_s, macd_h = compute_macd(close)
    macd_hist = float(macd_h.iloc[-1])
    macd_hist_rising = bool(n >= 3 and macd_h.iloc[-1] > macd_h.iloc[-2])

    # 52-week range & drawdown geometry
    w252 = df.tail(min(252, n))
    high_52w = float(w252["High"].max())
    low_52w = float(w252["Low"].min())
    dd_from_52w_high = (last / high_52w - 1.0) * 100.0 if high_52w > 0 else 0.0
    pos_in_52w = (
        (last - low_52w) / (high_52w - low_52w) * 100.0
        if (high_52w > low_52w)
        else 50.0
    )

    # Institutional volume footprint: RVOL, Up/Down Vol Ratio, Accumulation vs Distribution Days
    avg_vol_20 = float(vol.tail(min(20, n)).mean()) if n >= 5 else float(vol.iloc[-1])
    rvol = float(vol.iloc[-1]) / avg_vol_20 if avg_vol_20 > 0 else 1.0
    ret_1d_series = close.pct_change().fillna(0.0)
    tail20 = df.tail(min(20, n)).copy()
    tail20_ret = ret_1d_series.tail(min(20, n))
    up_vol = float(tail20.loc[tail20_ret > 0, "Volume"].sum())
    down_vol = float(tail20.loc[tail20_ret < 0, "Volume"].sum())
    up_down_vol_ratio = round(up_vol / down_vol, 2) if down_vol > 0 else (2.0 if up_vol > 0 else 1.0)

    # Institutional Distribution Days (IBD-style: close <= -0.2% on higher volume than prior day)
    vol_prev = vol.shift(1)
    dist_days_20 = int(
        ((ret_1d_series <= -0.002) & (vol > vol_prev * 1.02)).tail(min(20, n)).sum()
    )
    accum_days_20 = int(
        ((ret_1d_series >= 0.005) & (vol > vol_prev * 1.05)).tail(min(20, n)).sum()
    )

    # VCP (Volatility Contraction Ratio): recent 10D ATR vs 50D ATR
    atr_10 = float(compute_atr(df, 10).iloc[-1]) if n >= 12 else atr14
    atr_50 = float(compute_atr(df, 50).iloc[-1]) if n >= 52 else atr14
    vcp_ratio = round(atr_10 / atr_50, 2) if atr_50 > 0 else 1.0
    is_vcp_tight = bool(vcp_ratio <= 0.78 and sma50 and last > sma50 * 0.97)

    # Relative Strength vs QQQ & Sector Benchmark (SOXX / QQQ / SPY)
    rs_qqq_21d = _compute_relative_return(close, qqq_df, 21)
    rs_qqq_63d = _compute_relative_return(close, qqq_df, 63)
    rs_sec_21d = _compute_relative_return(close, sector_bench_df, 21)
    rs_sec_63d = _compute_relative_return(close, sector_bench_df, 63)

    # Realized annualized volatility (20D)
    realized_vol_20d = (
        float(ret_1d_series.tail(20).std() * np.sqrt(252) * 100.0) if n >= 21 else atr_pct * np.sqrt(252)
    )

    # Stage & Structure
    stage_num, stage_label = detect_weinstein_stage(close, sma50, sma150, sma200)
    levels = detect_support_resistance(df)

    # Sub-pillar 1: Trend & Stage Score (0-100)
    trend_score = 50.0
    if stage_num == 2:
        trend_score += 22
    elif stage_num == 1:
        trend_score += 4
    elif stage_num == 3:
        trend_score -= 10
    elif stage_num == 4:
        trend_score -= 25

    if sma20:
        dist20 = (last / sma20 - 1.0) * 100.0
        if 0 <= dist20 <= 6.0:
            trend_score += 10  # sweet spot above 20MA without chasing
        elif dist20 > 12.0:
            trend_score -= 8   # overextended risk!
        elif dist20 < -8.0:
            trend_score -= 8
    else:
        dist20 = None

    if sma50:
        dist50 = (last / sma50 - 1.0) * 100.0
        trend_score += 10 if last > sma50 else -10
    else:
        dist50 = None

    if sma200:
        dist200 = (last / sma200 - 1.0) * 100.0
        trend_score += 8 if last > sma200 else -10
    else:
        dist200 = None

    if is_vcp_tight:
        trend_score += 8
    trend_score = round(max(5.0, min(98.0, trend_score)), 1)

    # Sub-pillar 2: Volume & Institutional Flow Score (0-100)
    vol_score = 50.0
    vol_score += max(-18.0, min(18.0, cmf20 * 80.0))
    if up_down_vol_ratio >= 1.35:
        vol_score += 14
    elif up_down_vol_ratio >= 1.05:
        vol_score += 6
    elif up_down_vol_ratio <= 0.70:
        vol_score -= 14
    elif up_down_vol_ratio <= 0.90:
        vol_score -= 6
    vol_score += (accum_days_20 - dist_days_20) * 3.0
    vol_score = round(max(5.0, min(98.0, vol_score)), 1)

    # Sub-pillar 3: Relative Strength Score (0-100)
    rs_score = 50.0
    if rs_qqq_21d is not None:
        rs_score += max(-20.0, min(20.0, rs_qqq_21d * 1.5))
    if rs_qqq_63d is not None:
        rs_score += max(-15.0, min(15.0, rs_qqq_63d * 0.7))
    if rs_sec_21d is not None:
        rs_score += max(-15.0, min(15.0, rs_sec_21d * 1.3))
    rs_score = round(max(5.0, min(98.0, rs_score)), 1)

    # Crash Classifier: "错杀抄底 vs 接飞刀"
    crash_info = classify_crash_vs_knife(
        dd_from_52w_high=dd_from_52w_high,
        pos_in_52w=pos_in_52w,
        chg_1d=chg_1d,
        ret_5d=_ret(5) or 0.0,
        rsi14=rsi14,
        rvol=rvol,
        cmf20=cmf20,
        dist_days_20=dist_days_20,
        stage_num=stage_num,
        macd_hist_rising=macd_hist_rising,
        fundamental_score=fundamental_score,
    )

    return {
        "symbol": symbol,
        "history_bars": n,
        "short_history_warning": n < 60,
        "last_date": str(df.index[-1].date()),
        "close": round(last, 2),
        "chg_1d_pct": round(chg_1d, 2),
        "ret_5d_pct": _ret(5),
        "ret_21d_pct": _ret(21),
        "ret_63d_pct": _ret(63),
        "sma20": round(sma20, 2) if sma20 else None,
        "sma50": round(sma50, 2) if sma50 else None,
        "sma150": round(sma150, 2) if sma150 else None,
        "sma200": round(sma200, 2) if sma200 else None,
        "dist_sma20_pct": round(dist20, 2) if dist20 is not None else None,
        "dist_sma50_pct": round(dist50, 2) if dist50 is not None else None,
        "dist_sma200_pct": round(dist200, 2) if dist200 is not None else None,
        "rsi14": round(rsi14, 1),
        "adx14": round(adx14, 1),
        "atr14": round(atr14, 2),
        "atr_pct": round(atr_pct, 2),
        "realized_vol_20d_pct": round(realized_vol_20d, 1),
        "macd_hist": round(macd_hist, 4),
        "macd_hist_rising": macd_hist_rising,
        "cmf20": round(cmf20, 3),
        "rvol": round(rvol, 2),
        "up_down_vol_ratio_20d": up_down_vol_ratio,
        "accum_days_20d": accum_days_20,
        "dist_days_20d": dist_days_20,
        "vcp_ratio": vcp_ratio,
        "is_vcp_tight": is_vcp_tight,
        "high_52w": round(high_52w, 2),
        "low_52w": round(low_52w, 2),
        "dd_from_52w_high_pct": round(dd_from_52w_high, 2),
        "pos_in_52w_pct": round(pos_in_52w, 1),
        "rs_qqq_21d_pct": rs_qqq_21d,
        "rs_qqq_63d_pct": rs_qqq_63d,
        "rs_sector_21d_pct": rs_sec_21d,
        "rs_sector_63d_pct": rs_sec_63d,
        "stage_num": stage_num,
        "stage_label": stage_label,
        "levels": levels,
        "trend_score": trend_score,
        "volume_score": vol_score,
        "rs_score": rs_score,
        "crash_analysis": crash_info,
    }


def classify_crash_vs_knife(
    *,
    dd_from_52w_high: float,
    pos_in_52w: float,
    chg_1d: float,
    ret_5d: float,
    rsi14: float,
    rvol: float,
    cmf20: float,
    dist_days_20: int,
    stage_num: int,
    macd_hist_rising: bool,
    fundamental_score: float,
) -> dict[str, Any]:
    """
    Distinguish between:
      - Falling Knife (放量破位下行，机构派发未结束 -> 禁止盲目抄底)
      - Oversold Watch / Left-side Stabilization (超卖但仍缺右侧止跌确认)
      - High-Probability Mispriced Pullback (基本面稳健 + 情绪超卖 + 抛压衰竭)
      - Healthy Uptrend / Breakout Setup
    """
    in_deep_drawdown = dd_from_52w_high <= -18.0 or pos_in_52w <= 22.0 or ret_5d <= -7.0
    heavy_selling = (chg_1d <= -3.5 and rvol >= 1.35) or dist_days_20 >= 6 or cmf20 <= -0.15

    if not in_deep_drawdown and stage_num == 2:
        return {
            "status": "UPTREND_HEALTHY",
            "badge": "🔵 上升趋势/常规波段",
            "knife_risk_score": 20,
            "reason": "未处于深幅回撤，处于 Stage 2 正常上升结构，按趋势回踩或突破策略操作。",
        }

    # Calculate Falling Knife Risk Score (0 = safe pullback, 100 = lethal falling knife)
    knife_risk = 35
    reasons: list[str] = []

    if stage_num == 4:
        knife_risk += 25
        reasons.append("处于 Stage 4 中长期下跌趋势")
    if heavy_selling:
        knife_risk += 22
        reasons.append(f"机构放量派发未止(近20日派发日 {dist_days_20} 次, RVOL {rvol:.1f}x, CMF {cmf20:+.2f})")
    if chg_1d <= -4.0:
        knife_risk += 10
        reasons.append(f"当日长阴杀跌({chg_1d:.1f}%)，惯性未消")
    if fundamental_score < 45:
        knife_risk += 15
        reasons.append("基本面量化评分偏弱，警惕戴维斯双杀")
    elif fundamental_score >= 65:
        knife_risk -= 14
        reasons.append("基本面业绩与盈利质量支撑较强")

    if rsi14 <= 32 and macd_hist_rising and chg_1d > -1.5:
        knife_risk -= 18
        reasons.append(f"RSI({rsi14:.0f})深度超卖且 MACD 柱状图出现底背离企稳迹象")
    elif rsi14 <= 30:
        knife_risk -= 6
        reasons.append(f"RSI({rsi14:.0f})进入极端超卖区，具备反弹弹性但需确认止跌")

    if cmf20 >= 0.05 and rvol < 1.15:
        knife_risk -= 10
        reasons.append("回调阶段资金未明显外流，偏缩量洗盘特征")

    knife_risk = max(5, min(98, knife_risk))

    if knife_risk >= 68:
        status = "FALLING_KNIFE"
        badge = "🔴 警惕接飞刀 (禁左侧重仓)"
    elif knife_risk >= 45:
        status = "LEFT_SIDE_WATCH"
        badge = "🟡 左侧超卖观察 (等止跌放量阳线)"
    else:
        status = "MISPRICED_PULLBACK"
        badge = "🟢 高概率错杀回踩 (留意分批低吸)"

    return {
        "status": status,
        "badge": badge,
        "knife_risk_score": int(knife_risk),
        "reason": "；".join(reasons) if reasons else "正常区间震荡，按支撑与止损纪律执行。",
    }


def _compute_relative_return(
    stock_close: pd.Series,
    bench_df: pd.DataFrame | None,
    days: int,
) -> float | None:
    if bench_df is None or bench_df.empty or len(stock_close) <= days or len(bench_df) <= days:
        return None
    aligned = pd.concat(
        [stock_close.rename("s"), bench_df["Close"].rename("b")], axis=1
    ).dropna()
    if len(aligned) <= days:
        return None
    s_ret = aligned["s"].iloc[-1] / aligned["s"].iloc[-1 - days] - 1.0
    b_ret = aligned["b"].iloc[-1] / aligned["b"].iloc[-1 - days] - 1.0
    return round((s_ret - b_ret) * 100.0, 2)
