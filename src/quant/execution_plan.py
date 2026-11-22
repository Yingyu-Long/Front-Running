"""Professional Execution Plan, Composite Swing Scoring, Smart Position Sizer & Holdings Diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_composite_swing_score(
    *,
    tech: dict[str, Any],
    fundamentals: dict[str, Any],
    analog: dict[str, Any] | None,
    ml_alpha: dict[str, Any] | None,
    sec_info: dict[str, Any] | None,
    regime_score: float,
) -> dict[str, Any]:
    """
    Transparent 5-Pillar Composite Swing Score (0-100) tailored for 2-week to 2-month US spot equities:
      1. Trend & Stage Structure       (25%)
      2. Volume & Institutional Flow   (20%)
      3. Relative Strength & Momentum  (20%)
      4. GPU Quant Edge (Analog + ML)  (20%)
      5. Fundamental Quality & Events  (15%)
    Plus explicit Risk Adjustments (Earnings proximity, SEC red flags, Falling Knife cap).
    """
    trend_s = float(tech.get("trend_score", 50.0))
    vol_s = float(tech.get("volume_score", 50.0))
    rs_s = float(tech.get("rs_score", 50.0))
    fund_s = float(fundamentals.get("fundamental_score", 50.0))

    analog_s = float((analog or {}).get("analog_score", 50.0))
    ml_s = float((ml_alpha or {}).get("ml_alpha_score", 50.0))
    if analog and ml_alpha:
        gpu_edge_s = 0.55 * analog_s + 0.45 * ml_s
    elif analog:
        gpu_edge_s = analog_s
    elif ml_alpha:
        gpu_edge_s = ml_s
    else:
        gpu_edge_s = 50.0

    raw_score = (
        0.25 * trend_s
        + 0.20 * vol_s
        + 0.20 * rs_s
        + 0.20 * gpu_edge_s
        + 0.15 * fund_s
    )

    penalties: list[str] = []
    # Earnings risk guard: penalize brand-new swing entries right before earnings roulette
    days_to_ed = fundamentals.get("days_to_earnings")
    if days_to_ed is not None and days_to_ed <= 7:
        raw_score -= 7.0
        penalties.append(f"财报仅剩 {days_to_ed} 天(-7分防赌财报)")
    elif days_to_ed is not None and days_to_ed <= 14:
        raw_score -= 3.0
        penalties.append(f"两周内发布财报(-3分)")

    # SEC red flags penalty
    red_flags = (sec_info or {}).get("red_flag_filings") or []
    if red_flags:
        raw_score -= min(10.0, len(red_flags) * 4.0)
        penalties.append(f"SEC 存在 {len(red_flags)} 条敏感公告提示")

    # Falling knife guardrail
    crash = tech.get("crash_analysis") or {}
    if crash.get("status") == "FALLING_KNIFE":
        raw_score = min(raw_score, 48.0)
        penalties.append("处于放量破位下跌期(综合分上限封顶 48)")

    # Macro regime modest tilt (+/- 4 pts)
    raw_score += (regime_score - 50.0) * 0.08
    final_score = round(max(5.0, min(98.0, raw_score)), 1)

    if final_score >= 74:
        grade = "A · 高确定性波段标的"
    elif final_score >= 62:
        grade = "B · 优质观察/择机低吸"
    elif final_score >= 48:
        grade = "C · 中性震荡/耐心等待"
    else:
        grade = "D · 弱势/高风险回避"

    return {
        "composite_score": final_score,
        "grade": grade,
        "pillars": {
            "trend": round(trend_s, 1),
            "volume_flow": round(vol_s, 1),
            "relative_strength": round(rs_s, 1),
            "gpu_quant_edge": round(gpu_edge_s, 1),
            "fundamentals": round(fund_s, 1),
        },
        "penalties": penalties,
    }


def build_trade_ticket(
    *,
    symbol: str,
    tech: dict[str, Any],
    composite: dict[str, Any],
    fundamentals: dict[str, Any],
    account_cfg: dict[str, Any],
    regime: dict[str, Any],
    sector_key: str,
    current_sector_weight_pct: float = 0.0,
) -> dict[str, Any]:
    """
    Generate an institutional-grade Trade Ticket for 2-week to 2-month swing execution:
      - Optimal Entry Zone (avoids chasing extended moves)
      - Structural + ATR Hard Stop-Loss & Chandelier Trailing Stop
      - Target 1 (2R) & Target 2 (3R / Resistance)
      - Risk-Reward Ratio (R:R)
      - Volatility-Targeted + Regime-Scaled + Correlation-Adjusted Position Sizing
    """
    close = float(tech["close"])
    atr = float(tech.get("atr14") or close * 0.03)
    sma20 = tech.get("sma20")
    levels = tech.get("levels") or {}
    sup1 = float(levels.get("support_1", close * 0.92))
    res1 = float(levels.get("resistance_1", close * 1.12))
    res2 = float(levels.get("resistance_2", close * 1.22))

    # 1. Entry Zone: avoid chasing far above 20MA
    if sma20 and close > sma20 * 1.07:
        entry_high = round(min(close, sma20 * 1.045), 2)
        entry_low = round(max(sup1, sma20 * 0.995), 2)
        entry_style = "偏离20日线较远，禁追高，等缩量回踩入场区"
    elif (tech.get("crash_analysis") or {}).get("status") in ("FALLING_KNIFE", "LEFT_SIDE_WATCH"):
        entry_low = round(max(sup1 * 0.99, close - 0.8 * atr), 2)
        entry_high = round(close + 0.3 * atr, 2)
        entry_style = "左侧超卖阶段，需等放量止跌阳线或站回5日线后分批介入"
    else:
        entry_low = round(max(sup1 * 1.005, close - 0.65 * atr), 2)
        entry_high = round(close + 0.35 * atr, 2)
        entry_style = "结构健康，可在入场区间内分 2 批建仓"

    if entry_low > entry_high:
        entry_low, entry_high = round(min(entry_low, entry_high), 2), round(max(entry_low, entry_high), 2)
    ref_entry = round((entry_low + entry_high) / 2.0, 2)

    # 2. Hard Stop-Loss: combine structural support & 2.0x ATR, bounded to [4.5%, 14%] risk
    atr_stop = ref_entry - 2.0 * atr
    struct_stop = sup1 - 0.5 * atr
    raw_stop = min(atr_stop, struct_stop) if struct_stop > ref_entry * 0.86 else atr_stop
    min_stop = ref_entry * 0.86   # never wider than 14% for single swing trade
    max_stop = ref_entry * 0.955  # never tighter than 4.5% (avoids normal noise shakeout)
    hard_stop = round(max(min_stop, min(max_stop, raw_stop)), 2)

    risk_per_share = max(ref_entry - hard_stop, ref_entry * 0.03)
    stop_dist_pct = round((1.0 - hard_stop / ref_entry) * 100.0, 2)

    # 3. Chandelier Trailing Stop (for riding 1-2 month winners)
    trailing_stop_guide = round(max(hard_stop, close - 2.6 * atr), 2)

    # 4. Targets & Reward/Risk
    target_1 = round(max(ref_entry + 2.0 * risk_per_share, res1), 2)
    target_2 = round(max(ref_entry + 3.2 * risk_per_share, res2, target_1 * 1.08), 2)
    rr_t1 = round((target_1 - ref_entry) / risk_per_share, 2)
    rr_t2 = round((target_2 - ref_entry) / risk_per_share, 2)

    # 5. Smart Position Sizing
    total_equity = float(account_cfg.get("total_equity_usd", 100_000.0))
    max_risk_pct = float(account_cfg.get("max_risk_per_trade_pct", 1.25))
    max_pos_pct = float(account_cfg.get("max_single_position_pct", 20.0))
    max_sec_pct = float(account_cfg.get("max_sector_exposure_pct", 45.0))
    min_rr = float(account_cfg.get("min_reward_risk_ratio", 2.0))

    # Regime scaling: cut new position sizes when market regime is defensive
    regime_exp = float(regime.get("suggested_max_exposure_pct", 70.0)) / 100.0
    size_mult = regime_exp

    # Correlation/Sector concentration penalty (crucial when holding multiple AI/Storage names!)
    conc_note = ""
    if sector_key != "defensive" and current_sector_weight_pct >= max_sec_pct * 0.85:
        size_mult *= 0.60
        conc_note = f"同赛道当前敞口已达 {current_sector_weight_pct:.0f}%，自动折减 40% 新仓规模"
    elif sector_key != "defensive" and current_sector_weight_pct >= max_sec_pct * 0.60:
        size_mult *= 0.80
        conc_note = f"同赛道敞口 {current_sector_weight_pct:.0f}%，自动折减 20% 规模防集中度过高"

    # Earnings proximity penalty
    days_to_ed = fundamentals.get("days_to_earnings")
    if days_to_ed is not None and days_to_ed <= 10:
        size_mult *= 0.50
        conc_note = (conc_note + "；" if conc_note else "") + f"距财报仅 {days_to_ed} 天，新仓规模减半"

    risk_budget_usd = total_equity * (max_risk_pct / 100.0) * size_mult
    raw_shares = int(risk_budget_usd / risk_per_share)
    max_dollar_cap = total_equity * (max_pos_pct / 100.0) * size_mult
    cap_shares = int(max_dollar_cap / max(ref_entry, 0.01))
    rec_shares = max(0, min(raw_shares, cap_shares))
    rec_dollar = round(rec_shares * ref_entry, 0)
    rec_weight_pct = round(rec_dollar / max(total_equity, 1.0) * 100.0, 1)
    max_loss_usd = round(rec_shares * risk_per_share, 0)

    # Final Action Signal
    comp_score = float(composite["composite_score"])
    crash_status = (tech.get("crash_analysis") or {}).get("status", "")
    if crash_status == "FALLING_KNIFE":
        action_signal = "观望/禁接飞刀"
    elif comp_score >= 72 and rr_t1 >= min_rr and regime.get("regime_score", 50) >= 45:
        action_signal = "优先买入候选"
    elif comp_score >= 60 and rr_t1 >= min_rr:
        action_signal = "择机低吸/分批试仓"
    elif comp_score >= 48:
        action_signal = "跟踪观察"
    else:
        action_signal = "偏弱回避"

    return {
        "symbol": symbol,
        "action_signal": action_signal,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "ref_entry": ref_entry,
        "entry_style": entry_style,
        "hard_stop": hard_stop,
        "stop_dist_pct": stop_dist_pct,
        "trailing_stop_guide": trailing_stop_guide,
        "target_1": target_1,
        "target_2": target_2,
        "rr_target_1": rr_t1,
        "rr_target_2": rr_t2,
        "recommended_shares": rec_shares,
        "recommended_usd": rec_dollar,
        "recommended_weight_pct": rec_weight_pct,
        "planned_max_risk_usd": max_loss_usd,
        "sizing_note": conc_note or "按单笔固定风险与大盘温度自动测算",
    }


def diagnose_active_holdings(
    *,
    portfolio_cfg: dict[str, Any],
    stock_analyses: dict[str, dict[str, Any]],
    regime: dict[str, Any],
    risk_mc: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Produce an explicit, actionable review table for every current holding
    (GOOG, STX, MCD, AMZN, ORCL, SKHY): P&L, Stop Distance, Risk Contribution, Action Directive.
    """
    holdings = portfolio_cfg.get("holdings") or []
    account = portfolio_cfg.get("account") or {}
    total_nav = float((risk_mc or {}).get("total_nav_usd") or account.get("total_equity_usd", 100_000.0))

    rc_map = {
        row["symbol"]: row["risk_contribution_pct"]
        for row in ((risk_mc or {}).get("risk_breakdown") or [])
    }

    rows: list[dict[str, Any]] = []
    for h in holdings:
        sym = str(h.get("symbol", "")).upper().strip()
        shares = float(h.get("shares", 0.0))
        avg_cost = float(h.get("avg_cost", 0.0))
        manual_stop = float(h.get("stop_loss", 0.0))
        thesis = str(h.get("thesis", ""))

        info = stock_analyses.get(sym)
        if not info or info.get("error"):
            rows.append(
                {
                    "symbol": sym,
                    "name": sym,
                    "sector_label": "",
                    "shares": shares,
                    "avg_cost": avg_cost,
                    "close": None,
                    "chg_1d_pct": None,
                    "market_value_usd": 0.0,
                    "weight_pct": 0.0,
                    "pnl_pct": None,
                    "pnl_usd": None,
                    "composite_score": None,
                    "active_stop": None,
                    "trailing_stop": None,
                    "dist_to_stop_pct": None,
                    "risk_contribution_pct": None,
                    "earnings_flag": "待刷新",
                    "action": "待刷新行情",
                    "rationale": "未能拉取该标的最新行情",
                    "thesis": thesis,
                }
            )
            continue

        close = float(info["close"])
        ticket = info.get("trade_ticket") or {}
        comp_score = float(info.get("composite_score", 50.0))
        crash = info.get("crash_analysis") or {}
        fund = info.get("fundamentals") or {}

        mkt_val = shares * close
        weight_pct = mkt_val / max(total_nav, 1.0) * 100.0
        pnl_pct = ((close / avg_cost - 1.0) * 100.0) if avg_cost > 0 else None
        pnl_usd = ((close - avg_cost) * shares) if avg_cost > 0 else None

        active_stop = manual_stop if manual_stop > 0 else float(ticket.get("hard_stop", close * 0.90))
        trail_stop = float(ticket.get("trailing_stop_guide", active_stop))
        # If profitably up > 10%, suggest raising trailing stop at least to breakeven
        if pnl_pct is not None and pnl_pct >= 10.0 and avg_cost > 0:
            trail_stop = max(trail_stop, round(avg_cost * 1.005, 2))

        dist_to_stop_pct = round((close / active_stop - 1.0) * 100.0, 2) if active_stop > 0 else 0.0
        risk_contrib = rc_map.get(sym)

        # Action Directive Logic
        reasons: list[str] = []
        if close <= active_stop:
            action = "触发止损线 · 建议减仓/止损"
            reasons.append(f"现价 ${close:.2f} 已跌破风控止损线 ${active_stop:.2f}")
        elif crash.get("status") == "FALLING_KNIFE" and info.get("sector") != "defensive":
            action = "破位警戒 · 逢反抽减仓/严守止损"
            reasons.append("放量破位下跌，禁止盲目摊平补仓")
        elif info.get("sector") == "defensive":
            if regime.get("regime_score", 50) < 55:
                action = "坚守防御底仓"
                reasons.append("当前科技逆风期，防御底仓有效平滑净值回撤")
            else:
                action = "持有底仓 (进攻期可适度置换至高分龙头)"
                reasons.append("大盘回暖，可保留底仓或小幅腾挪资金")
        elif comp_score >= 68:
            if pnl_pct is not None and pnl_pct >= 15.0:
                action = "强势持有 · 上移移动止盈"
                reasons.append(f"趋势与综合分({comp_score})强，建议移动止盈保护利润至 ${trail_stop:.2f}")
            else:
                action = "核心持有 · 逢回踩可加仓"
                reasons.append(f"综合评分 {comp_score}，趋势与筹码结构健康")
        elif comp_score >= 52:
            action = "持有观察 · 守住止损线"
            reasons.append(f"处于震荡整理区，严守 ${active_stop:.2f} 止损")
        else:
            action = "弱势预警 · 考虑去弱留强"
            reasons.append(f"综合评分偏低({comp_score})，若反弹无力建议换仓至同赛道更强标的")

        days_to_ed = fund.get("days_to_earnings")
        if days_to_ed is not None and days_to_ed <= 10:
            reasons.append(f"距财报仅 {days_to_ed} 天，提前规划持仓过财报仓位")
        if risk_contrib is not None and risk_contrib >= 28.0:
            reasons.append(f"单股贡献了组合 {risk_contrib:.1f}% 的总风险，集中度偏高")

        rows.append(
            {
                "symbol": sym,
                "name": info.get("name", sym),
                "sector_label": info.get("sector_label", ""),
                "shares": shares,
                "avg_cost": avg_cost,
                "close": close,
                "chg_1d_pct": info.get("chg_1d_pct"),
                "market_value_usd": round(mkt_val, 0),
                "weight_pct": round(weight_pct, 1),
                "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
                "pnl_usd": round(pnl_usd, 0) if pnl_usd is not None else None,
                "composite_score": comp_score,
                "active_stop": round(active_stop, 2),
                "trailing_stop": round(trail_stop, 2),
                "dist_to_stop_pct": dist_to_stop_pct,
                "risk_contribution_pct": risk_contrib,
                "earnings_flag": fund.get("earnings_risk_flag", "待确认"),
                "action": action,
                "rationale": "；".join(reasons),
                "thesis": thesis,
            }
        )

    return rows
