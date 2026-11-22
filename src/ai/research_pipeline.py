"""Multi-Agent Local RTX 5070 Ti Research Orchestrator (Supply Chain -> Stock Debate -> PM Plan)."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from src.ai.llm_client import LocalLLMClient
from src.ai.prompts import get_prompts

logger = logging.getLogger(__name__)


def _compact_stock_context(
    symbol: str,
    stock_data: dict[str, Any],
    sec_info: dict[str, Any] | None,
    news_items: list[dict[str, Any]] | None,
) -> str:
    """Serialize real quant, GPU analog/ML, fundamental, SEC, and news facts into tight LLM context."""
    fund = stock_data.get("fundamentals") or {}
    analog = stock_data.get("analog") or {}
    ml = stock_data.get("ml_alpha") or {}
    crash = stock_data.get("crash_analysis") or {}
    ticket = stock_data.get("trade_ticket") or {}

    headlines = [
        f"- [{n.get('published', '')[:10]}] {n.get('title', '')} ({n.get('publisher', '')})"
        for n in (news_items or [])[:7]
    ]
    news_block = "\n".join(headlines) if headlines else "- 暂无最新抓取新闻"

    return (
        f"【标的】{symbol} ({stock_data.get('name', symbol)}) | 板块: {stock_data.get('sector_label', '')} | 角色: {stock_data.get('role', '')}\n"
        f"【价格与趋势】现价 ${stock_data.get('close')} | 1日 {stock_data.get('chg_1d_pct')}% | 5日 {stock_data.get('ret_5d_pct')}% | "
        f"21日 {stock_data.get('ret_21d_pct')}% | 63日 {stock_data.get('ret_63d_pct')}%\n"
        f"  均线位置: 距SMA20 {stock_data.get('dist_sma20_pct')}% | 距SMA50 {stock_data.get('dist_sma50_pct')}% | 距SMA200 {stock_data.get('dist_sma200_pct')}%\n"
        f"  52周区间: ${stock_data.get('low_52w')} - ${stock_data.get('high_52w')} (距高点回撤 {stock_data.get('dd_from_52w_high_pct')}%, 区间分位 {stock_data.get('pos_in_52w_pct')}%)\n"
        f"  周期阶段: {stock_data.get('stage_label')} | RSI14={stock_data.get('rsi14')} | ADX={stock_data.get('adx14')} | ATR%={stock_data.get('atr_pct')}% | VCP={stock_data.get('vcp_ratio')}\n"
        f"【机构量价】RVOL={stock_data.get('rvol')}x | 20日涨跌量比={stock_data.get('up_down_vol_ratio_20d')} | CMF={stock_data.get('cmf20')} | "
        f"吸筹日={stock_data.get('accum_days_20d')} / 派发日={stock_data.get('dist_days_20d')}\n"
        f"【相对强度】21日跑赢QQQ {stock_data.get('rs_qqq_21d_pct')}% | 21日跑赢板块基准 {stock_data.get('rs_sector_21d_pct')}%\n"
        f"【量化暴跌/风险初判】{crash.get('badge', '')} (飞刀风险分 {crash.get('knife_risk_score')}/100): {crash.get('reason', '')}\n"
        f"【GPU 历史相似走势(10年)】21日胜率 {analog.get('win_rate_21d_pct', 'N/A')}% | 中位收益 {analog.get('median_ret_21d_pct', 'N/A')}% | "
        f"42日胜率 {analog.get('win_rate_42d_pct', 'N/A')}% | 盈亏比PF={analog.get('profit_factor_21d', 'N/A')} | 中位最大回撤MAE {analog.get('median_mae_21d_pct', 'N/A')}%\n"
        f"【GPU Walk-Forward ML】未来21日跑赢QQQ概率: {ml.get('prob_outperform_21d_pct', 'N/A')}%\n"
        f"【基本面与财报风控】市值 ${fund.get('market_cap_b')}B | Forward P/E {fund.get('forward_pe')} | 营收同比 {fund.get('revenue_growth_yoy_pct')}% | "
        f"毛利率 {fund.get('gross_margin_pct')}% | 营业利润率 {fund.get('operating_margin_pct')}% | 最近EPS超预期 {fund.get('last_eps_surprise_pct')}% | "
        f"财报状态: {fund.get('earnings_risk_flag')} (预计日期 {fund.get('next_earnings_date')})\n"
        f"【SEC 官方披露】{(sec_info or {}).get('sec_digest', '无')}\n"
        f"【量化交易计划】综合分 {stock_data.get('composite_score')}/100 ({stock_data.get('grade', '')}) | 信号: {ticket.get('action_signal')} | "
        f"入场区 ${ticket.get('entry_low')}-${ticket.get('entry_high')} | 硬止损 ${ticket.get('hard_stop')} (-{ticket.get('stop_dist_pct')}%) | "
        f"目标1 ${ticket.get('target_1')} (R:R={ticket.get('rr_target_1')}) | 目标2 ${ticket.get('target_2')}\n"
        f"【近期新闻标题】\n{news_block}\n"
    )


def analyze_single_stock_with_llm(
    client: LocalLLMClient,
    symbol: str,
    stock_data: dict[str, Any],
    sec_info: dict[str, Any] | None,
    news_items: list[dict[str, Any]] | None,
    lang: str = "en",
) -> dict[str, Any]:
    """Run local 5070 Ti Bull/Bear & Crash Attribution debate on one stock."""
    _, sys_stock, _ = get_prompts(lang)
    ctx = _compact_stock_context(symbol, stock_data, sec_info, news_items)
    if lang == "zh":
        user_prompt = f"请对以下美股标的进行 2 周 ~ 2 个月波段维度的深度多空辩论与风险排雷，严格输出 JSON：\n\n{ctx}"
    else:
        user_prompt = f"Perform a 2-week to 2-month spot equity swing bull/bear debate & drawdown attribution on the following US stock. Output strict JSON only:\n\n{ctx}"
    return client.chat_json(sys_stock, user_prompt, max_tokens=1500)


def run_local_llm_research_suite(
    *,
    client: LocalLLMClient,
    regime: dict[str, Any],
    stock_analyses: dict[str, dict[str, Any]],
    holdings_diagnosis: list[dict[str, Any]],
    lead_lag: list[dict[str, Any]],
    rotation_pairs: list[dict[str, Any]],
    risk_mc: dict[str, Any] | None,
    sec_map: dict[str, dict[str, Any]],
    news_map: dict[str, list[dict[str, Any]]],
    deep_dive_symbols: list[str] | None = None,
    lang: str = "en",
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Run the 3-Stage Local RTX 5070 Ti Multi-Agent Research Suite:
      Stage A: Supply-Chain Strategist (cross-company CapEx -> Compute -> Storage synthesis)
      Stage B: Per-Stock Bull/Bear & Crash Attribution (all holdings + top/extreme watchlist names)
      Stage C: Portfolio Manager Weekly Action Plan (Markdown executive memo)
    """
    health = client.check_health()
    if not health.get("online"):
        return {
            "available": False,
            "model": client.model,
            "error": f"未能连接本地 5070 Ti 推理服务 ({client.base_url})。请先启动 Ollama，或检查 .env 配置。",
            "supply_chain_report": {},
            "stock_reports": {},
            "pm_weekly_memo": "",
        }

    sys_chain, _, sys_pm = get_prompts(lang)

    # --- Stage A: Supply-Chain Strategist ---
    if progress_cb:
        msg = (
            f"[5070 Ti · {client.model}] 正在运行：产业链首席策略师分析..."
            if lang == "zh"
            else f"[5070 Ti · {client.model}] Running Supply-Chain Strategist..."
        )
        progress_cb(msg)

    chain_lines = []
    for sym, sdata in stock_analyses.items():
        if sdata.get("error"):
            continue
        top_news = "; ".join(n.get("title", "") for n in (news_map.get(sym) or [])[:3])
        chain_lines.append(
            f"- {sym} [{sdata.get('sector_label', '')}]: 现价 ${sdata.get('close')}, 1D {sdata.get('chg_1d_pct')}%, "
            f"21D {sdata.get('ret_21d_pct')}%, 综合分 {sdata.get('composite_score')}, "
            f"财报状态 {(sdata.get('fundamentals') or {}).get('earnings_risk_flag', '')}, "
            f"暴跌初判 {(sdata.get('crash_analysis') or {}).get('badge', '')} | 新闻: {top_news}"
        )

    ll_lines = [
        f"- {e['source']} -> {e['target']} ({e['logic']}): 同日相关 {e['same_day_corr']}, "
        f"近10日背离 {e['divergence_10d_pct']}% | {e['signal_note']}"
        for e in lead_lag[:14]
    ]
    pair_lines = [
        f"- {p['label']}: 60日价差Z={p['zscore_60d']}, 近21日 {p['left']} {p['left_21d_pct']}% vs {p['right']} {p['right_21d_pct']}% | {p['interpretation']}"
        for p in rotation_pairs
    ]

    chain_prompt = (
        f"【大盘与半导体温度】评分 {regime.get('regime_score')}/100 ({regime.get('regime_label')})\n"
        f"指标: {json.dumps(regime.get('metrics', {}), ensure_ascii=False)}\n\n"
        f"【全产业链标的快照】\n" + "\n".join(chain_lines) + "\n\n"
        f"【上下游传导 Lead-Lag 统计】\n" + "\n".join(ll_lines) + "\n\n"
        f"【同赛道配对价差偏离】\n" + "\n".join(pair_lines)
    )
    try:
        supply_chain_report = client.chat_json(sys_chain, chain_prompt, max_tokens=1800)
    except Exception as exc:
        logger.warning("Supply chain LLM agent error: %s", exc)
        supply_chain_report = {"error": str(exc)}

    # --- Stage B: Individual Stock Bull/Bear & Crash Attribution ---
    # Default priority: all holdings + top composite scores + biggest drawdown/crash alerts
    if deep_dive_symbols is None:
        holding_syms = [h["symbol"] for h in holdings_diagnosis if h.get("symbol") in stock_analyses]
        ranked = sorted(
            [s for s, d in stock_analyses.items() if not d.get("error")],
            key=lambda s: float(stock_analyses[s].get("composite_score", 0.0)),
            reverse=True,
        )
        # Analyze all active holdings + top watchlist candidates (keeps runtime crisp ~2-4 mins on 5070 Ti)
        chosen: list[str] = []
        for s in holding_syms + ranked:
            if s not in chosen:
                chosen.append(s)
        deep_dive_symbols = chosen[:16]

    stock_reports: dict[str, dict[str, Any]] = {}
    total_d = len(deep_dive_symbols)
    for idx, sym in enumerate(deep_dive_symbols, start=1):
        sdata = stock_analyses.get(sym)
        if not sdata or sdata.get("error"):
            continue
        if progress_cb:
            msg = (
                f"[5070 Ti · {client.model}] 个股多空辩论与排雷 ({idx}/{total_d}): {sym}..."
                if lang == "zh"
                else f"[5070 Ti · {client.model}] Bull/Bear Debate ({idx}/{total_d}): {sym}..."
            )
            progress_cb(msg)
        try:
            stock_reports[sym] = analyze_single_stock_with_llm(
                client=client,
                symbol=sym,
                stock_data=sdata,
                sec_info=sec_map.get(sym),
                news_items=news_map.get(sym),
                lang=lang,
            )
        except Exception as exc:
            logger.warning("Stock LLM agent error for %s: %s", sym, exc)
            stock_reports[sym] = {"error": str(exc)}

    # --- Stage C: Portfolio Manager Executive Weekly Plan ---
    if progress_cb:
        msg = (
            f"[5070 Ti · {client.model}] 正在生成：基金经理《本周波段交易执行计划》..."
            if lang == "zh"
            else f"[5070 Ti · {client.model}] Generating Portfolio Manager Weekly Execution Plan..."
        )
        progress_cb(msg)

    pm_context = {
        "regime": {
            "score": regime.get("regime_score"),
            "label": regime.get("regime_label"),
            "suggested_max_exposure_pct": regime.get("suggested_max_exposure_pct"),
            "signals": regime.get("signals"),
        },
        "portfolio_risk_mc": {
            "gross_exposure_pct": (risk_mc or {}).get("gross_exposure_pct"),
            "portfolio_vol_ann_pct": (risk_mc or {}).get("portfolio_vol_ann_pct"),
            "horizon_21d": (risk_mc or {}).get("horizon_21d"),
            "risk_breakdown": (risk_mc or {}).get("risk_breakdown"),
            "stress_tests": (risk_mc or {}).get("stress_tests"),
        },
        "holdings_diagnosis": holdings_diagnosis,
        "supply_chain_report": supply_chain_report,
        "stock_llm_verdicts": {
            sym: {
                "quant_score": (stock_analyses.get(sym) or {}).get("composite_score"),
                "signal": ((stock_analyses.get(sym) or {}).get("trade_ticket") or {}).get("action_signal"),
                "entry": [
                    ((stock_analyses.get(sym) or {}).get("trade_ticket") or {}).get("entry_low"),
                    ((stock_analyses.get(sym) or {}).get("trade_ticket") or {}).get("entry_high"),
                ],
                "hard_stop": ((stock_analyses.get(sym) or {}).get("trade_ticket") or {}).get("hard_stop"),
                "verdict": rep.get("conviction_verdict"),
                "crash_attribution": rep.get("crash_attribution"),
                "summary": rep.get("one_line_summary"),
            }
            for sym, rep in stock_reports.items()
            if not rep.get("error")
        },
    }

    try:
        pm_intro = (
            "以下是最新全量量化、GPU 蒙特卡洛风险、产业链和个股排雷数据，请撰写《本周波段交易执行计划》：\n\n"
            if lang == "zh"
            else "Here is the complete quant, GPU Student-t Monte Carlo risk, supply-chain, and stock-level dataset. Write the Weekly Swing Execution Plan:\n\n"
        )
        pm_memo = client.chat_text(
            sys_pm,
            pm_intro + json.dumps(pm_context, ensure_ascii=False, indent=2),
            max_tokens=2800,
        )
    except Exception as exc:
        logger.warning("PM LLM agent error: %s", exc)
        pm_memo = f"PM memo generation failed: {exc}"

    return {
        "available": True,
        "model": client.model,
        "lang": lang,
        "supply_chain_report": supply_chain_report,
        "stock_reports": stock_reports,
        "pm_weekly_memo": pm_memo,
    }
