"""End-to-end 8-stage trading intelligence orchestrator with RTX 5070 Ti VRAM handoff."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.ai.llm_client import LocalLLMClient
from src.ai.research_pipeline import (
    analyze_single_stock_with_llm,
    run_local_llm_research_suite,
)
from src.config_loader import (
    AppSettings,
    get_all_equity_symbols,
    load_portfolio,
    load_settings,
    load_universe,
)
from src.data.fundamentals import FundamentalFetcher
from src.data.market_data import MarketDataFetcher
from src.data.news_feed import NewsAggregator
from src.data.sec_edgar import SecEdgarClient
from src.quant.device import get_device_profile, release_gpu_memory
from src.quant.execution_plan import (
    build_trade_ticket,
    compute_composite_swing_score,
    diagnose_active_holdings,
)
from src.quant.gpu_analog import run_gpu_historical_analog_search
from src.quant.gpu_backtest import run_gpu_grid_backtest
from src.quant.gpu_ml_alpha import run_gpu_walkforward_ml_alpha
from src.quant.gpu_risk_mc import run_gpu_portfolio_risk_mc
from src.quant.indicators import analyze_stock_technicals
from src.quant.regime import evaluate_market_regime
from src.quant.supply_chain import (
    compute_pair_rotation_spreads,
    compute_supply_chain_lead_lag,
    rank_within_sectors,
)
from src.storage.db import TradingDB

logger = logging.getLogger(__name__)


def run_full_analysis_pipeline(
    *,
    settings: AppSettings | None = None,
    universe_cfg: dict[str, Any] | None = None,
    portfolio_cfg: dict[str, Any] | None = None,
    force_refresh_data: bool = False,
    run_ml_alpha: bool = True,
    run_backtest: bool = True,
    run_llm: bool = True,
    llm_model_override: str | None = None,
    llm_base_url_override: str | None = None,
    lang: str = "en",
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """
    Execute the complete institutional swing-trading pipeline and persist snapshot + Markdown report.
    Designed for on-demand execution on Windows WSL2 + RTX 5070 Ti (run in ~3-8 min, then shut down PC).
    """
    t_start = time.perf_counter()
    cfg = settings or load_settings()
    u_cfg = universe_cfg or load_universe()
    p_cfg = portfolio_cfg or load_portfolio()
    db = TradingDB(cfg.db_path)

    dev_profile = get_device_profile(cfg.compute_device)
    llm_model = llm_model_override or cfg.llm_model
    llm_base_url = llm_base_url_override or cfg.llm_base_url

    def _emit(pct: float, msg: str) -> None:
        logger.info("[%.0f%%] %s", pct * 100, msg)
        if progress_cb:
            progress_cb(pct, msg)

    # Combine universe symbols + any extra custom holdings
    eq_symbols = get_all_equity_symbols(u_cfg)
    holding_syms = [
        str(h.get("symbol", "")).upper().strip()
        for h in (p_cfg.get("holdings") or [])
        if h.get("symbol")
    ]
    all_equities = list(dict.fromkeys(eq_symbols + holding_syms))

    # --- Stage 1: Market OHLCV Data ---
    _emit(0.06, f"[1/8] 拉取美股 {len(all_equities)} 只标的 + 宏观指数 10 年历史行情...")
    md_fetcher = MarketDataFetcher(cfg.data_dir, history_years=cfg.history_years)
    ohlcv_map, resolved_map = md_fetcher.fetch_universe(
        u_cfg, extra_symbols=holding_syms, force_refresh=force_refresh_data
    )

    # --- Stage 2: Fundamentals + SEC EDGAR + Financial News ---
    _emit(0.18, "[2/8] 拉取财务基本面、财报日历风控、SEC 官方公告与最新新闻...")
    fund_fetcher = FundamentalFetcher(db=db)
    fundamentals_map = fund_fetcher.fetch_batch(
        all_equities, resolved_map=resolved_map, force_refresh=force_refresh_data
    )

    sec_client = SecEdgarClient(user_agent=cfg.sec_user_agent, db=db)
    sec_map = sec_client.fetch_batch(all_equities, force_refresh=force_refresh_data)

    news_agg = NewsAggregator(db=db)
    news_map = news_agg.fetch_batch(
        u_cfg, all_equities, resolved_map=resolved_map, max_items=8, force_refresh=force_refresh_data
    )

    # --- Stage 3: Macro Regime + Technicals & Crash Classifier ---
    _emit(0.32, "[3/8] 计算宏观与半导体温度计、Weinstein周期、机构量价与暴跌排雷分...")
    regime = evaluate_market_regime(ohlcv_map, all_equities)

    u_meta = u_cfg.get("universe") or {}
    sectors_meta = u_cfg.get("sectors") or {}
    qqq_df = ohlcv_map.get("QQQ")

    tech_map: dict[str, dict[str, Any]] = {}
    for sym in all_equities:
        df = ohlcv_map.get(sym)
        if df is None or df.empty:
            tech_map[sym] = {"symbol": sym, "error": "行情数据拉取失败"}
            continue
        sec_key = str((u_meta.get(sym) or {}).get("sector", "other"))
        bench_sym = str((sectors_meta.get(sec_key) or {}).get("benchmark", "QQQ"))
        bench_df = ohlcv_map.get(bench_sym) or qqq_df
        f_score = float((fundamentals_map.get(sym) or {}).get("fundamental_score", 50.0))
        tech_map[sym] = analyze_stock_technicals(
            symbol=sym,
            df=df,
            qqq_df=qqq_df,
            sector_bench_df=bench_df,
            fundamental_score=f_score,
        )

    # --- Stage 4: [GPU Engine #1] Historical Analog Search ---
    _emit(0.45, f"[4/8] [{dev_profile.backend.upper()}] 运行 GPU 10年历史相似形态张量检索...")
    analog_res = run_gpu_historical_analog_search(
        ohlcv_map, all_equities, window=30, top_k=60, device_pref=cfg.compute_device
    )

    # --- Stage 5: [GPU Engine #2] Strict Purged Walk-Forward ML Alpha ---
    if run_ml_alpha:
        _emit(0.56, f"[5/8] [{dev_profile.backend.upper()}] 运行 GPU 严格样本外 Walk-Forward 机器学习 (XGBoost CUDA + PyTorch ResMLP)...")
        ml_res = run_gpu_walkforward_ml_alpha(
            ohlcv_map, all_equities, horizon=21, n_splits=4, device_pref=cfg.compute_device
        )
    else:
        ml_res = {"available": False, "reason": "本次跳过 ML 模块", "per_symbol": {}}

    # --- Stage 6: [GPU Engine #3] 100k Student-t Monte Carlo Risk + Supply Chain + Trade Tickets ---
    _emit(0.68, f"[6/8] [{dev_profile.backend.upper()}] 运行 {cfg.mc_simulations:,} 路径 Student-t 相关蒙特卡洛风险模拟 & 产业链传导...")
    risk_mc = run_gpu_portfolio_risk_mc(
        ohlcv_map=ohlcv_map,
        portfolio_cfg=p_cfg,
        universe_cfg=u_cfg,
        n_sims=cfg.mc_simulations,
        horizon_days=21,
        device_pref=cfg.compute_device,
    )

    lead_lag = compute_supply_chain_lead_lag(
        ohlcv_map,
        u_cfg.get("supply_chain_edges") or [],
        device_pref=cfg.compute_device,
    )
    rotation_pairs = compute_pair_rotation_spreads(
        ohlcv_map, u_cfg.get("rotation_pairs") or []
    )

    # Compute current sector weight % for correlation-aware sizing
    sector_wt_by_key: dict[str, float] = {}
    if risk_mc.get("available"):
        total_nav = float(risk_mc.get("total_nav_usd", 1.0))
        for rb in risk_mc.get("risk_breakdown") or []:
            sk = str((u_meta.get(rb["symbol"]) or {}).get("sector", "other"))
            sector_wt_by_key[sk] = sector_wt_by_key.get(sk, 0.0) + float(rb["market_value_usd"]) / max(total_nav, 1.0) * 100.0

    # Assemble per-stock unified record + Composite Score + Trade Ticket
    stock_analyses: dict[str, dict[str, Any]] = {}
    analog_per = analog_res.get("per_symbol") or {}
    ml_per = ml_res.get("per_symbol") or {}
    account_cfg = p_cfg.get("account") or {}

    for sym in all_equities:
        t = tech_map.get(sym) or {"symbol": sym, "error": "missing"}
        meta = u_meta.get(sym) or {}
        sec_key = str(meta.get("sector", "other"))
        sec_label = str((sectors_meta.get(sec_key) or {}).get("label", sec_key))
        fund = fundamentals_map.get(sym) or {}
        sec_info = sec_map.get(sym) or {}
        an = analog_per.get(sym)
        ml = ml_per.get(sym)

        if t.get("error"):
            stock_analyses[sym] = {
                **t,
                "name": meta.get("name", sym),
                "sector": sec_key,
                "sector_label": sec_label,
                "role": meta.get("role", "watchlist"),
            }
            continue

        comp = compute_composite_swing_score(
            tech=t,
            fundamentals=fund,
            analog=an,
            ml_alpha=ml,
            sec_info=sec_info,
            regime_score=float(regime.get("regime_score", 50.0)),
        )
        ticket = build_trade_ticket(
            symbol=sym,
            tech=t,
            composite=comp,
            fundamentals=fund,
            account_cfg=account_cfg,
            regime=regime,
            sector_key=sec_key,
            current_sector_weight_pct=sector_wt_by_key.get(sec_key, 0.0),
        )
        stock_analyses[sym] = {
            **t,
            "name": meta.get("name", sym),
            "resolved_symbol": resolved_map.get(sym, sym),
            "sector": sec_key,
            "sector_label": sec_label,
            "role": "holding" if sym in holding_syms else str(meta.get("role", "watchlist")),
            "notes": meta.get("notes", ""),
            "fundamentals": fund,
            "sec": sec_info,
            "news": news_map.get(sym) or [],
            "analog": an,
            "ml_alpha": ml,
            **comp,
            "trade_ticket": ticket,
        }

    sector_rankings = rank_within_sectors(stock_analyses, u_cfg)
    holdings_diagnosis = diagnose_active_holdings(
        portfolio_cfg=p_cfg,
        stock_analyses=stock_analyses,
        regime=regime,
        risk_mc=risk_mc,
    )

    # --- Stage 7: [GPU Engine #4] Vectorized Grid Backtest ---
    if run_backtest:
        _emit(0.76, f"[7/8] [{dev_profile.backend.upper()}] 运行 GPU 288 组多参数网格策略回测与稳健性检验...")
        backtest_res = run_gpu_grid_backtest(
            ohlcv_map=ohlcv_map,
            symbols=all_equities,
            lookback_years=6,
            cost_bps_per_side=10.0,
            max_positions=5,
            device_pref=cfg.compute_device,
        )
    else:
        backtest_res = {"available": False, "reason": "本次跳过回测模块"}

    # CRITICAL VRAM HANDOFF: Release PyTorch CUDA allocations before loading Ollama 14B/32B on 16GB 5070 Ti!
    release_gpu_memory()

    # --- Stage 8: [RTX 5070 Ti Local LLM] Multi-Agent Deep Research ---
    if run_llm:
        _emit(0.82, f"[8/8] 显存已释放并移交本地 LLM ({llm_model})，启动多角色投研团队...")
        llm_client = LocalLLMClient(
            base_url=llm_base_url,
            api_key=cfg.llm_api_key,
            model=llm_model,
            timeout_sec=cfg.llm_timeout_sec,
            temperature=cfg.llm_temperature,
        )
        llm_res = run_local_llm_research_suite(
            client=llm_client,
            regime=regime,
            stock_analyses=stock_analyses,
            holdings_diagnosis=holdings_diagnosis,
            lead_lag=lead_lag,
            rotation_pairs=rotation_pairs,
            risk_mc=risk_mc,
            sec_map=sec_map,
            news_map=news_map,
            lang=lang,
            progress_cb=lambda m: _emit(0.90, m),
        )
    else:
        llm_res = {
            "available": False,
            "model": llm_model,
            "error": "本次未勾选本地 AI 大模型深度解读（纯 GPU 量化模式）",
            "supply_chain_report": {},
            "stock_reports": {},
            "pm_weekly_memo": "",
        }

    # Attach per-stock LLM verdicts back into stock_analyses for seamless UI rendering
    for sym, rep in (llm_res.get("stock_reports") or {}).items():
        if sym in stock_analyses:
            stock_analyses[sym]["llm_report"] = rep

    elapsed_total = round(time.perf_counter() - t_start, 1)

    payload: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": elapsed_total,
        "device_profile": {
            "backend": dev_profile.backend,
            "summary": dev_profile.summary_str(),
            "is_blackwell": dev_profile.is_blackwell,
            "vram_total_gb": dev_profile.vram_total_gb,
        },
        "regime": regime,
        "stocks": stock_analyses,
        "holdings_diagnosis": holdings_diagnosis,
        "sector_rankings": sector_rankings,
        "lead_lag": lead_lag,
        "rotation_pairs": rotation_pairs,
        "analog_meta": {
            "backend": analog_res.get("backend"),
            "total_historical_patterns": analog_res.get("total_historical_patterns"),
            "elapsed_ms": analog_res.get("elapsed_ms"),
        },
        "ml_alpha": {
            k: v for k, v in ml_res.items() if k != "per_symbol"
        },
        "risk_mc": risk_mc,
        "backtest": backtest_res,
        "llm": llm_res,
    }

    # Save snapshot & compute diff vs previous run
    snap_id = db.save_snapshot(
        payload=payload,
        run_label="full_scan",
        device_info=dev_profile.summary_str(),
        llm_model=llm_model if run_llm else "quant_only",
    )
    payload["snapshot_id"] = snap_id
    prev_snap = db.get_previous_snapshot(snap_id)
    payload["weekly_diff"] = compute_snapshot_diff(payload, (prev_snap or {}).get("payload"))

    # Export standalone offline Markdown report
    report_path = export_markdown_report(payload, cfg.reports_dir)
    payload["report_path"] = str(report_path)

    _emit(1.0, f"全流程完成！耗时 {elapsed_total:.1f}s，已归档快照 #{snap_id} 并导出离线报告。")
    return payload


def run_on_demand_single_ticker_check(
    symbol: str,
    *,
    settings: AppSettings | None = None,
    portfolio_cfg: dict[str, Any] | None = None,
    run_llm: bool = True,
    llm_model_override: str | None = None,
    llm_base_url_override: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Instant deep-dive checkup on ANY US ticker (even outside the configured watchlist)."""
    sym = symbol.upper().strip()
    cfg = settings or load_settings()
    u_cfg = load_universe()
    p_cfg = portfolio_cfg or load_portfolio()
    db = TradingDB(cfg.db_path)

    md = MarketDataFetcher(cfg.data_dir, history_years=cfg.history_years)
    df, resolved = md.fetch_one(sym)
    if df.empty:
        return {"symbol": sym, "error": f"未能获取到 {sym} 的行情数据，请检查美股代码是否正确。"}

    qqq_df, _ = md.fetch_one("QQQ")
    soxx_df, _ = md.fetch_one("SOXX")
    spy_df, _ = md.fetch_one("SPY")
    vix_df, _ = md.fetch_one("^VIX")

    ohlcv_mini = {sym: df}
    if not qqq_df.empty:
        ohlcv_mini["QQQ"] = qqq_df
    if not soxx_df.empty:
        ohlcv_mini["SOXX"] = soxx_df
    if not spy_df.empty:
        ohlcv_mini["SPY"] = spy_df
    if not vix_df.empty:
        ohlcv_mini["^VIX"] = vix_df

    regime = evaluate_market_regime(ohlcv_mini, [sym])
    fund = FundamentalFetcher(db=db).fetch_one(sym, resolved_symbol=resolved)
    sec = SecEdgarClient(user_agent=cfg.sec_user_agent, db=db).fetch_company_filings(sym)
    news = NewsAggregator(db=db).fetch_one(sym, resolved_symbol=resolved, max_items=10)

    tech = analyze_stock_technicals(
        symbol=sym,
        df=df,
        qqq_df=qqq_df if not qqq_df.empty else None,
        sector_bench_df=soxx_df if not soxx_df.empty else None,
        fundamental_score=float(fund.get("fundamental_score", 50.0)),
    )

    analog_res = run_gpu_historical_analog_search(
        ohlcv_mini, [sym], window=30, top_k=40, device_pref=cfg.compute_device
    )
    analog = (analog_res.get("per_symbol") or {}).get(sym)

    comp = compute_composite_swing_score(
        tech=tech,
        fundamentals=fund,
        analog=analog,
        ml_alpha=None,
        sec_info=sec,
        regime_score=float(regime.get("regime_score", 50.0)),
    )
    ticket = build_trade_ticket(
        symbol=sym,
        tech=tech,
        composite=comp,
        fundamentals=fund,
        account_cfg=p_cfg.get("account") or {},
        regime=regime,
        sector_key="custom",
    )

    result = {
        **tech,
        "name": (u_cfg.get("universe") or {}).get(sym, {}).get("name", sym),
        "resolved_symbol": resolved,
        "sector": "custom",
        "sector_label": "自定义单股体检",
        "role": "custom_check",
        "fundamentals": fund,
        "sec": sec,
        "news": news,
        "analog": analog,
        **comp,
        "trade_ticket": ticket,
    }

    release_gpu_memory()

    if run_llm:
        client = LocalLLMClient(
            base_url=llm_base_url_override or cfg.llm_base_url,
            api_key=cfg.llm_api_key,
            model=llm_model_override or cfg.llm_model,
            timeout_sec=cfg.llm_timeout_sec,
            temperature=cfg.llm_temperature,
        )
        if client.check_health().get("online"):
            result["llm_report"] = analyze_single_stock_with_llm(
                client, sym, result, sec, news, lang=lang
            )
        else:
            result["llm_report"] = {
                "error": (
                    f"本地 Ollama ({client.base_url}) 未启动，仅展示硬核量化与 SEC 数据"
                    if lang == "zh"
                    else f"Local Ollama ({client.base_url}) offline — displaying quant & SEC data only."
                )
            }

    return result


def compute_snapshot_diff(
    curr: dict[str, Any],
    prev: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Compute score & signal deltas compared with the previous saved run snapshot."""
    if not prev:
        return []
    curr_stocks = curr.get("stocks") or {}
    prev_stocks = prev.get("stocks") or {}
    diffs: list[dict[str, Any]] = []
    for sym, cdata in curr_stocks.items():
        if cdata.get("error") or sym not in prev_stocks or prev_stocks[sym].get("error"):
            continue
        pdata = prev_stocks[sym]
        c_sc = float(cdata.get("composite_score", 50.0))
        p_sc = float(pdata.get("composite_score", 50.0))
        delta = round(c_sc - p_sc, 1)
        c_px = float(cdata.get("close", 0.0))
        p_px = float(pdata.get("close", 0.0))
        px_chg = round((c_px / p_px - 1.0) * 100.0, 2) if p_px > 0 else 0.0
        diffs.append(
            {
                "symbol": sym,
                "prev_score": p_sc,
                "curr_score": c_sc,
                "score_delta": delta,
                "price_change_since_last_pct": px_chg,
                "prev_signal": (pdata.get("trade_ticket") or {}).get("action_signal", ""),
                "curr_signal": (cdata.get("trade_ticket") or {}).get("action_signal", ""),
            }
        )
    diffs.sort(key=lambda x: abs(x["score_delta"]), reverse=True)
    return diffs


def export_markdown_report(payload: dict[str, Any], reports_dir: Path) -> Path:
    """Export a clean offline Chinese Markdown report so you can shut down the PC and read anywhere."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = reports_dir / f"weekly_trading_report_{ts}.md"

    regime = payload.get("regime") or {}
    risk_mc = payload.get("risk_mc") or {}
    llm = payload.get("llm") or {}
    sc_rep = llm.get("supply_chain_report") or {}

    lines: list[str] = [
        f"# AI & 存储产业链 · 美股波段投研周报 ({payload.get('generated_at')})",
        "",
        f"> 计算硬件: `{payload.get('device_profile', {}).get('summary', '')}` | 本地 AI 模型: `{llm.get('model', 'N/A')}` | 总耗时: `{payload.get('elapsed_sec')}s`",
        "",
        "## 一、大盘与半导体温度计",
        f"- **综合温度**: **{regime.get('regime_score')}/100** — {regime.get('regime_label')}",
        f"- **建议最大股票仓位**: `{regime.get('suggested_max_exposure_pct')}%` | **建议现金/防御(MCD)比例**: `{regime.get('suggested_defensive_cash_pct')}%`",
        f"- **总控指南**: {regime.get('action_guide')}",
        "",
    ]
    for sig in regime.get("signals") or []:
        lines.append(f"  - {sig}")

    lines.extend(["", "## 二、当前持仓诊断 (GOOG / STX / MCD / AMZN / ORCL / SKHY)", ""])
    lines.append("| 代码 | 现价 | 仓位% | 综合分 | 止损线 | 风险贡献% | 财报风控 | 建议指令 |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |")
    for h in payload.get("holdings_diagnosis") or []:
        lines.append(
            f"| **{h.get('symbol')}** | ${h.get('close', '-') } | {h.get('weight_pct', '-')}% | "
            f"{h.get('composite_score', '-')} | ${h.get('active_stop', '-')} | "
            f"{h.get('risk_contribution_pct', '-')}% | {h.get('earnings_flag', '-')} | {h.get('action', '-')} |"
        )

    lines.extend(["", "## 三、全股票池 Top 机会与交易计划", ""])
    stocks = payload.get("stocks") or {}
    ranked = sorted(
        [s for s in stocks.values() if not s.get("error")],
        key=lambda x: float(x.get("composite_score", 0.0)),
        reverse=True,
    )
    lines.append("| 代码 | 赛道 | 现价 | 综合分 | GPU历史21D胜率 | 暴跌/形态 | 入场区 | 硬止损 | 目标1(R:R) | 建议信号 |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | :--- | :--- | ---: | :--- | :--- |")
    for s in ranked:
        tk = s.get("trade_ticket") or {}
        an = s.get("analog") or {}
        cr = s.get("crash_analysis") or {}
        lines.append(
            f"| **{s.get('symbol')}** | {s.get('sector')} | ${s.get('close')} | **{s.get('composite_score')}** | "
            f"{an.get('win_rate_21d_pct', '-')}% | {cr.get('badge', '')} | "
            f"${tk.get('entry_low')}-${tk.get('entry_high')} | ${tk.get('hard_stop')} | "
            f"${tk.get('target_1')} ({tk.get('rr_target_1')}R) | {tk.get('action_signal')} |"
        )

    if risk_mc.get("available"):
        h21 = risk_mc.get("horizon_21d") or {}
        lines.extend(
            [
                "",
                "## 四、GPU 10万路径 Student-t 组合风险与压力测试",
                f"- 总资产(含现金): `${risk_mc.get('total_nav_usd', 0):,.0f}` | 股票敞口: `{risk_mc.get('gross_exposure_pct')}%` | 组合年化波动率: `{risk_mc.get('portfolio_vol_ann_pct')}%`",
                f"- **21日 (1个月) 95% VaR**: `{h21.get('var_95_pct')}%` (`${h21.get('var_95_usd', 0):,.0f}`) | **95% CVaR (尾部均值)**: `{h21.get('cvar_95_pct')}%` (`${h21.get('cvar_95_usd', 0):,.0f}`)",
                f"- 未来1个月回撤超过 10% 的概率: `{h21.get('prob_loss_gt_10pct')}%` | 涨幅超过 10% 的概率: `{h21.get('prob_gain_gt_10pct')}%`",
                "",
            ]
        )
        for st in risk_mc.get("stress_tests") or []:
            lines.append(
                f"  - **{st['scenario']}**: 预计组合影响 `{st['est_portfolio_impact_pct']}%` (`${st['est_portfolio_impact_usd']:,.0f}`)，最大拖累: `{st['largest_drag_symbol']}`"
            )

    if sc_rep and not sc_rep.get("error"):
        lines.extend(
            [
                "",
                "## 五、5070 Ti 本地 AI · 产业链首席策略师研判",
                f"- **产业链冷暖**: {sc_rep.get('chain_temperature', '')}",
                f"- **云巨头 CapEx 动向**: {sc_rep.get('capex_verdict', '')}",
                f"- **内存/存储周期**: {sc_rep.get('storage_cycle_verdict', '')}",
                f"- **首选细分赛道**: {sc_rep.get('top_sector_pick', '')}",
            ]
        )

    if llm.get("pm_weekly_memo"):
        lines.extend(["", "## 六、基金经理《本周波段交易执行计划》", "", llm["pm_weekly_memo"]])

    lines.extend(["", "---", "*免责声明：本报告由本地量化与开源大模型自动生成，仅供投研辅助与风险管理参考，不构成任何具体证券买卖建议。*"])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
