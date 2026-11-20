"""Bilingual i18n dictionary & dynamic quant-badge translator (Default: EN, Toggle: ZH)."""

from __future__ import annotations

from typing import Any

# ==============================================================================
# Core UI Strings (en = Default English Broker Terminal, zh = 中文专业终端)
# ==============================================================================
STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "page_title": "Front-Running",
        "brand_title": "FRONT-RUNNING PRO",
        "brand_sub": "AI & STORAGE SUPPLY-CHAIN EQUITY DESK",
        "lang_slider_label": "EN ⇄ 中文",
        "lang_badge": "🇺🇸 EN",
        "sidebar_title": "5070 Ti ENGINE",
        "sidebar_caption": "Local GPU Quant + LLM Workstation",
        "gpu_ready": "CUDA Active",
        "gpu_cpu_mode": "Compute Backend",
        "blackwell_note": "Blackwell sm_120 detected · Auto VRAM handoff enabled",
        "llm_section": "LOCAL LLM (5070 Ti)",
        "llm_url": "Ollama / vLLM Endpoint",
        "llm_model": "Active Model",
        "llm_online": "LLM Online",
        "llm_offline": "Ollama Offline (Quant-only mode still works; run `ollama serve` in WSL for AI research)",
        "scan_section": "RUN ENGINE (ON-DEMAND)",
        "opt_refresh": "Force refresh quotes, SEC & news",
        "opt_ml": "GPU Walk-Forward ML Alpha",
        "opt_bt": "GPU 288-Combo Grid Backtest",
        "opt_llm": "5070 Ti Multi-Agent AI Research",
        "btn_run_scan": "⚡ RUN FULL SCAN",
        "scan_done": "Scan complete in {sec}s. Snapshot archived — safe to shut down PC after review.",
        "scan_failed": "Pipeline error",
        "snap_section": "SAVED SNAPSHOTS",
        "snap_select": "Load archived run (instant offline)",
        "btn_load_snap": "Load Snapshot",
        "btn_download_md": "Export Report (.md)",
        "no_snap_banner": "Click **⚡ RUN FULL SCAN** in the sidebar to pull 10Y market history, run RTX 5070 Ti GPU quant/risk/backtest engines, and generate your AI trading brief.",
        "snap_meta": "Snapshot `{ts}` · Engine `{dev}` · LLM `{llm}` · `{sec}s`",
        # Tabs
        "tab_desk": "📈 Trade Desk",
        "tab_scanner": "🎯 Signals & Regime",
        "tab_portfolio": "💼 Positions & Risk",
        "tab_chain": "📡 Supply Chain",
        "tab_backtest": "🧪 Backtest & ML",
        "tab_grid": "🕸️ Grid Lab",
        "tab_journal": "📓 Trade Journal",
        "tab_guide": "⚙️ Desk Guide",
        # Grid Lab (4-Stage RTX 5070 Ti GPU Grid Suite)
        "grid_title": "GPU Grid Trading Lab (Hurst · 60k 4D Hyper-Cube · 50k Black-Swan MC · ML Switch)",
        "grid_s1_hdr": "Stage 1 · Watchlist Grid Suitability Screener (PyTorch CUDA)",
        "grid_s1_btn": "⚡ Scan Watchlist for Grid Suitability",
        "grid_s2_hdr": "Stage 2 · 4D Hyper-Cube Tensor Optimization & SciPy Gaussian Plateau",
        "grid_s3_hdr": "Stage 3 · 10,000-Path Black-Swan Monte Carlo Stress Test (50,000 Sims)",
        "grid_s4_hdr": "Stage 4 · XGBoost/LightGBM 5-Day Crash Switch & Live Order Monitor",
        "grid_run_deep": "🚀 Run Stages 2–4 GPU Lab",
        "grid_pick_sym": "Target Stock for Stages 2–4",
        "grid_budget": "Grid Capital ($)",
        "grid_held": "Grid Shares Held",
        "grid_break_lim": "Max Break Rate (%)",
        "grid_fidelity": "5070 Ti Precision",
        "grid_ladder_hdr": "Today's Pre-Market Limit Order Ladder",
        # Trade Desk (3-column Broker View)
        "watchlist_hdr": "WATCHLIST",
        "any_ticker_ph": "Lookup any US ticker (e.g. PLTR, ANET)...",
        "btn_lookup": "Lookup",
        "lookup_spinner": "Running GPU Quant + SEC + News + 5070 Ti AI checkup on {sym}...",
        "timeframe_lbl": "Range",
        "order_ticket_hdr": "EXECUTION TICKET",
        "advisory_only": "SPOT EQUITY · MANUAL EXEC",
        "lbl_score": "Conviction Score",
        "lbl_signal": "Desk Signal",
        "lbl_entry_zone": "Entry Zone",
        "lbl_hard_stop": "Hard Stop-Loss",
        "lbl_trail_stop": "Trailing Stop",
        "lbl_target1": "Target 1 (2R)",
        "lbl_target2": "Target 2 (3R)",
        "lbl_pos_size": "Suggested Size",
        "lbl_max_risk": "Planned Risk",
        "lbl_earnings": "Earnings Guard",
        "lbl_52w": "52W Range",
        "btn_quick_log": "📝 Log Setup to Journal",
        "quick_log_ok": "Logged {sym} plan into Trade Journal.",
        "sub_quant": "Quant & Crash Guard",
        "sub_analog": "GPU 10Y Analogs",
        "sub_ai": "5070 Ti AI Debate",
        "sub_sec": "SEC Filings & News",
        "short_hist_warn": "{sym} has limited history ({bars} daily bars). Treat long-term MAs & analogs as tentative; prioritize price action & catalysts.",
        "crash_title": "Dip Buy vs. Falling Knife Classifier",
        "knife_score_lbl": "Falling Knife Risk",
        "penalties_lbl": "Risk Adjustments",
        "ai_verdict_lbl": "AI Verdict",
        "ai_conf_lbl": "Confidence",
        "ai_crash_lbl": "Drawdown Cause",
        "ai_summary_lbl": "Desk Summary",
        "ai_bull_hdr": "Bull Thesis (1–2M)",
        "ai_bear_hdr": "Bear Risks & Red Flags",
        "ai_cat_hdr": "Key Catalysts (1–2M)",
        "ai_plan_hdr": "Execution Plan",
        "ai_unavailable": "Local AI report not generated for this ticker yet (enable Ollama in sidebar or run Lookup).",
        "analog_hdr": "Top 5 Historical Precedents (GPU 10Y Tensor Search)",
        "analog_empty": "Insufficient trading history for 10Y analog matching.",
        "sec_hdr": "Official SEC EDGAR Filings & News Wire",
        # Tab 2: Signals & Regime
        "regime_exposure": "Max Equity Exposure",
        "regime_cash": "Cash / MCD Hedge",
        "regime_vix": "VIX Volatility",
        "regime_us10y": "US 10Y Yield",
        "regime_guide_lbl": "Desk Playbook",
        "pm_memo_hdr": "🤖 5070 Ti Portfolio Manager · Weekly Execution Brief",
        "diff_hdr": "Score Movers vs. Previous Snapshot",
        "screener_hdr": "Full Universe Swing Scorecard & Trade Plans",
        # Tab 3: Portfolio & Risk
        "pos_hdr": "Active Positions Book (GOOG / STX / MCD / AMZN / ORCL / SKHY)",
        "edit_pos_exp": "Edit Account Equity, Cash & Holdings (Persists to portfolio.yaml)",
        "lbl_equity": "Total Account Equity (USD)",
        "lbl_cash": "Available Cash (USD)",
        "lbl_risk_pct": "Max Risk / Trade (%)",
        "lbl_max_pos": "Max Single Position (%)",
        "btn_save_pos": "Save Portfolio Config",
        "saved_pos_ok": "Saved to config/portfolio.yaml. Click ⚡ RUN FULL SCAN to refresh Monte Carlo & risk metrics.",
        "mc_hdr": "GPU Student-t Fat-Tail Monte Carlo Risk ({sims:,} Paths · df={df} · {ms}ms)",
        "mc_gross": "Gross Exposure",
        "mc_vol": "Annualized Vol",
        "mc_var": "21D 95% VaR",
        "mc_cvar": "21D 95% CVaR (Tail)",
        "mc_pdd": "42D P(Loss > 10%)",
        "stress_hdr": "Institutional Macro & AI Stress Tests",
        "sector_wt_hdr": "Sector Exposure Breakdown",
        # Tab 4: Supply Chain
        "sc_ai_hdr": "5070 Ti Supply-Chain Strategist",
        "sc_temp": "Chain Temperature",
        "sc_capex": "Hyperscaler CapEx Cycle",
        "sc_storage": "HBM / DRAM / NAND / HDD Cycle",
        "sc_leadlag_ops": "Lead-Lag Rotation Opportunities",
        "sc_redflags": "Supply-Chain Red Flags",
        "sc_top_pick": "Top 1–2M Sector & Leader Pick",
        "peer_rank_hdr": "Intra-Sector Peer Rankings (Best-in-Class Selection)",
        "pair_hdr": "Pair Spread Z-Score Monitor (Rotation Radar)",
        "ll_hdr": "Upstream → Downstream Lead-Lag Matrix",
        # Tab 5: Backtest & ML
        "bt_hdr": "GPU 288-Combo Grid Backtest (6Y Watchlist Validation)",
        "bt_cagr": "CAGR",
        "bt_sharpe": "Sharpe Ratio",
        "bt_sortino": "Sortino Ratio",
        "bt_mdd": "Max Drawdown",
        "bt_win": "Win Rate",
        "bt_pf": "Profit Factor",
        "ml_hdr": "Purged Walk-Forward GPU ML Validation (Out-of-Sample Honesty Check)",
        "ml_samples": "OOS Samples",
        "ml_auc": "OOS ROC-AUC",
        "ml_hit": "Top-Quintile Hit Rate",
        "ml_spread": "Top vs Bottom 21D Spread",
        "ml_feats": "Top Predictive Features (XGBoost CUDA)",
        # Tab 6: Journal
        "jrn_hdr": "Execution Ledger & Post-Mortem Trade Journal",
        "jrn_sym": "Ticker",
        "jrn_act": "Side",
        "jrn_sh": "Shares",
        "jrn_px": "Fill Price ($)",
        "jrn_conv": "Conviction (1–5)",
        "jrn_stop": "Stop-Loss ($)",
        "jrn_tgt": "Target ($)",
        "jrn_tag": "Setup Tag",
        "jrn_notes": "Execution Thesis & Discipline Notes",
        "jrn_submit": "Record Trade",
        "jrn_saved": "Trade logged for {sym}.",
        "jrn_empty": "No trades logged yet. Log each manual fill to audit your real-world win rate & stop discipline.",
        # Tab 7: Guide
        "guide_hdr": "RTX 5070 Ti (WSL2) Architecture & Weekly 10-Min Broker Workflow",
        # Column headers
        "col_sym": "Ticker",
        "col_role": "Role",
        "col_sector": "Sector",
        "col_price": "Last",
        "col_1d": "1D %",
        "col_21d": "21D %",
        "col_dd52": "From 52W Hi",
        "col_score": "Score",
        "col_analog_wr": "10Y 21D Win%",
        "col_ml_prob": "ML Beat%",
        "col_crash": "Setup / Knife Risk",
        "col_earn": "Earnings",
        "col_entry": "Entry Zone",
        "col_stop": "Hard Stop",
        "col_t1": "Target 1 (R:R)",
        "col_shares": "Rec. Shs",
        "col_signal": "Signal",
        "col_ai": "AI Verdict",
        "col_weight": "Weight %",
        "col_pnl": "P&L %",
        "col_trail": "Trail Stop",
        "col_dist_stop": "To Stop %",
        "col_rc": "Risk Share %",
        "col_action": "Directive",
        "col_rationale": "Rationale",
        "col_rank": "Rank",
        "col_name": "Name",
        "col_rs_sec": "21D RS vs Sec",
        "col_stage": "Stage",
    },
    "zh": {
        "page_title": "Front-Running",
        "brand_title": "FRONT-RUNNING PRO",
        "brand_sub": "AI 算力 & 存储产业链 · 美股现货波段交易终端",
        "lang_slider_label": "EN ⇄ 中文",
        "lang_badge": "🇨🇳 中文",
        "sidebar_title": "5070 Ti 算力中心",
        "sidebar_caption": "本地 GPU 量化 + 大模型工作站",
        "gpu_ready": "CUDA 已就绪",
        "gpu_cpu_mode": "当前计算后端",
        "blackwell_note": "已识别 Blackwell sm_120 · 显存自动接力防 OOM 已开启",
        "llm_section": "本地 5070 Ti 大模型",
        "llm_url": "Ollama / vLLM 接口地址",
        "llm_model": "本地推理模型",
        "llm_online": "推理服务在线",
        "llm_offline": "Ollama 未启动（纯 GPU 量化仍可用；如需 AI 排雷请在 WSL 运行 `ollama serve`）",
        "scan_section": "随用随开 · 一键全量引擎",
        "opt_refresh": "强制刷新行情、SEC 公告与新闻",
        "opt_ml": "GPU Walk-Forward 机器学习",
        "opt_bt": "GPU 288 组参数网格回测",
        "opt_llm": "5070 Ti 本地多角色 AI 研报",
        "btn_run_scan": "⚡ 一键启动全量分析",
        "scan_done": "全量分析完成，耗时 {sec} 秒。已归档快照，看完报告即可关机。",
        "scan_failed": "运行失败",
        "snap_section": "历史归档快照",
        "snap_select": "加载历史快照（离线秒开）",
        "btn_load_snap": "切换至该快照",
        "btn_download_md": "导出离线周报 (.md)",
        "no_snap_banner": "请点击左侧 **⚡ 一键启动全量分析**，系统将拉取 10 年历史数据、调用 RTX 5070 Ti 完成量化/蒙特卡洛/回测与本地 AI 排雷。",
        "snap_meta": "快照时间 `{ts}` · 后端 `{dev}` · 本地模型 `{llm}` · 耗时 `{sec}s`",
        # Tabs
        "tab_desk": "📈 交易工作台",
        "tab_scanner": "🎯 信号与大盘",
        "tab_portfolio": "💼 持仓与风控",
        "tab_chain": "📡 产业链轮动",
        "tab_backtest": "🧪 回测与 ML",
        "tab_grid": "🕸️ 网格实验室",
        "tab_journal": "📓 交易日志",
        "tab_guide": "⚙️ 使用指南",
        # Grid Lab
        "grid_title": "RTX 5070 Ti 四阶段 GPU 网格实验室 (赫斯特体检 · 6万组4D高原 · 5万次黑天鹅压测 · ML智能开关)",
        "grid_s1_hdr": "阶段一 · 自选池网格适配度客观体检 (PyTorch CUDA 并行)",
        "grid_s1_btn": "⚡ 一键扫描 Watchlist 网格适配度",
        "grid_s2_hdr": "阶段二 · 4D 参数超立方体 GPU 搜索 & SciPy 高斯参数高原",
        "grid_s3_hdr": "阶段三 · 10,000 平行宇宙黑天鹅蒙特卡洛压力测试 (共 50,000 次模拟)",
        "grid_s4_hdr": "阶段四 · XGBoost/LightGBM 单边暴跌智能开关 & 实盘挂单监控台",
        "grid_run_deep": "🚀 运行阶段二~四 GPU 深度优化",
        "grid_pick_sym": "选择深测标的 (阶段二~四)",
        "grid_budget": "网格专款预算 ($)",
        "grid_held": "当前网格持股数",
        "grid_break_lim": "破网率底线 (%)",
        "grid_fidelity": "5070 Ti 算力档位",
        "grid_ladder_hdr": "今日盘前实盘挂单点位梯 (直接照单挂限价单)",
        # Trade Desk
        "watchlist_hdr": "自选行情报价",
        "any_ticker_ph": "输入任意美股代码即时体检 (如 PLTR, ANET)...",
        "btn_lookup": "即时体检",
        "lookup_spinner": "正在对 {sym} 执行 GPU 量化 + SEC + 新闻 + 5070 Ti 本地 AI 体检...",
        "timeframe_lbl": "周期",
        "order_ticket_hdr": "交易执行单 (参考)",
        "advisory_only": "美股现货 · 手动执行参考",
        "lbl_score": "波段综合分",
        "lbl_signal": "系统信号",
        "lbl_entry_zone": "参考入场区",
        "lbl_hard_stop": "硬止损位",
        "lbl_trail_stop": "移动止盈线",
        "lbl_target1": "目标位 1 (2R)",
        "lbl_target2": "目标位 2 (3R)",
        "lbl_pos_size": "建议买入仓位",
        "lbl_max_risk": "计划最大止损额",
        "lbl_earnings": "财报风控",
        "lbl_52w": "52周区间",
        "btn_quick_log": "📝 一键记入交易日志",
        "quick_log_ok": "已将 {sym} 计划写入交易复盘日志。",
        "sub_quant": "量价与排雷",
        "sub_analog": "GPU 10年相似走势",
        "sub_ai": "5070 Ti AI 多空辩论",
        "sub_sec": "SEC 公告与新闻",
        "short_hist_warn": "{sym} 历史交易日较少（仅 {bars} 根日线），长周期均线与历史统计仅供参考，请重点结合量价结构与催化剂判断。",
        "crash_title": "错杀抄底 vs 接飞刀量化识别",
        "knife_score_lbl": "飞刀风险指数",
        "penalties_lbl": "风控扣分项",
        "ai_verdict_lbl": "AI 结论",
        "ai_conf_lbl": "置信度",
        "ai_crash_lbl": "下跌归因",
        "ai_summary_lbl": "一句话总结",
        "ai_bull_hdr": "多头核心逻辑 (1~2个月)",
        "ai_bear_hdr": "空头与排雷风险",
        "ai_cat_hdr": "关键催化剂 (1~2个月)",
        "ai_plan_hdr": "实战操作建议",
        "ai_unavailable": "本次尚未对该标的生成本地 AI 报告（请在左侧启用 Ollama 或点击上方即时体检）。",
        "analog_hdr": "GPU 10年历史最相似走势先例 (Top 5)",
        "analog_empty": "历史数据不足，暂无 10 年相似形态匹配。",
        "sec_hdr": "SEC EDGAR 官方披露与近期新闻",
        # Tab 2
        "regime_exposure": "建议最大股票仓位",
        "regime_cash": "现金 / MCD 防御",
        "regime_vix": "恐慌指数 VIX",
        "regime_us10y": "10Y 美债利率",
        "regime_guide_lbl": "总控交易指南",
        "pm_memo_hdr": "🤖 5070 Ti 本地基金经理 · 《本周波段交易执行计划》",
        "diff_hdr": "较上次快照评分异动 Top 6",
        "screener_hdr": "全股票池波段量化评分与交易计划总表",
        # Tab 3
        "pos_hdr": "当前持仓诊断 (GOOG / STX / MCD / AMZN / ORCL / SKHY)",
        "edit_pos_exp": "编辑账户资金、现金与实际持仓股数/成本价（自动保存至 portfolio.yaml）",
        "lbl_equity": "账户总资金 (USD)",
        "lbl_cash": "可用现金 (USD)",
        "lbl_risk_pct": "单笔最大止损风险 (%)",
        "lbl_max_pos": "单股最大仓位上限 (%)",
        "btn_save_pos": "保存持仓与风控配置",
        "saved_pos_ok": "已保存至 config/portfolio.yaml，请点击左侧「⚡ 一键启动全量分析」刷新风险模拟。",
        "mc_hdr": "GPU Student-t 肥尾蒙特卡洛风险模拟 ({sims:,} 路径 · df={df} · {ms}ms)",
        "mc_gross": "股票总敞口",
        "mc_vol": "组合年化波动率",
        "mc_var": "1个月 95% VaR",
        "mc_cvar": "1个月 95% CVaR(尾部)",
        "mc_pdd": "2个月回撤>10%概率",
        "stress_hdr": "机构级极端情景压力测试",
        "sector_wt_hdr": "赛道敞口分布",
        # Tab 4
        "sc_ai_hdr": "5070 Ti 产业链首席策略师研判",
        "sc_temp": "产业链冷暖",
        "sc_capex": "上游云巨头 CapEx 动向",
        "sc_storage": "HBM / DRAM / NAND / HDD 存储周期",
        "sc_leadlag_ops": "滞后补涨 / 轮动机会",
        "sc_redflags": "产业链风险警报",
        "sc_top_pick": "未来 1~2 个月首选赛道与龙头",
        "peer_rank_hdr": "同赛道“优中选优”横向排行榜",
        "pair_hdr": "同赛道配对价差偏离 (换仓雷达)",
        "ll_hdr": "上下游传导 Lead-Lag 统计",
        # Tab 5
        "bt_hdr": "GPU 288 组多参数网格回测（6年历史真数据验真）",
        "bt_cagr": "年化收益 CAGR",
        "bt_sharpe": "夏普比率 Sharpe",
        "bt_sortino": "索提诺 Sortino",
        "bt_mdd": "最大回撤 MaxDD",
        "bt_win": "单笔胜率",
        "bt_pf": "盈亏比 PF",
        "ml_hdr": "GPU 严格样本外 Walk-Forward 机器学习验真",
        "ml_samples": "样本外验证数",
        "ml_auc": "样本外 ROC-AUC",
        "ml_hit": "Top 20% 真实跑赢率",
        "ml_spread": "Top vs Bottom 21日超额差",
        "ml_feats": "核心预测特征 Top 8 (XGBoost GPU)",
        # Tab 6
        "jrn_hdr": "美股波段交易与复盘日志",
        "jrn_sym": "美股代码",
        "jrn_act": "操作方向",
        "jrn_sh": "股数",
        "jrn_px": "成交价 ($)",
        "jrn_conv": "把握度 (1-5)",
        "jrn_stop": "硬止损 ($)",
        "jrn_tgt": "目标价 ($)",
        "jrn_tag": "形态标签",
        "jrn_notes": "交易逻辑与风控复盘备注",
        "jrn_submit": "写入交易日志",
        "jrn_saved": "已记录 {sym} 交易日志。",
        "jrn_empty": "暂无交易记录，建议每次手动下单后顺手记一笔以便复盘提升胜率。",
        # Tab 7
        "guide_hdr": "RTX 5070 Ti (WSL2) 算力架构与每周 10 分钟实战流程",
        # Columns
        "col_sym": "代码",
        "col_role": "角色",
        "col_sector": "赛道",
        "col_price": "现价",
        "col_1d": "1日%",
        "col_21d": "21日%",
        "col_dd52": "距52W高点%",
        "col_score": "综合分",
        "col_analog_wr": "10年21D胜率%",
        "col_ml_prob": "ML跑赢%",
        "col_crash": "暴跌/形态",
        "col_earn": "财报风控",
        "col_entry": "入场区",
        "col_stop": "硬止损",
        "col_t1": "目标1 (R:R)",
        "col_shares": "建议股数",
        "col_signal": "信号",
        "col_ai": "AI 结论",
        "col_weight": "仓位%",
        "col_pnl": "浮盈亏%",
        "col_trail": "移动止盈",
        "col_dist_stop": "距止损%",
        "col_rc": "风险贡献%",
        "col_action": "操作指令",
        "col_rationale": "核心依据",
        "col_rank": "排名",
        "col_name": "名称",
        "col_rs_sec": "21D跑赢板块%",
        "col_stage": "周期阶段",
    },
}


def t(key: str, lang: str = "en", **kwargs: Any) -> str:
    """Lookup localized UI string with English fallback."""
    bucket = STRINGS.get(lang, STRINGS["en"])
    text = bucket.get(key, STRINGS["en"].get(key, key))
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


# ==============================================================================
# Dynamic Quant Label Translator (Instant EN/ZH switching on saved snapshots)
# ==============================================================================
SECTOR_LABELS_EN: dict[str, str] = {
    "cloud_ai": "Tier 1 · Cloud & AI Hyperscalers (CapEx)",
    "compute_optical": "Tier 2 · Compute Chips & Optical",
    "memory_storage": "Tier 3 · HBM / Memory & Storage Cycle",
    "mega_disruptors": "High-Beta Mega Disruptors",
    "defensive": "Defensive Hedge (Low-Beta)",
    "custom": "Custom Ticker Lookup",
}

STAGE_EN: dict[int, str] = {
    1: "Stage 1 · Basing",
    2: "Stage 2 · Advancing",
    3: "Stage 3 · Distribution",
    4: "Stage 4 · Declining",
}

CRASH_BADGE_EN: dict[str, str] = {
    "UPTREND_HEALTHY": "🔵 Healthy Uptrend",
    "FALLING_KNIFE": "🔴 Falling Knife (Avoid)",
    "LEFT_SIDE_WATCH": "🟡 Oversold Watch (Wait Base)",
    "MISPRICED_PULLBACK": "🟢 Mispriced Dip Setup",
}

SIGNAL_MAP_EN: dict[str, str] = {
    "🔴 观望/禁接飞刀": "🔴 AVOID KNIFE",
    "🟢 优先买入候选": "🟢 PRIORITY BUY",
    "🟡 择机低吸/分批试仓": "🟡 PULLBACK BUY",
    "⚪ 跟踪观察": "⚪ WATCH",
    "🟠 偏弱回避": "🟠 UNDERWEIGHT",
}

HOLDING_ACTION_EN: dict[str, str] = {
    "🚨 触发止损线 · 建议减仓/止损": "🚨 STOP HIT · TRIM / EXIT",
    "🔴 破位警戒 · 逢反抽减仓/严守止损": "🔴 BROKEN TREND · TRIM BOUNCE",
    "🛡️ 坚守防御底仓": "🛡️ HOLD DEFENSIVE HEDGE",
    "🟢 持有底仓 (进攻期可适度置换至高分龙头)": "🟢 HOLD CORE (OPTIONAL ROTATE)",
    "🟢 强势持有 · 上移移动止盈": "🟢 HOLD STRONG · RAISE TRAIL STOP",
    "🟢 核心持有 · 逢回踩可加仓": "🟢 CORE HOLD · ADD ON DIPS",
    "🟡 持有观察 · 守住止损线": "🟡 HOLD · RESPECT STOP",
    "🟠 弱势预警 · 考虑去弱留强": "🟠 WEAK · CONSIDER ROTATING",
    "⚪ 待刷新行情": "⚪ REFRESH QUOTES",
}


def tr_sector(sector_key: str, zh_fallback: str, lang: str = "en") -> str:
    if lang == "zh":
        return zh_fallback or sector_key
    return SECTOR_LABELS_EN.get(sector_key, sector_key.replace("_", " ").title())


def tr_stage(stage_num: Any, zh_label: str, lang: str = "en") -> str:
    if lang == "zh":
        return zh_label
    try:
        return STAGE_EN.get(int(stage_num), zh_label)
    except Exception:
        return zh_label


def tr_crash_badge(crash_info: dict[str, Any] | None, lang: str = "en") -> str:
    cr = crash_info or {}
    if lang == "zh":
        return str(cr.get("badge", "-"))
    return CRASH_BADGE_EN.get(str(cr.get("status", "")), str(cr.get("badge", "-")))


def tr_signal(raw_signal: str, lang: str = "en") -> str:
    if lang == "zh":
        return raw_signal or "-"
    return SIGNAL_MAP_EN.get(raw_signal, raw_signal or "-")


def tr_holding_action(raw_action: str, lang: str = "en") -> str:
    if lang == "zh":
        return raw_action or "-"
    return HOLDING_ACTION_EN.get(raw_action, raw_action or "-")


def tr_earnings_flag(fund: dict[str, Any] | None, lang: str = "en") -> str:
    fn = fund or {}
    if lang == "zh":
        return str(fn.get("earnings_risk_flag", "待确认"))
    days = fn.get("days_to_earnings")
    if days is None:
        return "TBD"
    if days <= 10:
        return f"⚠️ HIGH RISK ({days}d)"
    if days <= 21:
        return f"🟡 SOON ({days}d)"
    return f"🟢 CLEAR ({days}d)"


def tr_regime_label(score: float, zh_label: str, lang: str = "en") -> str:
    if lang == "zh":
        return zh_label
    if score >= 75:
        return "🟢 BULL EXPANSION"
    if score >= 58:
        return "🟡 SELECTIVE BULL"
    if score >= 42:
        return "🟠 CHOPPY / DEFENSIVE"
    return "🔴 RISK-OFF / BEARISH"


def tr_regime_guide(score: float, zh_guide: str, lang: str = "en") -> str:
    if lang == "zh":
        return zh_guide
    if score >= 75:
        return "Favorable tailwinds. Stay invested in Stage-2 leaders, add on low-volume 20MA pullbacks, and trail winners."
    if score >= 58:
        return "Selective tape. Focus on top-ranked leaders with R:R >= 2:1 near support; keep 20–25% cash or MCD hedge."
    if score >= 42:
        return "Choppy tape. Avoid catching falling knives; cut broken Stage-4 laggards and maintain higher cash / MCD defense."
    return "Tech/Semi headwind. Capital preservation first — respect hard stops strictly, avoid averaging down, wait for stabilization."


def tr_role(role: str, lang: str = "en") -> str:
    if lang == "zh":
        return "💼持仓" if role == "holding" else ("📡风向标" if role == "bellwether" else "👀自选")
    return "POS" if role == "holding" else ("BELL" if role == "bellwether" else "WATCH")
