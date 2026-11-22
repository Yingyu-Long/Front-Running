"""Front-Running Pro · Broker-Grade US Equity Swing Trading Terminal (Bilingual EN/ZH, Default EN)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.ai.llm_client import LocalLLMClient
from src.config_loader import (
    get_all_equity_symbols,
    load_portfolio,
    load_settings,
    load_universe,
    save_portfolio,
    save_universe,
)
from src.data.market_data import MarketDataFetcher
from src.engine.orchestrator import (
    run_full_analysis_pipeline,
    run_on_demand_single_ticker_check,
)
from src.quant.device import get_device_profile
from src.quant.gpu_grid_lab import (
    build_live_grid_order_ladder,
    run_complete_grid_lab_for_symbol,
    screen_watchlist_for_grid,
)
from src.storage.db import TradingDB
from src.ui.charts import (
    make_backtest_equity_chart,
    make_backtest_heatmap,
    make_candlestick_trade_chart,
    make_correlation_heatmap,
    make_mc_distribution_chart,
    make_pillar_radar,
    make_regime_gauge,
    make_risk_contribution_chart,
)
from src.ui.grid_charts import (
    make_grid_mc_fan_chart,
    make_grid_suitability_scatter,
    make_live_grid_candlestick,
    make_plateau_comparison_heatmaps,
)
from src.ui.i18n import (
    t,
    tr_crash_badge,
    tr_earnings_flag,
    tr_holding_action,
    tr_regime_guide,
    tr_regime_label,
    tr_role,
    tr_sector,
    tr_signal,
    tr_stage,
)

st.set_page_config(
    page_title="Front-Running",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Broker Desktop Terminal Styling (Webull / Thinkorswim / TradingView Carbon Dark)
# ==============================================================================
BROKER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp {
    background-color: #090C10;
    color: #D9E1EC;
}
section[data-testid="stSidebar"] {
    background-color: #0D1117;
    border-right: 1px solid #1C2430;
}
.block-container {
    padding-top: 2.35rem !important;
    padding-bottom: 1.4rem !important;
    padding-left: 1.35rem !important;
    padding-right: 1.35rem !important;
    max-width: 100% !important;
}
/* Clean native broker window chrome: hide Deploy & rainbow strip, keep sidebar toggle */
div[data-testid="stDecoration"] { display: none !important; }
.stAppDeployButton { display: none !important; }
header[data-testid="stHeader"] {
    background: rgba(9, 12, 16, 0.85) !important;
    height: 2.2rem !important;
    border-bottom: none !important;
}
/* Minimalist right-aligned EN / 中文 switch */
div[data-testid="stToggle"] {
    display: flex;
    justify-content: flex-end;
    margin: 0 !important;
    padding: 0 !important;
}
div[data-testid="stToggle"] label {
    gap: 8px !important;
}
div[data-testid="stToggle"] label p {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
    color: #B8C5D6 !important;
    white-space: nowrap !important;
}
div[data-testid="stCheckbox"] label p {
    color: #D9E1EC !important;
}
/* Monospace tabular numbers for all broker KPI metrics */
div[data-testid="stMetric"] {
    background: #0F141C;
    border: 1px solid #1C2430;
    border-radius: 8px;
    padding: 10px 12px;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.74rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #7F8EA3 !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-variant-numeric: tabular-nums;
    font-size: 1.38rem !important;
    font-weight: 700 !important;
    color: #F0F6FC !important;
}
/* Sleek Broker Top Bar */
.broker-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(90deg, #0E141E 0%, #0B1018 100%);
    border: 1px solid #1C2430;
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 10px;
}
.brand-wrap {
    display: flex;
    align-items: center;
    gap: 9px;
    white-space: nowrap;
}
.live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #0ECB81;
    box-shadow: 0 0 7px rgba(14, 203, 129, 0.85);
    display: inline-block;
    flex-shrink: 0;
}
.brand-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.96rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #F0F6FC;
}
.header-divider {
    border-bottom: 1px solid #1C2430;
    margin: 4px 0 8px 0;
}
/* Live Market Tape Strip */
.ticker-tape {
    display: flex;
    flex-wrap: nowrap;
    overflow-x: auto;
    gap: 6px;
    align-items: center;
    justify-content: center;
}
.tape-chip {
    background: #0F141C;
    border: 1px solid #1B2432;
    border-radius: 5px;
    padding: 3px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    font-variant-numeric: tabular-nums;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
}
.tape-sym { color: #8B9BB4; font-weight: 600; }
.tape-px { color: #F0F6FC; font-weight: 600; }
.pos-green { color: #0ECB81; font-weight: 600; }
.neg-red { color: #F6465D; font-weight: 600; }
.neu-amber { color: #F0B90B; font-weight: 600; }
.accent-blue { color: #58A6FF; font-weight: 600; }

/* Broker Quote Banner above Chart */
.quote-banner {
    background: #0F141C;
    border: 1px solid #1C2430;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}
.qb-sym {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.45rem;
    font-weight: 700;
    color: #FFFFFF;
}
.qb-name {
    font-size: 0.85rem;
    color: #8B9BB4;
    margin-left: 8px;
}
.qb-price {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.65rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: #FFFFFF;
}
.qb-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 6px;
}
.pill-green { background: rgba(14, 203, 129, 0.16); color: #0ECB81; border: 1px solid rgba(14, 203, 129, 0.35); }
.pill-red { background: rgba(246, 70, 93, 0.16); color: #F6465D; border: 1px solid rgba(246, 70, 93, 0.35); }
.pill-amber { background: rgba(240, 185, 11, 0.16); color: #F0B90B; border: 1px solid rgba(240, 185, 11, 0.35); }
.pill-blue { background: rgba(41, 98, 255, 0.18); color: #6CA0FF; border: 1px solid rgba(41, 98, 255, 0.40); }
.pill-muted { background: #161D29; color: #8B9BB4; border: 1px solid #253041; }

/* Broker Order Ticket Panel (Right Column) */
.order-ticket {
    background: #0F141C;
    border: 1px solid #1E2837;
    border-radius: 10px;
    padding: 14px 15px;
    margin-bottom: 10px;
}
.ot-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 8px;
    margin-bottom: 10px;
    border-bottom: 1px solid #1C2430;
}
.ot-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.80rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #9FB1C8;
}
.ot-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px dashed #1A222E;
    font-size: 0.83rem;
}
.ot-row:last-child { border-bottom: none; }
.ot-k { color: #7F8EA3; }
.ot-v {
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    color: #E6EDF3;
}
.score-bar-bg {
    width: 100%;
    height: 7px;
    background: #1B2330;
    border-radius: 4px;
    overflow: hidden;
    margin: 6px 0 10px 0;
}
.score-bar-fill { height: 100%; border-radius: 4px; }

/* Watchlist Mini 52W Bar */
.wl-card {
    background: #0D121A;
    border: 1px solid #1A2230;
    border-radius: 7px;
    padding: 6px 9px;
    margin-bottom: 5px;
}
.wl-card-active {
    background: #141E2E;
    border: 1px solid #2962FF;
    border-radius: 7px;
    padding: 6px 9px;
    margin-bottom: 5px;
}
.wl-meta {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #7F8EA3;
    margin-top: 2px;
}
.range-track {
    width: 100%;
    height: 4px;
    background: #1D2736;
    border-radius: 2px;
    position: relative;
    margin-top: 4px;
}
.range-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #58A6FF;
    position: absolute;
    top: -1.5px;
    transform: translateX(-50%);
}

/* Compact Broker Tabs */
button[data-baseweb="tab"] {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 8px 14px !important;
    color: #8B9BB4 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #FFFFFF !important;
    border-bottom-color: #2962FF !important;
}
/* Make watchlist buttons crisp & flat */
div.wl-btn-wrap button {
    padding: 0.25rem 0.55rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.79rem !important;
    justify-content: space-between !important;
    background-color: #0F141C !important;
    border: 1px solid #1C2430 !important;
}
div.wl-btn-wrap button:hover {
    border-color: #2962FF !important;
    background-color: #151E2E !important;
}
</style>
"""
st.markdown(BROKER_CSS, unsafe_allow_html=True)


def _fmt_pct(val: Any, signed: bool = True) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    try:
        f = float(val)
        return f"{f:+.2f}%" if signed else f"{f:.1f}%"
    except Exception:
        return str(val)


def _fmt_usd(val: Any, digits: int = 2) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    try:
        return f"${float(val):,.{digits}f}"
    except Exception:
        return str(val)


def _color_cls(val: Any) -> str:
    try:
        return "pos-green" if float(val) >= 0 else "neg-red"
    except Exception:
        return ""


def _render_html(html: str) -> None:
    """Render multiline HTML cleanly in Streamlit without CommonMark indented-code-block artifacts."""
    compact = "\n".join(line.strip() for line in html.splitlines() if line.strip())
    st.markdown(compact, unsafe_allow_html=True)


def _fmt_ext_badge_html(quote: dict[str, Any] | None, compact: bool = False) -> str:
    """Format a compact Pre-Market / After-Hours HTML snippet when extended hours pricing is available."""
    if not quote:
        return ""
    sess = quote.get("ext_session")
    ext_px = quote.get("ext_price")
    ext_chg = quote.get("ext_change_pct")
    if not sess or ext_px is None:
        return ""
    cls = _color_cls(ext_chg if ext_chg is not None else 0.0)
    chg_s = f" {float(ext_chg):+.2f}%" if ext_chg is not None else ""
    if compact:
        return (
            f'<span style="color:#8B9BB4;margin-left:4px;">{sess}</span> '
            f'<span class="{cls}">${float(ext_px):,.2f}{chg_s}</span>'
        )
    pill_cls = "pill-green" if (ext_chg or 0.0) >= 0 else "pill-red"
    return (
        f'<div style="margin-top:4px;font-size:0.78rem;font-family:JetBrains Mono,monospace;">'
        f'<span style="color:#8B9BB4;margin-right:6px;">{sess}-MKT</span>'
        f'<span style="color:#E6EDF3;font-weight:600;">${float(ext_px):,.2f}</span>'
        f'<span class="qb-pill {pill_cls}" style="font-size:0.68rem;padding:1px 6px;">{chg_s.strip() or "0.00%"}</span>'
        f'</div>'
    )


def _build_ticker_tape_html(
    payload: dict[str, Any] | None,
    live_quotes: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render live macro index tape (SPY / QQQ / SOXX / VIX / US10Y) with optional Pre/Post-Market prices."""
    lq_map = live_quotes or (payload or {}).get("live_quotes") or {}
    if not payload and not lq_map:
        return '<div class="ticker-tape"><span class="tape-chip"><span class="tape-sym">MARKET TAPE</span><span class="tape-px">AWAITING SCAN</span></span></div>'
    rm = ((payload or {}).get("regime") or {}).get("metrics") or {}
    chips: list[str] = []
    for sym in ("SPY", "QQQ", "SOXX"):
        info = rm.get(sym)
        lq = lq_map.get(sym) or {}
        px = lq.get("regular_price") or (info.get("close") if isinstance(info, dict) else None)
        if px is not None:
            r5 = info.get("ret_5d_pct") if isinstance(info, dict) else None
            r1 = lq.get("regular_change_pct")
            metric_val = r1 if r1 is not None else r5
            metric_lbl = f"{float(r1):+.2f}%" if r1 is not None else (f"{float(r5):+.1f}% 5D" if r5 is not None else "")
            cls = _color_cls(metric_val)
            ext_html = _fmt_ext_badge_html(lq, compact=True)
            chips.append(
                f'<span class="tape-chip"><span class="tape-sym">{sym}</span>'
                f'<span class="tape-px">${float(px):,.2f}</span>'
                f'<span class="{cls}">{metric_lbl}</span>{ext_html}</span>'
            )
    if rm.get("VIX") is not None:
        v = float(rm["VIX"])
        v_cls = "pos-green" if v < 18 else ("neu-amber" if v < 24 else "neg-red")
        chips.append(
            f'<span class="tape-chip"><span class="tape-sym">VIX</span><span class="{v_cls}">{v:.2f}</span></span>'
        )
    if rm.get("US10Y_pct") is not None:
        chips.append(
            f'<span class="tape-chip"><span class="tape-sym">US10Y</span><span class="tape-px">{rm["US10Y_pct"]:.2f}%</span></span>'
        )
    if not chips:
        return ""
    return f'<div class="ticker-tape">{"".join(chips)}</div>'


def main() -> None:
    settings = load_settings()
    universe_cfg = load_universe()
    portfolio_cfg = load_portfolio()
    db = TradingDB(settings.db_path)

    # Load latest snapshot automatically on startup if nothing in session
    if "active_payload" not in st.session_state:
        latest = db.get_snapshot(None)
        if latest:
            st.session_state["active_payload"] = latest["payload"]
            st.session_state["active_payload"]["snapshot_id"] = latest["id"]

    # ==========================================================================
    # SINGLE-ROW MINIMAL BROKER HEADER + SLEEK EN / ZH SWITCH (DEFAULT: EN)
    # ==========================================================================
    top_c1, top_c2, top_c3 = st.columns([2.0, 4.1, 1.05], vertical_alignment="center")
    with top_c3:
        is_zh = st.toggle("EN / 中文", value=False, key="ui_lang_toggle")
        lang = "zh" if is_zh else "en"

    payload: dict[str, Any] | None = st.session_state.get("active_payload")
    dev = get_device_profile(settings.compute_device)

    with top_c1:
        short_gpu = dev.device_name.replace("NVIDIA GeForce ", "").replace("RTX ", "")
        gpu_pill = (
            f'<span class="qb-pill pill-green" style="font-size:0.68rem;padding:1px 6px;">{short_gpu}</span>'
            if dev.backend == "cuda"
            else f'<span class="qb-pill pill-muted" style="font-size:0.68rem;padding:1px 6px;">{dev.backend.upper()}</span>'
        )
        ts_badge = (
            f'<span style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#5F7088;margin-left:4px;">{str(payload.get("generated_at", ""))[5:16]}</span>'
            if payload
            else ""
        )
        st.markdown(
            f'<div class="brand-wrap"><span class="live-dot"></span><span class="brand-name">FRONT-RUNNING</span>{gpu_pill}{ts_badge}</div>',
            unsafe_allow_html=True,
        )

    with top_c2:
        merged_live_quotes = dict((payload or {}).get("live_quotes") or {})
        merged_live_quotes.update(st.session_state.get("live_quotes") or {})
        st.markdown(_build_ticker_tape_html(payload, merged_live_quotes), unsafe_allow_html=True)

    st.markdown('<div class="header-divider"></div>', unsafe_allow_html=True)

    # ==========================================================================
    # LEFT CONTROL SIDEBAR (RTX 5070 Ti Engine + Ollama + Scan Trigger + Snapshots)
    # ==========================================================================
    with st.sidebar:
        st.markdown(f"### {t('sidebar_title', lang)}")
        st.caption(t("sidebar_caption", lang))

        if dev.backend == "cuda":
            st.success(f"{t('gpu_ready', lang)}: {dev.summary_str()}")
            if dev.is_blackwell:
                st.caption(f"{t('blackwell_note', lang)}")
        else:
            st.info(f"{t('gpu_cpu_mode', lang)}: {dev.summary_str()}")

        st.divider()
        st.markdown(f"**{t('llm_section', lang)}**")
        llm_base_url = st.text_input(t("llm_url", lang), value=settings.llm_base_url)
        probe = LocalLLMClient(
            base_url=llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        ).check_health()

        if probe.get("online"):
            avail_models = probe.get("models") or [settings.llm_model]
            default_idx = avail_models.index(settings.llm_model) if settings.llm_model in avail_models else 0
            llm_model = st.selectbox(t("llm_model", lang), options=avail_models, index=default_idx)
            st.success(f"{t('llm_online', lang)} ({len(avail_models)})")
        else:
            llm_model = st.text_input(t("llm_model", lang), value=settings.llm_model)
            st.warning(f"{t('llm_offline', lang)}")

        st.divider()
        st.markdown(f"**{t('scan_section', lang)}**")
        force_refresh = st.checkbox(t("opt_refresh", lang), value=False)
        opt_ml = st.checkbox(t("opt_ml", lang), value=True)
        opt_bt = st.checkbox(t("opt_bt", lang), value=True)
        opt_llm = st.checkbox(t("opt_llm", lang), value=bool(probe.get("online")))

        if st.button(t("btn_run_scan", lang), type="primary", use_container_width=True):
            pbar = st.progress(0.0)
            status_box = st.empty()

            def _cb(pct: float, msg: str) -> None:
                pbar.progress(min(1.0, max(0.0, pct)))
                status_box.info(msg)

            try:
                new_payload = run_full_analysis_pipeline(
                    settings=settings,
                    universe_cfg=universe_cfg,
                    portfolio_cfg=portfolio_cfg,
                    force_refresh_data=force_refresh,
                    run_ml_alpha=opt_ml,
                    run_backtest=opt_bt,
                    run_llm=opt_llm,
                    llm_model_override=llm_model,
                    llm_base_url_override=llm_base_url,
                    lang=lang,
                    progress_cb=_cb,
                )
                st.session_state["active_payload"] = new_payload
                payload = new_payload
                st.success(t("scan_done", lang, sec=new_payload.get("elapsed_sec")))
                st.rerun()
            except Exception as exc:
                st.error(f"{t('scan_failed', lang)}: {exc}")

        if payload and payload.get("report_path") and Path(payload["report_path"]).exists():
            st.download_button(
                f"{t('btn_download_md', lang)}",
                data=Path(payload["report_path"]).read_bytes(),
                file_name=Path(payload["report_path"]).name,
                mime="text/markdown",
                use_container_width=True,
            )

        st.divider()
        st.markdown(f"**{t('snap_section', lang)}**")
        snaps = db.list_snapshots(limit=15)
        if snaps:
            labels = [
                f"#{s['id']} · {s['created_at'][:16].replace('T', ' ')} · R={s['regime_score']:.0f}"
                for s in snaps
            ]
            chosen_idx = st.selectbox(t("snap_select", lang), range(len(labels)), format_func=lambda i: labels[i])
            if st.button(t("btn_load_snap", lang), use_container_width=True):
                loaded = db.get_snapshot(int(snaps[chosen_idx]["id"]))
                if loaded:
                    st.session_state["active_payload"] = loaded["payload"]
                    st.session_state["active_payload"]["snapshot_id"] = loaded["id"]
                    st.rerun()

    if not payload:
        st.info(t("no_snap_banner", lang))

    # ==========================================================================
    # MAIN BROKER NAVIGATION TABS
    # ==========================================================================
    tabs = st.tabs(
        [
            t("tab_desk", lang),
            t("tab_scanner", lang),
            t("tab_portfolio", lang),
            t("tab_chain", lang),
            t("tab_backtest", lang),
            t("tab_grid", lang),
            t("tab_journal", lang),
            t("tab_guide", lang),
        ]
    )

    with tabs[0]:
        _render_trade_desk_tab(payload, settings, universe_cfg, portfolio_cfg, db, llm_model, llm_base_url, lang)
    with tabs[1]:
        _render_scanner_regime_tab(payload, lang)
    with tabs[2]:
        _render_portfolio_risk_tab(payload, portfolio_cfg, lang)
    with tabs[3]:
        _render_supply_chain_tab(payload, lang)
    with tabs[4]:
        _render_backtest_ml_tab(payload, lang)
    with tabs[5]:
        _render_grid_lab_tab(settings, universe_cfg, portfolio_cfg, db, lang)
    with tabs[6]:
        _render_journal_tab(db, lang)
    with tabs[7]:
        _render_guide_tab(dev, lang)


# ==============================================================================
# TAB 1: BROKER TRADE DESK (3-Column: Watchlist | Pro Chart + Research | Execution Ticket)
# ==============================================================================
def _render_trade_desk_tab(
    payload: dict[str, Any] | None,
    settings: Any,
    universe_cfg: dict[str, Any],
    portfolio_cfg: dict[str, Any],
    db: TradingDB,
    llm_model: str,
    llm_base_url: str,
    lang: str,
) -> None:
    is_zh = lang == "zh"
    stocks = (payload or {}).get("stocks") or {}
    u_meta = universe_cfg.get("universe") or {}
    holding_syms = {
        str(h.get("symbol", "")).upper().strip()
        for h in (portfolio_cfg.get("holdings") or [])
        if h.get("symbol")
    }
    configured_syms = list(dict.fromkeys(list(u_meta.keys()) + list(holding_syms)))
    valid_syms = [s for s in configured_syms if s in stocks and not stocks[s].get("error")]
    display_syms = valid_syms if valid_syms else configured_syms

    # Default selected ticker
    if "desk_selected_sym" not in st.session_state:
        st.session_state["desk_selected_sym"] = display_syms[0] if display_syms else "GOOG"

    col_wl, col_chart, col_ticket = st.columns([1.08, 2.22, 1.05], gap="medium")

    # --- LEFT RAIL: BROKER WATCHLIST QUOTE LADDER + ANY-TICKER LOOKUP + EDIT ---
    with col_wl:
        wl_h1, wl_h2 = st.columns([1.6, 1.0], vertical_alignment="center")
        wl_h1.markdown(f"**{t('watchlist_hdr', lang)} ({len(display_syms)})**")
        with wl_h2:
            if st.button("Quotes" if not is_zh else "刷报价", use_container_width=True, key="btn_quick_wl_quotes"):
                with st.spinner("Fetching regular & pre/post-market quotes..." if not is_zh else "拉取常规及盘前/盘后实时报价..."):
                    md_q = MarketDataFetcher(settings.data_dir, history_years=2)
                    q_map, _ = md_q.fetch_universe(universe_cfg, extra_symbols=list(holding_syms))
                    live_q = md_q.fetch_live_quotes_batch(
                        universe_cfg, extra_symbols=list(holding_syms), include_benchmarks=True
                    )
                    quick_quotes: dict[str, dict[str, float]] = {}
                    for sym_q, df_q in q_map.items():
                        if df_q is not None and not df_q.empty:
                            c_last = float(df_q["Close"].iloc[-1])
                            c_prev = float(df_q["Close"].iloc[-2]) if len(df_q) >= 2 else c_last
                            w252 = df_q.tail(min(252, len(df_q)))
                            hi52, lo52 = float(w252["High"].max()), float(w252["Low"].min())
                            pos52 = (c_last - lo52) / (hi52 - lo52) * 100.0 if hi52 > lo52 else 50.0
                            lq_item = live_q.get(sym_q) or {}
                            disp_px = lq_item.get("regular_price") or c_last
                            disp_chg = (
                                lq_item.get("regular_change_pct")
                                if lq_item.get("regular_change_pct") is not None
                                else (round((c_last / c_prev - 1.0) * 100.0, 2) if c_prev > 0 else 0.0)
                            )
                            quick_quotes[sym_q] = {
                                "close": round(float(disp_px), 2),
                                "chg_1d_pct": round(float(disp_chg), 2),
                                "pos_in_52w_pct": round(pos52, 1),
                            }
                    st.session_state["quick_quotes"] = quick_quotes
                    st.session_state["live_quotes"] = live_q
                    st.rerun()

        lc1, lc2 = st.columns([2.2, 1.0])
        custom_sym = lc1.text_input(
            "lookup",
            placeholder=t("any_ticker_ph", lang),
            label_visibility="collapsed",
        ).upper().strip()
        if lc2.button(t("btn_lookup", lang), use_container_width=True) and custom_sym:
            with st.spinner(t("lookup_spinner", lang, sym=custom_sym)):
                res = run_on_demand_single_ticker_check(
                    custom_sym,
                    settings=settings,
                    portfolio_cfg=portfolio_cfg,
                    run_llm=True,
                    llm_model_override=llm_model,
                    llm_base_url_override=llm_base_url,
                    lang=lang,
                )
                st.session_state["custom_check_result"] = res
                st.session_state["desk_selected_sym"] = custom_sym
                st.rerun()

        sort_mode = st.selectbox(
            "sort",
            options=["POS_FIRST", "SCORE_DESC", "CHG_ASC", "CHG_DESC"],
            format_func=lambda x: {
                "POS_FIRST": "Holdings First / Score" if lang == "en" else "持仓优先 / 评分",
                "SCORE_DESC": "Highest Score" if lang == "en" else "综合分从高到低",
                "CHG_ASC": "Biggest Dip (1D %)" if lang == "en" else "今日跌幅最大",
                "CHG_DESC": "Top Gainers (1D %)" if lang == "en" else "今日涨幅最大",
            }[x],
            label_visibility="collapsed",
        )

        quick_quotes = st.session_state.get("quick_quotes") or {}
        live_quotes = dict((payload or {}).get("live_quotes") or {})
        live_quotes.update(st.session_state.get("live_quotes") or {})
        ordered_syms = list(display_syms)
        if sort_mode == "POS_FIRST":
            ordered_syms.sort(
                key=lambda s: (
                    0 if (s in holding_syms or (stocks.get(s) or {}).get("role") == "holding") else 1,
                    -float((stocks.get(s) or {}).get("composite_score", 0.0)),
                )
            )
        elif sort_mode == "SCORE_DESC":
            ordered_syms.sort(key=lambda s: -float((stocks.get(s) or {}).get("composite_score", 0.0)))
        elif sort_mode == "CHG_ASC":
            ordered_syms.sort(
                key=lambda s: float((stocks.get(s) or quick_quotes.get(s) or {}).get("chg_1d_pct") or 0.0)
            )
        elif sort_mode == "CHG_DESC":
            ordered_syms.sort(
                key=lambda s: -float((stocks.get(s) or quick_quotes.get(s) or {}).get("chg_1d_pct") or 0.0)
            )

        # Render sleek scrollable quote ladder (ALWAYS populated from config/universe.yaml!)
        active_sym = st.session_state.get("desk_selected_sym", "GOOG")
        with st.container(height=620, border=False):
            for sym in ordered_syms:
                sd = stocks.get(sym) if (sym in stocks and not stocks[sym].get("error")) else None
                qq = quick_quotes.get(sym)
                lq = live_quotes.get(sym) or (sd or {}).get("extended_quote")
                meta = u_meta.get(sym) or {}
                is_pos = sym in holding_syms or (sd or {}).get("role") == "holding"
                is_bell = meta.get("role") == "bellwether" and not is_pos
                is_active = sym == active_sym

                role_tag = " [POS]" if is_pos else (" [BELL]" if is_bell else "")
                px = (qq or {}).get("close") or (lq or {}).get("regular_price") or (sd or {}).get("close")
                chg = (
                    (qq or {}).get("chg_1d_pct")
                    if (qq and qq.get("chg_1d_pct") is not None)
                    else (
                        (lq or {}).get("regular_change_pct")
                        if (lq and lq.get("regular_change_pct") is not None)
                        else (sd or {}).get("chg_1d_pct")
                    )
                )

                if px is not None and chg is not None:
                    arrow = "▲" if float(chg) >= 0 else "▼"
                    btn_label = f"{sym}{role_tag}   ${float(px):,.2f}   {arrow}{abs(float(chg)):.2f}%"
                else:
                    short_name = str(meta.get("name", sym)).split("(")[0].strip()[:14]
                    btn_label = f"{sym}{role_tag}   ·   {short_name}"

                st.markdown('<div class="wl-btn-wrap">', unsafe_allow_html=True)
                if st.button(
                    btn_label,
                    key=f"wl_btn_{sym}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["desk_selected_sym"] = sym
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

                ext_line = ""
                if lq and lq.get("ext_session") and lq.get("ext_price") is not None:
                    e_chg = lq.get("ext_change_pct")
                    e_col = "#0ECB81" if (e_chg or 0.0) >= 0 else "#F6465D"
                    e_chg_s = f" ({float(e_chg):+.2f}%)" if e_chg is not None else ""
                    ext_line = (
                        f'<div class="wl-meta" style="margin-bottom:2px;">'
                        f'<span>{lq["ext_session"]}-MKT</span>'
                        f'<span style="color:{e_col};font-weight:600;">${float(lq["ext_price"]):,.2f}{e_chg_s}</span>'
                        f'</div>'
                    )

                if sd:
                    sc = float(sd.get("composite_score") or 50.0)
                    pos52 = max(2.0, min(98.0, float(sd.get("pos_in_52w_pct") or 50.0)))
                    sc_color = "#0ECB81" if sc >= 70 else ("#F0B90B" if sc >= 52 else "#F6465D")
                    _render_html(
                        f"""
                        <div style="margin:-8px 2px 8px 2px;">
                            {ext_line}
                            <div class="wl-meta">
                                <span>52W: {pos52:.0f}%</span>
                                <span style="color:{sc_color};font-weight:700;">SCR {sc:.0f}</span>
                            </div>
                            <div class="range-track"><div class="range-dot" style="left:{pos52:.1f}%;"></div></div>
                        </div>
                        """
                    )
                elif qq or ext_line:
                    pos52 = max(2.0, min(98.0, float((qq or {}).get("pos_in_52w_pct") or 50.0)))
                    _render_html(
                        f"""
                        <div style="margin:-8px 2px 8px 2px;">
                            {ext_line}
                            <div class="wl-meta"><span>52W: {pos52:.0f}%</span><span>{meta.get('sector', '')}</span></div>
                            <div class="range-track"><div class="range-dot" style="left:{pos52:.1f}%;"></div></div>
                        </div>
                        """
                    )

        # Quick Watchlist Add / Remove Manager (persists to config/universe.yaml)
        with st.expander("Edit Watchlist (+ / -)" if not is_zh else "编辑自选股 (+ / -)", expanded=False):
            new_tk = st.text_input("Add US Ticker" if not is_zh else "添加美股代码", placeholder="e.g. PLTR", key="wl_add_tk").upper().strip()
            new_sec = st.selectbox(
                "Sector" if not is_zh else "所属赛道",
                options=list((universe_cfg.get("sectors") or {}).keys()) or ["cloud_ai", "compute_optical", "memory_storage", "mega_disruptors", "defensive"],
                key="wl_add_sec",
            )
            ac1, ac2 = st.columns(2)
            if ac1.button("Add" if not is_zh else "添加", use_container_width=True) and new_tk:
                u_copy = dict(universe_cfg)
                u_copy["universe"] = dict(u_copy.get("universe") or {})
                u_copy["universe"][new_tk] = {
                    "name": new_tk,
                    "sector": new_sec,
                    "role": "watchlist",
                    "bellwether": False,
                    "notes": "Added via UI",
                }
                save_universe(u_copy)
                st.session_state["desk_selected_sym"] = new_tk
                st.rerun()
            rem_tk = st.selectbox("Remove" if not is_zh else "移除代码", options=[""] + list(u_meta.keys()), key="wl_rem_tk")
            if ac2.button("Remove" if not is_zh else "移除", use_container_width=True) and rem_tk:
                u_copy = dict(universe_cfg)
                u_copy["universe"] = dict(u_copy.get("universe") or {})
                u_copy["universe"].pop(rem_tk, None)
                save_universe(u_copy)
                st.rerun()

    # Resolve currently active stock data (from full scan, ad-hoc lookup, or live chart preview!)
    active_sym = st.session_state.get("desk_selected_sym", "GOOG")
    custom_res = st.session_state.get("custom_check_result")
    if custom_res and custom_res.get("symbol") == active_sym and not custom_res.get("error"):
        stock_data = custom_res
    elif active_sym in stocks and not stocks[active_sym].get("error"):
        stock_data = stocks[active_sym]
    else:
        stock_data = None

    if not stock_data:
        # Even BEFORE a full scan, show live Candlestick preview for the clicked watchlist ticker!
        with col_chart:
            meta = u_meta.get(active_sym) or {}
            st.markdown(f"### `{active_sym}` · {meta.get('name', active_sym)}")
            md_pre = MarketDataFetcher(settings.data_dir, history_years=2)
            df_pre, _ = md_pre.fetch_one(active_sym, fallbacks=list(meta.get("yahoo_fallbacks") or []))
            if not df_pre.empty:
                st.plotly_chart(
                    make_candlestick_trade_chart(active_sym, df_pre, ticket=None, lookback_bars=130, lang=lang),
                    use_container_width=True,
                )
            st.info(
                f"Click **RUN FULL SCAN** in the sidebar (or type `{active_sym}` in Lookup) to unlock full GPU Quant Scores, Execution Ticket, 10Y Analogs & 5070 Ti AI Debate."
                if not is_zh
                else f"点击左侧 **一键启动全量分析**（或点击上方即时体检）即可解锁 `{active_sym}` 的完整 GPU 量化评分、执行单、10年历史胜率与 5070 Ti AI 排雷。"
            )
        with col_ticket:
            _render_html(
                f"""
                <div class="order-ticket">
                    <div class="ot-header">
                        <span class="ot-title">{t('order_ticket_hdr', lang)}</span>
                        <span class="qb-pill pill-blue">{active_sym}</span>
                    </div>
                    <div style="font-size:0.80rem;color:#7F8EA3;line-height:1.6;">
                        {'Run a Full Scan or Lookup to calculate Entry Zone, Hard Stop-Loss, 2R/3R Targets & Volatility-Adjusted Position Size.' if not is_zh else '点击左侧「一键启动全量分析」后，这里会自动生成参考入场区、硬止损、2R/3R 目标位与建议买入股数。'}
                    </div>
                </div>
                """
            )
        return

    tk = stock_data.get("trade_ticket") or {}
    cr = stock_data.get("crash_analysis") or {}
    fn = stock_data.get("fundamentals") or {}
    an = stock_data.get("analog") or {}
    ml = stock_data.get("ml_alpha") or {}
    lr = stock_data.get("llm_report") or {}
    sec = stock_data.get("sec") or {}

    # --- CENTER STAGE: BROKER QUOTE HEADER + CANDLESTICK CHART + RESEARCH TABS ---
    with col_chart:
        active_lq = live_quotes.get(active_sym) or stock_data.get("extended_quote")
        disp_close = (
            active_lq.get("regular_price")
            if (active_lq and active_lq.get("regular_price") is not None)
            else stock_data.get("close")
        )
        chg1 = float(
            active_lq.get("regular_change_pct")
            if (active_lq and active_lq.get("regular_change_pct") is not None)
            else (stock_data.get("chg_1d_pct") or 0.0)
        )
        chg_pill_cls = "pill-green" if chg1 >= 0 else "pill-red"
        sec_str = tr_sector(str(stock_data.get("sector", "")), str(stock_data.get("sector_label", "")), lang)
        stage_str = tr_stage(stock_data.get("stage_num"), str(stock_data.get("stage_label", "")), lang)
        earn_str = tr_earnings_flag(fn, lang)
        crash_str = tr_crash_badge(cr, lang)
        ext_banner_html = _fmt_ext_badge_html(active_lq, compact=False)

        _render_html(
            f"""
            <div class="quote-banner">
                <div>
                    <span class="qb-sym">{active_sym}</span>
                    <span class="qb-name">{stock_data.get('name', active_sym)}</span>
                    <span class="qb-pill pill-muted">{sec_str}</span>
                    <div style="margin-top:5px;font-size:0.78rem;color:#8B9BB4;font-family:JetBrains Mono,monospace;">
                        {stage_str} &nbsp;|&nbsp; 52W: ${_fmt_num(stock_data.get('low_52w'))} – ${_fmt_num(stock_data.get('high_52w'))}
                        ({ _fmt_pct(stock_data.get('dd_from_52w_high_pct')) }) &nbsp;|&nbsp; RVOL: {stock_data.get('rvol')}x &nbsp;|&nbsp; {earn_str}
                    </div>
                </div>
                <div style="text-align:right;">
                    <span class="qb-price">${_fmt_num(disp_close)}</span>
                    <span class="qb-pill {chg_pill_cls}">{chg1:+.2f}%</span>
                    {ext_banner_html}
                    <div style="margin-top:4px;font-size:0.76rem;color:#8B9BB4;font-family:JetBrains Mono,monospace;">
                        5D: {_fmt_pct(stock_data.get('ret_5d_pct'))} &nbsp; 21D: {_fmt_pct(stock_data.get('ret_21d_pct'))} &nbsp; 63D: {_fmt_pct(stock_data.get('ret_63d_pct'))}
                    </div>
                </div>
            </div>
            """
        )

        if stock_data.get("short_history_warning"):
            st.warning(t("short_hist_warn", lang, sym=active_sym, bars=stock_data.get("history_bars", 0)))

        # Timeframe selector just like TradingView / Webull
        tf_map = {"3M": 65, "6M": 130, "1Y": 252, "2Y": 504}
        tf_choice = st.radio(
            t("timeframe_lbl", lang),
            options=list(tf_map.keys()),
            index=1,
            horizontal=True,
            label_visibility="collapsed",
        )
        md_fetcher = MarketDataFetcher(settings.data_dir, history_years=settings.history_years)
        df_px, _ = md_fetcher.fetch_one(active_sym, fallbacks=[str(stock_data.get("resolved_symbol", active_sym))])
        if not df_px.empty:
            st.plotly_chart(
                make_candlestick_trade_chart(active_sym, df_px, tk, lookback_bars=tf_map[tf_choice], lang=lang),
                use_container_width=True,
            )

        # Compact 4-Subtab Deep Intelligence Drawer below Chart
        sub_tabs = st.tabs(
            [
                t("sub_quant", lang),
                t("sub_analog", lang),
                t("sub_ai", lang),
                t("sub_sec", lang),
            ]
        )

        with sub_tabs[0]:
            q1, q2 = st.columns([1.25, 1.0])
            with q1:
                st.markdown(f"**{t('crash_title', lang)}**")
                st.markdown(
                    f"`{crash_str}` &nbsp;|&nbsp; **{t('knife_score_lbl', lang)}**: `{cr.get('knife_risk_score', '-')}/100`"
                )
                if lang == "zh":
                    st.caption(cr.get("reason", ""))
                st.markdown(
                    f"""
                    - **Trend / Momentum**: RSI14 `{stock_data.get('rsi14')}` · ADX `{stock_data.get('adx14')}` · ATR14 `${stock_data.get('atr14')}` (`{stock_data.get('atr_pct')}%`) · VCP `{stock_data.get('vcp_ratio')}`
                    - **Institutional Flow**: Up/Down Vol `{stock_data.get('up_down_vol_ratio_20d')}` · CMF20 `{stock_data.get('cmf20')}` · Accum `{stock_data.get('accum_days_20d')}d` / Dist `{stock_data.get('dist_days_20d')}d`
                    - **Relative Strength**: vs QQQ 21D `{_fmt_pct(stock_data.get('rs_qqq_21d_pct'))}` · vs Sector 21D `{_fmt_pct(stock_data.get('rs_sector_21d_pct'))}`
                    - **Fundamentals**: MCap `${fn.get('market_cap_b')}B` · Fwd P/E `{fn.get('forward_pe')}` · Rev YoY `{_fmt_pct(fn.get('revenue_growth_yoy_pct'))}` · GM `{_fmt_pct(fn.get('gross_margin_pct'), False)}` · Last EPS Beat `{_fmt_pct(fn.get('last_eps_surprise_pct'))}`
                    """
                )
                if stock_data.get("penalties"):
                    st.warning(f"{t('penalties_lbl', lang)}: " + " · ".join(stock_data["penalties"]))
            with q2:
                st.plotly_chart(
                    make_pillar_radar(active_sym, stock_data.get("pillars") or {}, lang=lang),
                    use_container_width=True,
                )

        with sub_tabs[1]:
            if an:
                a1, a2, a3, a4, a5 = st.columns(5)
                a1.metric("21D Win Rate" if lang == "en" else "21日胜率", _fmt_pct(an.get("win_rate_21d_pct"), False))
                a2.metric("21D Median" if lang == "en" else "21日中位收益", _fmt_pct(an.get("median_ret_21d_pct")))
                a3.metric("42D Win Rate" if lang == "en" else "42日胜率", _fmt_pct(an.get("win_rate_42d_pct"), False))
                a4.metric("Profit Factor" if lang == "en" else "盈亏比 PF", f"{an.get('profit_factor_21d', '-')}")
                a5.metric("Median MAE" if lang == "en" else "中位最大回撤", _fmt_pct(an.get("median_mae_21d_pct")))
                st.markdown(f"**{t('analog_hdr', lang)}**")
                if an.get("top_matches"):
                    st.dataframe(pd.DataFrame(an["top_matches"]), use_container_width=True, hide_index=True)
            else:
                st.caption(t("analog_empty", lang))

        with sub_tabs[2]:
            if lr and not lr.get("error") and not lr.get("parse_error"):
                st.markdown(
                    f"**{t('ai_verdict_lbl', lang)}**: `{lr.get('conviction_verdict', '-')}` &nbsp;|&nbsp; "
                    f"**{t('ai_conf_lbl', lang)}**: `{lr.get('confidence_score', '-')}/100` &nbsp;|&nbsp; "
                    f"**{t('ai_crash_lbl', lang)}**: `{lr.get('crash_attribution', '-')}`"
                )
                st.success(f"**{t('ai_summary_lbl', lang)}**: {lr.get('one_line_summary', '')}")
                cb1, cb2 = st.columns(2)
                with cb1:
                    st.markdown(f"**{t('ai_bull_hdr', lang)}**")
                    for b in lr.get("bull_thesis") or []:
                        st.markdown(f"- {b}")
                with cb2:
                    st.markdown(f"**{t('ai_bear_hdr', lang)}**")
                    for r in lr.get("bear_risks") or []:
                        st.markdown(f"- {r}")
                if lr.get("catalysts_1_2m"):
                    st.markdown(f"**{t('ai_cat_hdr', lang)}**: " + " · ".join(lr["catalysts_1_2m"]))
                st.info(f"**{t('ai_plan_hdr', lang)}**: {lr.get('action_plan_cn', '')}")
            elif lr.get("raw_text"):
                st.write(lr["raw_text"])
            else:
                st.caption(lr.get("error") or t("ai_unavailable", lang))

        with sub_tabs[3]:
            st.markdown(f"**{t('sec_hdr', lang)}**")
            st.caption(sec.get("sec_digest", "No SEC EDGAR summary available."))
            for rf in sec.get("red_flag_filings") or []:
                st.error(f"[FLAG] {rf}")
            for item in (stock_data.get("news") or [])[:7]:
                link, title = item.get("link") or "", item.get("title") or ""
                pub, dt = item.get("publisher") or "", (item.get("published") or "")[:10]
                if link:
                    st.markdown(f"- [{title}]({link}) `{pub} · {dt}`")
                else:
                    st.markdown(f"- {title} `{pub} · {dt}`")

    # --- RIGHT RAIL: BROKER EXECUTION TICKET & ACTIVE POSITION CARD ---
    with col_ticket:
        comp_score = float(stock_data.get("composite_score") or 50.0)
        sc_bar_col = "#0ECB81" if comp_score >= 70 else ("#F0B90B" if comp_score >= 52 else "#F6465D")
        sig_text = tr_signal(str(tk.get("action_signal", "")), lang)
        sig_upper = sig_text.upper()
        sig_pill = (
            "pill-green"
            if ("PRIORITY BUY" in sig_upper or "优先买入" in sig_text)
            else ("pill-red" if ("AVOID KNIFE" in sig_upper or "禁接飞刀" in sig_text) else "pill-amber")
        )
        ml_prob = _fmt_pct(ml.get("prob_outperform_21d_pct"), False)
        an_wr = _fmt_pct(an.get("win_rate_21d_pct"), False)

        _render_html(
            f"""
            <div class="order-ticket">
                <div class="ot-header">
                    <span class="ot-title">{t('order_ticket_hdr', lang)}</span>
                    <span class="qb-pill pill-blue">{active_sym}</span>
                </div>
                <div style="font-size:0.70rem;color:#6E7F96;margin-bottom:8px;letter-spacing:0.05em;">{t('advisory_only', lang)}</div>
                <div style="display:flex;justify-content:space-between;align-items:baseline;">
                    <span class="ot-k" style="font-size:0.8rem;">{t('lbl_score', lang)}</span>
                    <span style="font-family:JetBrains Mono,monospace;font-size:1.35rem;font-weight:700;color:{sc_bar_col};">{comp_score:.1f}<span style="font-size:0.78rem;color:#7F8EA3;">/100</span></span>
                </div>
                <div class="score-bar-bg"><div class="score-bar-fill" style="width:{comp_score:.0f}%;background:{sc_bar_col};"></div></div>
                <div style="margin-bottom:10px;"><span class="qb-pill {sig_pill}" style="margin-left:0;">{sig_text}</span></div>
                <div class="ot-row"><span class="ot-k">{t('lbl_entry_zone', lang)}</span><span class="ot-v accent-blue">${_fmt_num(tk.get('entry_low'))} – ${_fmt_num(tk.get('entry_high'))}</span></div>
                <div class="ot-row"><span class="ot-k">{t('lbl_hard_stop', lang)}</span><span class="ot-v neg-red">${_fmt_num(tk.get('hard_stop'))} (-{tk.get('stop_dist_pct', 0)}%)</span></div>
                <div class="ot-row"><span class="ot-k">{t('lbl_trail_stop', lang)}</span><span class="ot-v neu-amber">${_fmt_num(tk.get('trailing_stop_guide'))}</span></div>
                <div class="ot-row"><span class="ot-k">{t('lbl_target1', lang)}</span><span class="ot-v pos-green">${_fmt_num(tk.get('target_1'))} ({tk.get('rr_target_1', 0)}R)</span></div>
                <div class="ot-row"><span class="ot-k">{t('lbl_target2', lang)}</span><span class="ot-v pos-green">${_fmt_num(tk.get('target_2'))} ({tk.get('rr_target_2', 0)}R)</span></div>
                <div class="ot-row"><span class="ot-k">GPU 10Y / ML</span><span class="ot-v">{an_wr} / {ml_prob}</span></div>
                <div class="ot-row"><span class="ot-k">{t('lbl_pos_size', lang)}</span><span class="ot-v">{tk.get('recommended_shares', 0)} shs ({tk.get('recommended_weight_pct', 0)}%)</span></div>
                <div class="ot-row"><span class="ot-k">Notional / Risk</span><span class="ot-v">{_fmt_usd(tk.get('recommended_usd'), 0)} / {_fmt_usd(tk.get('planned_max_risk_usd'), 0)}</span></div>
            </div>
            """
        )

        if st.button(t("btn_quick_log", lang), use_container_width=True):
            db.add_journal_entry(
                symbol=active_sym,
                action="BUY",
                shares=float(tk.get("recommended_shares") or 0),
                price=float(tk.get("ref_entry") or stock_data.get("close") or 0),
                stop_loss=float(tk.get("hard_stop") or 0),
                target_price=float(tk.get("target_1") or 0),
                setup_tag="Desk Plan",
                conviction=4 if comp_score >= 70 else 3,
                notes=f"Auto-logged from Trade Desk ({sig_text}, Score {comp_score})",
            )
            st.success(t("quick_log_ok", lang, sym=active_sym))

        # If this symbol is an active holding, display live Position Box right below Ticket
        h_diag = (payload or {}).get("holdings_diagnosis") or []
        matched_h = next((h for h in h_diag if h.get("symbol") == active_sym), None)
        if matched_h:
            h_act = tr_holding_action(str(matched_h.get("action", "")), lang)
            pnl_s = _fmt_pct(matched_h.get("pnl_pct"))
            pnl_cls = _color_cls(matched_h.get("pnl_pct"))
            _render_html(
                f"""
                <div class="order-ticket" style="border-color:#2962FF;">
                    <div class="ot-header">
                        <span class="ot-title">{'ACTIVE POSITION' if lang == 'en' else '当前持仓监控'}</span>
                        <span class="qb-pill pill-blue">{matched_h.get('shares', 0):g} shs</span>
                    </div>
                    <div style="margin-bottom:8px;font-size:0.80rem;font-weight:700;color:#F0F6FC;">{h_act}</div>
                    <div class="ot-row"><span class="ot-k">{'Market Value' if lang == 'en' else '持仓市值'}</span><span class="ot-v">{_fmt_usd(matched_h.get('market_value_usd'), 0)} ({matched_h.get('weight_pct', 0)}%)</span></div>
                    <div class="ot-row"><span class="ot-k">{'Unrealized P&L' if lang == 'en' else '浮动盈亏'}</span><span class="ot-v {pnl_cls}">{pnl_s}</span></div>
                    <div class="ot-row"><span class="ot-k">{'Active Stop' if lang == 'en' else '风控止损'}</span><span class="ot-v neg-red">${_fmt_num(matched_h.get('active_stop'))} ({_fmt_pct(matched_h.get('dist_to_stop_pct'))})</span></div>
                    <div class="ot-row"><span class="ot-k">{'Risk Share' if lang == 'en' else '组合风险贡献'}</span><span class="ot-v neu-amber">{_fmt_pct(matched_h.get('risk_contribution_pct'), False)}</span></div>
                </div>
                """
            )


# ==============================================================================
# TAB 2: SIGNALS, MARKET REGIME & PM WEEKLY BRIEF
# ==============================================================================
def _render_scanner_regime_tab(payload: dict[str, Any] | None, lang: str) -> None:
    if not payload:
        st.info(t("no_snap_banner", lang))
        return

    regime = payload.get("regime") or {}
    rm = regime.get("metrics") or {}
    r_score = float(regime.get("regime_score", 50.0))
    r_label = tr_regime_label(r_score, str(regime.get("regime_label", "")), lang)
    r_guide = tr_regime_guide(r_score, str(regime.get("action_guide", "")), lang)

    c1, c2 = st.columns([1.05, 2.35])
    with c1:
        st.plotly_chart(make_regime_gauge(r_score, r_label), use_container_width=True)
    with c2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t("regime_exposure", lang), f"{regime.get('suggested_max_exposure_pct', 60)}%")
        m2.metric(t("regime_cash", lang), f"{regime.get('suggested_defensive_cash_pct', 40)}%")
        m3.metric(t("regime_vix", lang), f"{rm.get('VIX', '—')}")
        m4.metric(t("regime_us10y", lang), f"{rm.get('US10Y_pct', '—')}%")
        st.info(f"**{t('regime_guide_lbl', lang)}**: {r_guide}")
        if lang == "zh":
            for sig in regime.get("signals") or []:
                st.markdown(f"- {sig}")
        else:
            b50 = rm.get("breadth_above_50ma_pct")
            b20 = rm.get("breadth_above_20ma_pct")
            soxx_rel = rm.get("soxx_vs_spy_20d_pct")
            st.markdown(
                f"- **Internal Breadth**: `{b50 if b50 is not None else '—'}%` above 50MA · `{b20 if b20 is not None else '—'}%` above 20MA\n"
                f"- **Semiconductor Leadership (SOXX/SPY 20D)**: `{_fmt_pct(soxx_rel)}`\n"
                f"- **Rates Anchor (US10Y 20D Chg)**: `{rm.get('US10Y_20d_chg_bps', '—')} bps`"
            )

    llm = payload.get("llm") or {}
    if llm.get("available") and llm.get("pm_weekly_memo"):
        with st.expander(t("pm_memo_hdr", lang), expanded=True):
            st.markdown(llm["pm_weekly_memo"])

    diffs = payload.get("weekly_diff") or []
    if diffs:
        st.subheader(t("diff_hdr", lang))
        st.dataframe(pd.DataFrame(diffs[:6]), use_container_width=True, hide_index=True)

    st.subheader(t("screener_hdr", lang))
    stocks = payload.get("stocks") or {}
    rows = []
    for sym, s in stocks.items():
        if s.get("error"):
            continue
        tk = s.get("trade_ticket") or {}
        an = s.get("analog") or {}
        ml = s.get("ml_alpha") or {}
        lr = s.get("llm_report") or {}
        rows.append(
            {
                t("col_sym", lang): sym,
                t("col_role", lang): tr_role(str(s.get("role", "watchlist")), lang),
                t("col_sector", lang): s.get("sector"),
                t("col_price", lang): s.get("close"),
                t("col_1d", lang): s.get("chg_1d_pct"),
                t("col_21d", lang): s.get("ret_21d_pct"),
                t("col_dd52", lang): s.get("dd_from_52w_high_pct"),
                t("col_score", lang): s.get("composite_score"),
                t("col_analog_wr", lang): an.get("win_rate_21d_pct"),
                t("col_ml_prob", lang): ml.get("prob_outperform_21d_pct"),
                t("col_crash", lang): tr_crash_badge(s.get("crash_analysis"), lang),
                t("col_earn", lang): tr_earnings_flag(s.get("fundamentals"), lang),
                t("col_entry", lang): f"${tk.get('entry_low')}–${tk.get('entry_high')}",
                t("col_stop", lang): tk.get("hard_stop"),
                t("col_t1", lang): f"${tk.get('target_1')} ({tk.get('rr_target_1')}R)",
                t("col_shares", lang): tk.get("recommended_shares"),
                t("col_signal", lang): tr_signal(str(tk.get("action_signal", "")), lang),
                t("col_ai", lang): lr.get("conviction_verdict", "—"),
            }
        )
    if rows:
        df_board = pd.DataFrame(rows).sort_values(t("col_score", lang), ascending=False)
        st.dataframe(df_board, use_container_width=True, hide_index=True, height=560)


# ==============================================================================
# TAB 3: POSITIONS BOOK & GPU 100K MONTE CARLO RISK
# ==============================================================================
def _render_portfolio_risk_tab(
    payload: dict[str, Any] | None,
    portfolio_cfg: dict[str, Any],
    lang: str,
) -> None:
    st.subheader(t("pos_hdr", lang))
    with st.expander(f"{t('edit_pos_exp', lang)}", expanded=False):
        acct = portfolio_cfg.get("account") or {}
        c1, c2, c3, c4 = st.columns(4)
        new_eq = c1.number_input(t("lbl_equity", lang), value=float(acct.get("total_equity_usd", 100000.0)), step=5000.0)
        new_cash = c2.number_input(t("lbl_cash", lang), value=float(acct.get("cash_usd", 15000.0)), step=1000.0)
        new_risk = c3.number_input(t("lbl_risk_pct", lang), value=float(acct.get("max_risk_per_trade_pct", 1.25)), step=0.25, min_value=0.25, max_value=5.0)
        new_cap = c4.number_input(t("lbl_max_pos", lang), value=float(acct.get("max_single_position_pct", 20.0)), step=2.5, min_value=5.0, max_value=50.0)

        h_df = pd.DataFrame(portfolio_cfg.get("holdings") or [])
        if h_df.empty:
            h_df = pd.DataFrame(columns=["symbol", "shares", "avg_cost", "stop_loss", "thesis"])
        edited_df = st.data_editor(h_df, num_rows="dynamic", use_container_width=True)
        if st.button(f"{t('btn_save_pos', lang)}", type="primary"):
            cleaned = []
            for _, row in edited_df.iterrows():
                sym = str(row.get("symbol", "")).upper().strip()
                if sym:
                    cleaned.append(
                        {
                            "symbol": sym,
                            "shares": float(row.get("shares") or 0.0),
                            "avg_cost": float(row.get("avg_cost") or 0.0),
                            "stop_loss": float(row.get("stop_loss") or 0.0),
                            "thesis": str(row.get("thesis") or ""),
                        }
                    )
            save_portfolio(
                {
                    "account": {
                        **acct,
                        "total_equity_usd": new_eq,
                        "cash_usd": new_cash,
                        "max_risk_per_trade_pct": new_risk,
                        "max_single_position_pct": new_cap,
                    },
                    "holdings": cleaned,
                }
            )
            st.success(t("saved_pos_ok", lang))

    if not payload:
        return

    h_diag = payload.get("holdings_diagnosis") or []
    if h_diag:
        disp_rows = []
        for h in h_diag:
            disp_rows.append(
                {
                    t("col_sym", lang): h.get("symbol"),
                    t("col_price", lang): h.get("close"),
                    t("col_1d", lang): h.get("chg_1d_pct"),
                    t("col_weight", lang): h.get("weight_pct"),
                    t("col_pnl", lang): h.get("pnl_pct"),
                    t("col_score", lang): h.get("composite_score"),
                    t("col_stop", lang): h.get("active_stop"),
                    t("col_trail", lang): h.get("trailing_stop"),
                    t("col_dist_stop", lang): h.get("dist_to_stop_pct"),
                    t("col_rc", lang): h.get("risk_contribution_pct"),
                    t("col_action", lang): tr_holding_action(str(h.get("action", "")), lang),
                }
            )
        st.dataframe(pd.DataFrame(disp_rows), use_container_width=True, hide_index=True)

    risk_mc = payload.get("risk_mc") or {}
    if not risk_mc.get("available"):
        st.warning(risk_mc.get("reason", "Monte Carlo unavailable."))
        return

    st.divider()
    st.subheader(
        t(
            "mc_hdr",
            lang,
            sims=risk_mc.get("n_simulations", 0),
            df=risk_mc.get("student_t_df", 5),
            ms=risk_mc.get("elapsed_ms", 0),
        )
    )
    h21 = risk_mc.get("horizon_21d") or {}
    h42 = risk_mc.get("horizon_42d") or {}

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(t("mc_gross", lang), f"{risk_mc.get('gross_exposure_pct')}%", f"NAV {_fmt_usd(risk_mc.get('total_nav_usd'), 0)}")
    k2.metric(t("mc_vol", lang), f"{risk_mc.get('portfolio_vol_ann_pct')}%", f"Div {risk_mc.get('diversification_ratio')}x")
    k3.metric(t("mc_var", lang), f"{h21.get('var_95_pct')}%", _fmt_usd(h21.get("var_95_usd"), 0))
    k4.metric(t("mc_cvar", lang), f"{h21.get('cvar_95_pct')}%", _fmt_usd(h21.get("cvar_95_usd"), 0))
    k5.metric(t("mc_pdd", lang), f"{h42.get('prob_loss_gt_10pct')}%", f"Gain>10%: {h42.get('prob_gain_gt_10pct')}%")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_mc_distribution_chart(risk_mc, lang=lang), use_container_width=True)
    with c2:
        st.plotly_chart(make_risk_contribution_chart(risk_mc.get("risk_breakdown") or [], lang=lang), use_container_width=True)

    c3, c4 = st.columns([1.2, 1.0])
    with c3:
        st.plotly_chart(make_correlation_heatmap(risk_mc.get("correlation_matrix") or {}, lang=lang), use_container_width=True)
    with c4:
        st.markdown(f"#### {t('stress_hdr', lang)}")
        st.dataframe(pd.DataFrame(risk_mc.get("stress_tests") or []), use_container_width=True, hide_index=True)
        st.markdown(f"#### {t('sector_wt_hdr', lang)}")
        sw = risk_mc.get("sector_weights") or {}
        st.dataframe(pd.DataFrame([{"Sector": k, "NAV %": v} for k, v in sw.items()]), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 4: SUPPLY CHAIN & INTRA-SECTOR ROTATION
# ==============================================================================
def _render_supply_chain_tab(payload: dict[str, Any] | None, lang: str) -> None:
    if not payload:
        st.info(t("no_snap_banner", lang))
        return

    sc_rep = (payload.get("llm") or {}).get("supply_chain_report") or {}
    if sc_rep and not sc_rep.get("error") and not sc_rep.get("parse_error"):
        st.subheader(f"{t('sc_ai_hdr', lang)} · {t('sc_temp', lang)}: {sc_rep.get('chain_temperature', '—')}")
        c1, c2 = st.columns(2)
        c1.info(f"**{t('sc_capex', lang)}**: {sc_rep.get('capex_verdict', '')}")
        c2.info(f"**{t('sc_storage', lang)}**: {sc_rep.get('storage_cycle_verdict', '')}")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f"**{t('sc_leadlag_ops', lang)}**")
            for x in sc_rep.get("lead_lag_opportunities") or []:
                st.markdown(f"- {x}")
        with c4:
            st.markdown(f"**{t('sc_redflags', lang)}**")
            for x in sc_rep.get("chain_red_flags") or []:
                st.markdown(f"- {x}")
        st.success(f"**{t('sc_top_pick', lang)}**: {sc_rep.get('top_sector_pick', '')}")
        st.divider()

    st.subheader(f"{t('peer_rank_hdr', lang)}")
    for sec_label, items in (payload.get("sector_rankings") or {}).items():
        with st.expander(f"{sec_label} ({len(items)})", expanded=True):
            st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)

    p1, p2 = st.columns(2)
    with p1:
        st.subheader(f"{t('pair_hdr', lang)}")
        pairs = payload.get("rotation_pairs") or []
        if pairs:
            st.dataframe(
                pd.DataFrame(pairs)[["pair", "zscore_60d", "corr_60d", "left_21d_pct", "right_21d_pct", "interpretation"]],
                use_container_width=True,
                hide_index=True,
            )
    with p2:
        st.subheader(f"{t('ll_hdr', lang)}")
        ll = payload.get("lead_lag") or []
        if ll:
            st.dataframe(
                pd.DataFrame(ll)[["source", "target", "same_day_corr", "best_lag_days", "best_lag_corr", "divergence_10d_pct"]],
                use_container_width=True,
                hide_index=True,
            )


# ==============================================================================
# TAB 5: GPU BACKTEST & PURGED WALK-FORWARD ML
# ==============================================================================
def _render_backtest_ml_tab(payload: dict[str, Any] | None, lang: str) -> None:
    if not payload:
        st.info(t("no_snap_banner", lang))
        return

    bt = payload.get("backtest") or {}
    st.subheader(f"{t('bt_hdr', lang)}")
    if bt.get("available"):
        best = bt["best_strategy"]
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(t("bt_cagr", lang), f"{best['cagr_pct']}%")
        m2.metric(t("bt_sharpe", lang), f"{best['sharpe']}")
        m3.metric(t("bt_sortino", lang), f"{best['sortino']}")
        m4.metric(t("bt_mdd", lang), f"{best['max_drawdown_pct']}%")
        m5.metric(t("bt_win", lang), f"{best['win_rate_pct']}%", f"{best['total_trades']} trades")
        m6.metric(t("bt_pf", lang), f"{best['profit_factor']}", f"{bt.get('elapsed_ms')}ms")

        c1, c2 = st.columns([1.3, 1.0])
        with c1:
            st.plotly_chart(make_backtest_equity_chart(bt, lang=lang), use_container_width=True)
        with c2:
            st.plotly_chart(make_backtest_heatmap(bt, lang=lang), use_container_width=True)
    else:
        st.info(bt.get("reason", "Backtest skipped."))

    st.divider()
    ml = payload.get("ml_alpha") or {}
    st.subheader(f"{t('ml_hdr', lang)}")
    if ml.get("available"):
        oos = ml.get("oos_validation") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("ml_samples", lang), f"{oos.get('oos_samples', 0):,}")
        c2.metric(t("ml_auc", lang), f"{oos.get('oos_auc', '—')}")
        c3.metric(t("ml_hit", lang), _fmt_pct(oos.get("oos_hit_rate_top_pct"), False))
        c4.metric(t("ml_spread", lang), _fmt_pct(oos.get("oos_top_vs_bottom_spread_pct")))
        if ml.get("top_features"):
            st.markdown(f"##### {t('ml_feats', lang)}")
            st.dataframe(pd.DataFrame(ml["top_features"]), use_container_width=True, hide_index=True)
    else:
        st.info(ml.get("reason", "ML module skipped."))


# ==============================================================================
# TAB 6: 4-STAGE RTX 5070 Ti GPU GRID TRADING LAB
# ==============================================================================
def _render_grid_lab_tab(
    settings: Any,
    universe_cfg: dict[str, Any],
    portfolio_cfg: dict[str, Any],
    db: TradingDB,
    lang: str,
) -> None:
    is_zh = lang == "zh"
    st.markdown(f"### {t('grid_title', lang)}")

    all_eq = get_all_equity_symbols(universe_cfg)
    holding_map = {
        str(h.get("symbol", "")).upper().strip(): float(h.get("shares") or 0.0)
        for h in (portfolio_cfg.get("holdings") or [])
        if h.get("symbol")
    }
    watchlist_syms = list(dict.fromkeys(all_eq + list(holding_map.keys())))

    # ------------------------------------------------------------------
    # STAGE 1: WATCHLIST GRID SUITABILITY SCREENER (PyTorch CUDA)
    # ------------------------------------------------------------------
    s1_c1, s1_c2 = st.columns([3.2, 1.3], vertical_alignment="center")
    with s1_c1:
        st.subheader(t("grid_s1_hdr", lang))
    with s1_c2:
        run_s1 = st.button(t("grid_s1_btn", lang), type="primary", use_container_width=True, key="btn_grid_s1")

    if run_s1:
        with st.spinner("Running PyTorch CUDA Hurst Exponent + Liquidity + 1% Grid Sweep..." if not is_zh else "正在调用 PyTorch CUDA 并行计算赫斯特指数、振幅流动性与 1% 基准网格频率..."):
            md = MarketDataFetcher(settings.data_dir, history_years=settings.history_years)
            ohlcv_map, _ = md.fetch_universe(universe_cfg, extra_symbols=list(holding_map.keys()))
            s1_res = screen_watchlist_for_grid(
                ohlcv_map=ohlcv_map,
                symbols=watchlist_syms,
                universe_cfg=universe_cfg,
                device_pref=settings.compute_device,
            )
            st.session_state["grid_stage1"] = s1_res
            db.set_cache("grid_lab:stage1", s1_res)

    s1_res = st.session_state.get("grid_stage1") or db.get_cache("grid_lab:stage1", max_age_hours=168.0)

    if s1_res:
        # Honest Top Verdict Banner (turns warning/error if 0 stocks truly qualify!)
        summary_msg = s1_res["summary_zh"] if is_zh else s1_res["summary_en"]
        if s1_res.get("suitable_count", 0) > 0:
            st.success(summary_msg)
        elif s1_res.get("conditional_count", 0) > 0:
            st.warning(summary_msg)
        else:
            st.error(summary_msg)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Suitable (Score>=72)" if not is_zh else "适合跑网格 (>=72)", s1_res.get("suitable_count", 0))
        m2.metric("Conditional (56-71)" if not is_zh else "谨慎可用 (56-71)", s1_res.get("conditional_count", 0))
        m3.metric("Scanned Tickers" if not is_zh else "扫描标的数", len(s1_res.get("candidates") or []))
        m4.metric("CUDA Backend" if not is_zh else "计算耗时", f"{s1_res.get('elapsed_ms', 0)} ms", str(s1_res.get("backend", "")).upper())

        st.plotly_chart(make_grid_suitability_scatter(s1_res.get("candidates") or [], lang=lang), use_container_width=True)

        cands = s1_res.get("candidates") or []
        if cands:
            df_s1 = pd.DataFrame(
                [
                    {
                        ("代码" if is_zh else "Ticker"): r["symbol"],
                        ("客观评分" if is_zh else "Grid Score"): r["grid_score"],
                        ("客观结论" if is_zh else "Verdict"): r["verdict_zh"] if is_zh else r["verdict_en"],
                        ("赫斯特 H" if is_zh else "Hurst H"): r["hurst"],
                        ("半衰期(天)" if is_zh else "Half-Life(d)"): r["half_life_days"],
                        "ADX14": r["adx14"],
                        ("日内振幅%" if is_zh else "Amp %"): r["intraday_amp_pct"],
                        ("隔夜跳空P95%" if is_zh else "Gap P95%"): r["gap_p95_pct"],
                        ("日均成交额$M" if is_zh else "Liq $M/d"): r["dollar_vol_m"],
                        ("1%年化套利次" if is_zh else "1% Ann. Trades"): r["annual_round_trips_1pct"],
                        ("区间内占比%" if is_zh else "In-Range %"): r["in_range_pct"],
                        ("客观诊断" if is_zh else "Rationale"): r["reason_zh"] if is_zh else r["reason_en"],
                    }
                    for r in cands
                ]
            )
            st.dataframe(df_s1, use_container_width=True, hide_index=True, height=390)
    else:
        st.info(
            "Click **Scan Watchlist for Grid Suitability** to objectively score every ticker via PyTorch CUDA Hurst Exponent, Liquidity/Gap Risk, and 1% Grid Frequency."
            if not is_zh
            else "点击上方 **一键扫描 Watchlist 网格适配度**，调用 PyTorch CUDA 对自选股全部做客观打分（若都不适合会直接红字劝退）。"
        )

    st.divider()

    # ------------------------------------------------------------------
    # STAGES 2 -> 3 -> 4 DEEP GPU GRID PIPELINE LAUNCHER
    # ------------------------------------------------------------------
    default_ranked = [r["symbol"] for r in (s1_res or {}).get("candidates", []) if r.get("hurst") is not None] or watchlist_syms
    ctl1, ctl2, ctl3, ctl4, ctl5, ctl6 = st.columns([1.15, 1.05, 0.95, 0.95, 1.45, 1.35], vertical_alignment="bottom")

    with ctl1:
        target_sym = st.selectbox(t("grid_pick_sym", lang), options=default_ranked, key="grid_target_sym")
    with ctl2:
        grid_budget = st.number_input(t("grid_budget", lang), min_value=1000.0, value=20000.0, step=2500.0, key="grid_budget_in")
    with ctl3:
        grid_held = st.number_input(
            t("grid_held", lang),
            min_value=0.0,
            value=float(holding_map.get(target_sym, 0.0)),
            step=1.0,
            key=f"grid_held_{target_sym}",
        )
    with ctl4:
        break_lim = st.number_input(t("grid_break_lim", lang), min_value=1.0, max_value=30.0, value=5.0, step=1.0, key="grid_break_lim_in")
    with ctl5:
        fidelity = st.selectbox(
            t("grid_fidelity", lang),
            options=["TURBO_30M", "ULTRA_15M"],
            format_func=lambda x: (
                "5070 Ti Turbo (60k Cube + 50k 30m)" if x == "TURBO_30M" else "5070 Ti Ultra (60k Cube + 50k 15m)"
            ),
            key="grid_fidelity_sel",
        )
    with ctl6:
        run_deep = st.button(t("grid_run_deep", lang), type="primary", use_container_width=True, key="btn_grid_deep")

    if run_deep:
        pbar = st.progress(0.0)
        st_box = st.empty()

        def _gcb(pct: float, msg: str) -> None:
            pbar.progress(min(1.0, max(0.0, pct)))
            st_box.info(msg)

        md = MarketDataFetcher(settings.data_dir, history_years=settings.history_years)
        ohlcv_map, _ = md.fetch_universe(universe_cfg, extra_symbols=[target_sym])
        deep_res = run_complete_grid_lab_for_symbol(
            symbol=target_sym,
            ohlcv_map=ohlcv_map,
            n_mc_paths=10_000,
            steps_per_day=26 if fidelity == "ULTRA_15M" else 13,
            break_rate_limit_pct=float(break_lim),
            high_density_gpu=True,
            device_pref=settings.compute_device,
            progress_cb=_gcb,
        )
        st.session_state[f"grid_deep:{target_sym}"] = deep_res
        if deep_res.get("available"):
            db.set_cache(f"grid_lab:deep:{target_sym}", deep_res)

    deep_res = st.session_state.get(f"grid_deep:{target_sym}") or db.get_cache(
        f"grid_lab:deep:{target_sym}", max_age_hours=168.0
    )

    if not deep_res or not deep_res.get("available"):
        st.caption(
            f"Select any ticker above and click **{t('grid_run_deep', lang)}** to run Stages 2–4 (4D Hyper-Cube + SciPy Plateau + 50,000 Black-Swan Sims + XGBoost/LightGBM Crash Switch)."
            if not is_zh
            else f"选择上方标的并点击 **{t('grid_run_deep', lang)}**，即可调用 5070 Ti 一键跑完阶段二（6万组4D参数高原）→ 阶段三（5万次黑天鹅破网压测）→ 阶段四（ML智能开关与挂单梯）。"
        )
        return

    s2 = deep_res["stage2_hypercube"]
    s3 = deep_res["stage3_montecarlo"]
    s4 = deep_res["stage4_ml_switch"]

    # ------------------------------------------------------------------
    # STAGE 2: 4D HYPER-CUBE + SCIPY GAUSSIAN PLATEAU DISPLAY
    # ------------------------------------------------------------------
    st.subheader(t("grid_s2_hdr", lang))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Combos Evaluated" if not is_zh else "并行测试参数组合", f"{s2['total_combos_evaluated']:,}", s2["hypercube_shape"])
    k2.metric("Winning Grid Type" if not is_zh else "胜出网格类型", str(s2["winning_type"]).upper())
    k3.metric("Live Anchor / ATR14" if not is_zh else "锚定现价 / ATR14", f"${s2['anchor_price_live']}", f"ATR ${s2['live_atr14']}")
    k4.metric("GPU Compute Time" if not is_zh else "超立方体耗时", f"{s2['elapsed_ms']} ms", str(s2["backend"]).upper())

    # Arithmetic vs Geometric vs ATR Dynamic Side-by-Side Comparison
    st.dataframe(
        pd.DataFrame(
            [
                {
                    ("网格类型" if is_zh else "Grid Type"): r["label_zh"] if is_zh else r["label_en"],
                    ("组合数" if is_zh else "Combos"): r["combos"],
                    ("高原稳健分" if is_zh else "Plateau Score"): r["best_plateau_score"],
                    "Sharpe": r["best_sharpe"],
                    ("总收益%" if is_zh else "Return %"): r["best_return_pct"],
                    ("最大回撤%" if is_zh else "Max DD %"): r["best_max_dd_pct"],
                    ("资金利用率%" if is_zh else "Cap Util %"): r["best_util_pct"],
                    ("盈利空间占比%" if is_zh else "Profitable Space %"): r["profitable_space_pct"],
                }
                for r in s2["type_comparison"]
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    hm_type = st.radio(
        "Inspect 4D Slice (Raw Sharpe vs SciPy Gaussian Plateau)" if not is_zh else "切换查看不同网格类型的「原始夏普 vs SciPy 高斯高原」热力图",
        options=["atr_dynamic", "geometric", "arithmetic"],
        index=["atr_dynamic", "geometric", "arithmetic"].index(s2["winning_type"])
        if s2["winning_type"] in ("atr_dynamic", "geometric", "arithmetic")
        else 0,
        horizontal=True,
    )
    if hm_type in (s2.get("heatmaps") or {}):
        st.plotly_chart(
            make_plateau_comparison_heatmaps(s2["heatmaps"][hm_type], hm_type, lang=lang),
            use_container_width=True,
        )

    st.markdown("**Top 5 Distinct Parameter Plateau Champions**" if not is_zh else "**SciPy 4D 高斯滤波精选 Top 5 参数高原组合**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "ID": c["config_id"],
                    ("网格类型" if is_zh else "Type"): c["type_label_zh"] if is_zh else c["type_label_en"],
                    ("间距" if is_zh else "Spacing"): c["spacing_display"],
                    ("区间(现价)" if is_zh else "Live Range"): f"${c['lower_bound_live']} – ${c['upper_bound_live']} (±{c['range_pct']}%)",
                    ("仓位配比" if is_zh else "Sizing"): f"{c['sizing_style']} ({c['martingale_r']}x)",
                    ("止损线" if is_zh else "Stop DD"): "None" if c["stop_loss_dd_pct"] is None else f"{c['stop_loss_dd_pct']:.0f}%",
                    ("高原分" if is_zh else "Plateau"): c["plateau_score"],
                    ("邻域波动" if is_zh else "Local Std"): c["neighborhood_std"],
                    "Sharpe": c["sharpe"],
                    ("回测收益%" if is_zh else "Hist Ret%"): c["total_return_pct"],
                    ("回测回撤%" if is_zh else "Hist DD%"): c["max_drawdown_pct"],
                }
                for c in s2["top_5_plateaus"]
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ------------------------------------------------------------------
    # STAGE 3: 10,000-PATH BLACK-SWAN MONTE CARLO STRESS TEST (50,000 SIMS)
    # ------------------------------------------------------------------
    st.subheader(t("grid_s3_hdr", lang))
    if s3 and s3.get("available"):
        if s3.get("any_passed"):
            st.success(
                f"Champion `{s3['champion_config_id']}` PASSED the < {s3['break_rate_limit_pct']}% Break-Rate stress gate across {s3['total_simulations']:,} parallel-universe simulations ({s3['elapsed_ms']}ms on {str(s3['backend']).upper()})."
                if not is_zh
                else f"最优组合 `{s3['champion_config_id']}` 在 {s3['total_simulations']:,} 次平行宇宙模拟中通过 < {s3['break_rate_limit_pct']}% 破网率底线（{str(s3['backend']).upper()} 耗时 {s3['elapsed_ms']}ms）。"
            )
        else:
            st.error(
                f"WARNING: ALL Top-5 configs EXCEEDED your {s3['break_rate_limit_pct']}% break-rate limit under 10,000 parallel futures! Widen grid range, lower leverage, or rely strictly on the Stage-4 ML Crash Switch."
                if not is_zh
                else f"严格风控预警：Top 5 候选组合在 10,000 条平行宇宙中的破网率均高于 {s3['break_rate_limit_pct']}% 底线！请拓宽区间或强制结合阶段四 ML 防暴跌开关，切勿盲目满仓跑网格。"
            )

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "ID": r["config_id"],
                        ("间距" if is_zh else "Spacing"): r["spacing_display"],
                        ("区间±%" if is_zh else "Range ±%"): f"±{r['range_pct']}%",
                        ("配比" if is_zh else "Sizing"): r["sizing_style"],
                        ("常规破网率%" if is_zh else "Empirical Break%"): r["empirical_break_rate_pct"],
                        ("含黑天鹅破网%" if is_zh else "All-Scenario Break%"): r["black_swan_break_rate_pct"],
                        ("触发止损%" if is_zh else "Stop-Out%"): r["stop_out_rate_pct"],
                        ("最大浮亏P95%" if is_zh else "Float DD P95%"): r["float_dd_p95_pct"],
                        ("最大浮亏P99%" if is_zh else "Float DD P99%"): r["float_dd_p99_pct"],
                        ("1年中位收益%" if is_zh else "Med 1Y Ret%"): r["median_1y_return_pct"],
                        ("盈利概率%" if is_zh else "Prob Profit%"): r["prob_profit_pct"],
                        ("过关判定" if is_zh else "Gate Verdict"): r["gate_badge_zh"] if is_zh else r["gate_badge_en"],
                    }
                    for r in s3["results"]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.plotly_chart(
            make_grid_mc_fan_chart(s3, deep_res["champion_config"], lang=lang),
            use_container_width=True,
        )

    st.divider()

    # ------------------------------------------------------------------
    # STAGE 4: XGBOOST/LIGHTGBM CRASH SWITCH + LIVE BROKER ORDER MONITOR
    # ------------------------------------------------------------------
    st.subheader(t("grid_s4_hdr", lang))
    ml_state = "ACTIVE_GREEN"
    if s4 and s4.get("available"):
        ml_state = str(s4.get("state_code", "ACTIVE_GREEN"))
        badge = s4["badge_zh"] if is_zh else s4["badge_en"]
        directive = s4["directive_zh"] if is_zh else s4["directive_en"]

        if ml_state == "PAUSE_RED":
            st.error(f"### {badge}\n{directive}")
        elif ml_state == "CAUTION_AMBER":
            st.warning(f"### {badge}\n{directive}")
        else:
            st.success(f"### {badge}\n{directive}")

        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        sm1.metric("5D Crash Prob (Ens)" if not is_zh else "未来5日单边暴跌概率", f"{s4['crash_prob_5d_pct']}%")
        sm2.metric("XGBoost CUDA / LGB", f"{s4['xgb_prob_pct']}% / {s4['lgb_prob_pct']}%")
        sm3.metric("Walk-Forward AUC" if not is_zh else "样本外 OOS AUC", f"{s4.get('oos_auc', '—')}")
        sm4.metric("Crash Recall" if not is_zh else "暴跌捕获率", _fmt_pct(s4.get("oos_crash_recall_pct"), False))
        sm5.metric("Red-Light Avg 5D" if not is_zh else "红灯期实际5日均值", _fmt_pct(s4.get("red_light_avg_5d_ret_pct")))
    else:
        st.info((s4 or {}).get("reason", "ML Switch unavailable."))

    # Pick which Top-5 config to display on the Live Broker Order Monitor (defaults to Stage-3 Champion)
    cfg_choices = s2["top_5_plateaus"]
    champ_idx = next(
        (i for i, c in enumerate(cfg_choices) if c["config_id"] == deep_res["champion_config"]["config_id"]),
        0,
    )
    active_cfg_idx = st.selectbox(
        "Active Grid Blueprint for Today's Limit Orders" if not is_zh else "选择当前实盘挂单方案（默认已选压力测试最优组合）",
        options=range(len(cfg_choices)),
        index=champ_idx,
        format_func=lambda i: (
            f"{cfg_choices[i]['config_id']} | {cfg_choices[i]['spacing_display']} | ±{cfg_choices[i]['range_pct']}% | {cfg_choices[i]['sizing_style']}"
        ),
        key=f"active_grid_blueprint_{target_sym}",
    )
    active_cfg = cfg_choices[active_cfg_idx]

    md_live = MarketDataFetcher(settings.data_dir, history_years=settings.history_years)
    fbs = list(((universe_cfg.get("universe") or {}).get(target_sym) or {}).get("yahoo_fallbacks") or [])
    df_live, _ = md_live.fetch_one(target_sym, fallbacks=fbs)

    # Optional Pre/Post-Market anchor price for today's limit order ladder
    gl_c1, gl_c2 = st.columns([2.6, 1.0], vertical_alignment="center")
    use_ext_anchor = gl_c1.checkbox(
        "Anchor order ladder to live Pre/Post-Market price when available"
        if not is_zh
        else "优先使用盘前/盘后实时价格作为今日挂单锚定基准价",
        value=True,
        key=f"grid_ext_anchor_{target_sym}",
    )
    if gl_c2.button(
        "Refresh Pre/Post Price" if not is_zh else "刷新盘前/盘后价",
        use_container_width=True,
        key=f"btn_grid_ext_refresh_{target_sym}",
    ):
        fresh_q = md_live.fetch_live_quote(target_sym, fallbacks=fbs, include_intraday_fallback=True)
        lq_store = dict(st.session_state.get("live_quotes") or {})
        lq_store[target_sym] = fresh_q
        st.session_state["live_quotes"] = lq_store
        st.rerun()

    grid_lq = (st.session_state.get("live_quotes") or {}).get(target_sym)
    reg_px = float(df_live["Close"].iloc[-1]) if not df_live.empty else float(s2["anchor_price_live"])
    if use_ext_anchor and grid_lq and grid_lq.get("latest_price") is not None:
        last_px = float(grid_lq["latest_price"])
        if grid_lq.get("ext_session") and grid_lq.get("ext_price") is not None:
            st.caption(
                f"Anchored to {grid_lq['ext_session']}-MKT ${last_px:,.2f} "
                f"({_fmt_pct(grid_lq.get('ext_change_pct'))} vs Regular ${grid_lq.get('regular_price') or reg_px:,.2f}) "
                f"· Updated {grid_lq.get('fetched_at', '')}"
            )
    else:
        last_px = reg_px
    ladder = build_live_grid_order_ladder(
        last_price=last_px,
        live_atr=float(s2["live_atr14"]),
        config=active_cfg,
        grid_capital_usd=float(grid_budget),
        current_shares_held=float(grid_held),
        ml_state_code=ml_state,
    )

    # Live Bullet & Watermark Strip
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Grid Budget" if not is_zh else "网格总预算", _fmt_usd(ladder["grid_capital_usd"], 0))
    b2.metric("Inventory Watermark" if not is_zh else "当前持仓水位", f"{ladder['inventory_pct']}%", _fmt_usd(ladder["held_value_usd"], 0))
    b3.metric("Remaining Bullets" if not is_zh else "剩余可用现金子弹", _fmt_usd(ladder["remaining_cash_usd"], 0))
    b4.metric("Lower Break Line" if not is_zh else "破网下沿价", _fmt_usd(ladder["lower_bound"]))
    b5.metric("Hard Stop Line" if not is_zh else "极端止损价", _fmt_usd(ladder["hard_stop_price"]))

    col_g1, col_g2 = st.columns([1.45, 1.05])
    with col_g1:
        if not df_live.empty:
            st.plotly_chart(
                make_live_grid_candlestick(target_sym, df_live, ladder, lang=lang),
                use_container_width=True,
            )
    with col_g2:
        st.markdown(f"**{t('grid_ladder_hdr', lang)} (`{target_sym}`)**")
        ord_df = pd.DataFrame(ladder["orders"]).rename(
            columns={
                "side": "方向" if is_zh else "Side",
                "level": "档位" if is_zh else "Lvl",
                "price": "挂单价($)" if is_zh else "Limit ($)",
                "dist_pct": "距现价%" if is_zh else "Dist %",
                "shares": "股数" if is_zh else "Shs",
                "notional_usd": "金额($)" if is_zh else "Notional",
                "status": "状态" if is_zh else "Status",
            }
        )
        st.dataframe(ord_df, use_container_width=True, hide_index=True, height=445)


# ==============================================================================
# TAB 7: TRADE JOURNAL
# ==============================================================================
def _render_journal_tab(db: TradingDB, lang: str) -> None:
    st.subheader(f"{t('jrn_hdr', lang)}")
    with st.form("add_journal_form", clear_on_submit=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        j_sym = c1.text_input(t("jrn_sym", lang), placeholder="STX").upper().strip()
        j_act = c2.selectbox(t("jrn_act", lang), ["BUY", "SELL", "ADD", "TRIM"])
        j_sh = c3.number_input(t("jrn_sh", lang), min_value=0.0, step=1.0)
        j_px = c4.number_input(t("jrn_px", lang), min_value=0.0, step=0.5)
        j_conv = c5.slider(t("jrn_conv", lang), 1, 5, 4)

        c6, c7, c8 = st.columns(3)
        j_stop = c6.number_input(t("jrn_stop", lang), min_value=0.0, step=0.5)
        j_tgt = c7.number_input(t("jrn_tgt", lang), min_value=0.0, step=0.5)
        tags = (
            ["Mispriced Dip", "PEAD Earnings Drift", "Supply-Chain Catch-Up", "20MA Pullback", "Defensive Rebalance", "Stop / Profit Exit"]
            if lang == "en"
            else ["错杀低吸", "财报后漂移(PEAD)", "产业链补涨", "趋势回踩20MA", "防御调仓", "止损/止盈平仓"]
        )
        j_tag = c8.selectbox(t("jrn_tag", lang), tags)
        j_notes = st.text_area(t("jrn_notes", lang))

        if st.form_submit_button(f"{t('jrn_submit', lang)}", type="primary") and j_sym:
            db.add_journal_entry(
                symbol=j_sym,
                action=j_act,
                shares=j_sh,
                price=j_px,
                stop_loss=j_stop if j_stop > 0 else None,
                target_price=j_tgt if j_tgt > 0 else None,
                setup_tag=j_tag,
                conviction=j_conv,
                notes=j_notes,
            )
            st.success(t("jrn_saved", lang, sym=j_sym))
            st.rerun()

    entries = db.list_journal_entries(limit=100)
    if entries:
        st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)
    else:
        st.caption(t("jrn_empty", lang))


# ==============================================================================
# TAB 7: DESK GUIDE (BILINGUAL)
# ==============================================================================
def _render_guide_tab(dev: Any, lang: str) -> None:
    st.subheader(f"{t('guide_hdr', lang)}")
    if lang == "zh":
        st.markdown(
            f"""
            ### 当前硬件后端：`{dev.summary_str()}`
            - **16GB GDDR7 显存防爆调度**：先跑 PyTorch CUDA 量化/ML/蒙特卡洛/回测 → 自动清空显存 → 再唤醒本地 Ollama 大模型。
            - **每周 2 次随用随开节奏**：周日复盘跑一次（约 3~6 分钟），周中如遇暴跌异动跑一次，看完报告/下载 `.md` 后直接关电脑即可。
            - **顶部滑块随时切换 `EN ⇄ 中文`**（默认英文专业券商界面）。
            """
        )
    else:
        st.markdown(
            f"""
            ### Active Hardware Backend: `{dev.summary_str()}`
            - **16GB GDDR7 VRAM Handoff**: PyTorch CUDA quant, 100k Monte Carlo, ML & 288-combo backtest execute first → VRAM is cleanly purged → Ollama 14B/32B local LLM runs with zero OOM risk.
            - **On-Demand Workflow (No 24/7 PC Needed)**: Boot twice a week (Sunday review + mid-week volatility spikes), click **RUN FULL SCAN**, review your **Trade Desk** & **Execution Tickets**, export `.md`, and shut down.
            - **Instant Bilingual Toggle**: Use the top-right **`EN ⇄ 中文`** slider anytime (defaults to English broker UI; also guides the local 5070 Ti LLM report language on the next scan).
            """
        )


def _fmt_num(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    try:
        return f"{float(val):,.2f}"
    except Exception:
        return str(val)


if __name__ == "__main__":
    main()
