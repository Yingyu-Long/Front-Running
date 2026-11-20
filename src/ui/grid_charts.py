"""Broker-grade dark Plotly visualizations for the 4-Stage GPU Grid Trading Lab (EN/ZH)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DARK_BG = "#090C10"
PANEL_BG = "#0F141C"
GRID_COLOR = "#1C2430"
TEXT_COLOR = "#D9E1EC"
GREEN = "#0ECB81"
RED = "#F6465D"
AMBER = "#F0B90B"
BLUE = "#2962FF"


def _apply_broker_layout(fig: go.Figure, title: str, height: int = 440, right_yaxis: bool = False) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=13.5, color=TEXT_COLOR), x=0.01),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR, family="JetBrains Mono, SF Mono, Inter, -apple-system, sans-serif", size=11.5),
        height=height,
        margin=dict(l=16 if right_yaxis else 46, r=58 if right_yaxis else 20, t=42, b=32),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        showgrid=True,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#3A475A",
        spikethickness=1,
        spikedash="dot",
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        showgrid=True,
        side="right" if right_yaxis else "left",
        showspikes=True,
        spikemode="across",
        spikecolor="#3A475A",
        spikethickness=1,
        spikedash="dot",
    )
    return fig


def make_grid_suitability_scatter(
    candidates: list[dict[str, Any]],
    lang: str = "en",
) -> go.Figure:
    """Quadrant map: Hurst Exponent (H) vs Annualized 1% Grid Round-Trips (H < 0.50 = Grid Sweet Spot)."""
    is_zh = lang == "zh"
    valid = [c for c in candidates if c.get("hurst") is not None]

    fig = go.Figure()
    if not valid:
        return _apply_broker_layout(fig, "Watchlist Grid Map", height=360)

    # Shaded Mean-Reversion Zone (H < 0.50) vs Trend Danger Zone (H > 0.55)
    fig.add_vrect(
        x0=0.20,
        x1=0.50,
        fillcolor="rgba(14, 203, 129, 0.08)",
        line_width=0,
        annotation_text="均值回归区 (H < 0.50 · 适合网格)" if is_zh else "MEAN-REVERTING ZONE (H < 0.50 · GRID IDEAL)",
        annotation_position="top left",
        annotation_font=dict(color=GREEN, size=10.5),
    )
    fig.add_vrect(
        x0=0.55,
        x1=0.85,
        fillcolor="rgba(246, 70, 93, 0.08)",
        line_width=0,
        annotation_text="强趋势危险区 (H > 0.55 · 网格必死)" if is_zh else "TREND DANGER ZONE (H > 0.55 · AVOID GRID)",
        annotation_position="top right",
        annotation_font=dict(color=RED, size=10.5),
    )
    fig.add_vline(x=0.50, line_color=AMBER, line_dash="dash", line_width=1.4)

    colors = [
        GREEN if c["verdict_code"] == "SUITABLE" else (AMBER if c["verdict_code"] == "CONDITIONAL" else RED)
        for c in valid
    ]
    sizes = [max(14, min(38, 12 + float(c.get("grid_score", 0)) * 0.25)) for c in valid]

    fig.add_trace(
        go.Scatter(
            x=[c["hurst"] for c in valid],
            y=[c["annual_round_trips_1pct"] for c in valid],
            mode="markers+text",
            text=[c["symbol"] for c in valid],
            textposition="top center",
            textfont=dict(size=11, color=TEXT_COLOR),
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=1.5, color="#090C10"),
                opacity=0.92,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                + ("赫斯特 H: %{x:.3f}<br>年化1%网格成交: %{y:.0f}次<br>" if is_zh else "Hurst H: %{x:.3f}<br>Annual 1% Round-Trips: %{y:.0f}<br>")
                + "Score: %{customdata[0]}/100<br>Amp: %{customdata[1]}% | Gap P95: %{customdata[2]}%<extra></extra>"
            ),
            customdata=[[c["grid_score"], c["intraday_amp_pct"], c["gap_p95_pct"]] for c in valid],
            showlegend=False,
        )
    )

    fig.update_xaxes(
        title_text="赫斯特指数 Hurst Exponent (H < 0.5 均值回归, H > 0.5 强趋势)" if is_zh else "Hurst Exponent (H < 0.5 Mean-Reverting · H > 0.5 Trending)",
        range=[0.28, 0.78],
    )
    fig.update_yaxes(
        title_text="基准 1% 网格年化套利次数" if is_zh else "Baseline 1% Grid Annualized Round-Trips",
    )
    title = (
        "阶段一 · Watchlist 网格适配度象限图 (气泡越大评分越高)"
        if is_zh
        else "STAGE 1 · WATCHLIST GRID SUITABILITY MAP (HURST H VS. ANNUAL ROUND-TRIPS)"
    )
    return _apply_broker_layout(fig, title, height=400)


def make_plateau_comparison_heatmaps(
    hm: dict[str, Any],
    grid_type: str,
    lang: str = "en",
) -> go.Figure:
    """Side-by-side Raw Sharpe vs SciPy 4D Gaussian-Filtered Plateau Slice (Spacing x Range)."""
    is_zh = lang == "zh"
    xs = [f"±{v}%" for v in (hm.get("x_range_pct") or [])]
    unit = hm.get("y_unit", "%")
    ys = [f"{v}{unit}" for v in (hm.get("y_spacing") or [])]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            ("原始夏普热力图 (易含过拟合孤岛)" if is_zh else "RAW SHARPE SLICE (OVERFIT SPIKES)"),
            ("SciPy 4D 高斯滤波参数高原 (稳健盈利区)" if is_zh else "SCIPY 4D GAUSSIAN PLATEAU (ROBUST ZONE)"),
        ),
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Heatmap(
            z=hm.get("z_raw_sharpe") or [],
            x=xs,
            y=ys,
            colorscale="Viridis",
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=hm.get("z_plateau_score") or [],
            x=xs,
            y=ys,
            colorscale="Turbo",
            colorbar=dict(title="Plateau" if not is_zh else "高原分", thickness=12),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="区间上下限宽度" if is_zh else "Grid Range Width (±%)", row=1, col=1)
    fig.update_xaxes(title_text="区间上下限宽度" if is_zh else "Grid Range Width (±%)", row=1, col=2)
    fig.update_yaxes(title_text=f"网格间距 ({unit})" if is_zh else f"Grid Spacing ({unit})", row=1, col=1)

    mg = hm.get("best_slice_martingale")
    st = hm.get("best_slice_stop_pct")
    stop_txt = "No Stop" if (st is not None and float(st) <= -90) else f"{st}%"
    title = (
        f"阶段二 · {grid_type.upper()} 4D 超立方体切片 (马丁比={mg} · 止损={stop_txt})"
        if is_zh
        else f"STAGE 2 · {grid_type.upper()} 4D HYPER-CUBE SLICE (MARTINGALE={mg} · STOP={stop_txt})"
    )
    return _apply_broker_layout(fig, title, height=400)


def make_grid_mc_fan_chart(
    stage3: dict[str, Any],
    champion_cfg: dict[str, Any],
    lang: str = "en",
) -> go.Figure:
    """Plot 24 parallel-universe Black-Swan price trajectories against the Champion Grid Bounds."""
    is_zh = lang == "zh"
    paths = stage3.get("sample_paths_daily") or []
    scens = stage3.get("sample_scenarios") or []

    scen_colors = {
        0: ("rgba(88, 166, 255, 0.38)", "Normal Jump-Diff" if not is_zh else "常规跳跃扩散 (60%)"),
        1: ("rgba(246, 70, 93, 0.72)", "Flash Crash -30%" if not is_zh else "黑天鹅闪崩 -30% (15%)"),
        2: ("rgba(240, 185, 11, 0.65)", "Slow Bleed" if not is_zh else "长期阴跌横盘 (15%)"),
        3: ("rgba(14, 203, 129, 0.65)", "V-Reversal" if not is_zh else "V型深跌反转 (10%)"),
    }

    fig = go.Figure()
    shown_leg: set[int] = set()
    for path, sc in zip(paths, scens):
        col, name = scen_colors.get(int(sc), ("rgba(150,150,150,0.4)", "Path"))
        show_l = int(sc) not in shown_leg
        shown_leg.add(int(sc))
        fig.add_trace(
            go.Scatter(
                x=list(range(len(path))),
                y=path,
                mode="lines",
                line=dict(color=col, width=1.4 if sc != 0 else 1.0),
                name=name,
                showlegend=show_l,
                hoverinfo="skip",
            )
        )

    lb = champion_cfg.get("lower_bound_live")
    ub = champion_cfg.get("upper_bound_live")
    if ub:
        fig.add_hline(
            y=ub,
            line_color=GREEN,
            line_dash="dash",
            line_width=1.6,
            annotation_text=f"UPPER ${ub}" if not is_zh else f"网格上沿 ${ub}",
            annotation_position="top right",
            annotation_font=dict(color=GREEN, size=10.5),
        )
    if lb:
        fig.add_hline(
            y=lb,
            line_color=RED,
            line_dash="dash",
            line_width=1.8,
            annotation_text=f"LOWER BREAK LINE ${lb}" if not is_zh else f"破网下沿 ${lb}",
            annotation_position="bottom right",
            annotation_font=dict(color=RED, size=10.5),
        )

    fig.update_xaxes(title_text="模拟未来交易日 (1年 = 252日)" if is_zh else "Simulated Future Trading Days (1 Year = 252d)")
    fig.update_yaxes(title_text="模拟股价 ($)" if is_zh else "Simulated Price ($)")
    title = (
        f"阶段三 · 平行宇宙黑天鹅压力路径抽样 (总计 {stage3.get('total_simulations', 0):,} 次 GPU 模拟)"
        if is_zh
        else f"STAGE 3 · BLACK-SWAN PARALLEL UNIVERSE PATHS ({stage3.get('total_simulations', 0):,} GPU SIMS)"
    )
    return _apply_broker_layout(fig, title, height=370, right_yaxis=True)


def make_live_grid_candlestick(
    symbol: str,
    df: pd.DataFrame,
    ladder: dict[str, Any],
    lang: str = "en",
) -> go.Figure:
    """Broker candlestick chart overlaid with live Buy/Sell Grid Limit Rungs & Lower Break Stop."""
    is_zh = lang == "zh"
    sub = df.tail(min(90, len(df))).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=sub.index,
            open=sub["Open"],
            high=sub["High"],
            low=sub["Low"],
            close=sub["Close"],
            name=symbol,
            increasing_line_color=GREEN,
            increasing_fillcolor=GREEN,
            decreasing_line_color=RED,
            decreasing_fillcolor=RED,
        )
    )

    # Buy Limit Rungs (emerald)
    for idx, p in enumerate((ladder.get("buy_prices") or [])[:8], start=1):
        fig.add_hline(
            y=p,
            line_color="rgba(14, 203, 129, 0.55)",
            line_dash="dot",
            line_width=1.0,
            annotation_text=f"BUY -{idx}: ${p:.2f}",
            annotation_position="left",
            annotation_font=dict(color=GREEN, size=9.5),
        )

    # Sell Limit Rungs (amber)
    for idx, p in enumerate((ladder.get("sell_prices") or [])[:6], start=1):
        fig.add_hline(
            y=p,
            line_color="rgba(240, 185, 11, 0.55)",
            line_dash="dot",
            line_width=1.0,
            annotation_text=f"SELL +{idx}: ${p:.2f}",
            annotation_position="left",
            annotation_font=dict(color=AMBER, size=9.5),
        )

    # Lower Break & Hard Stop
    if ladder.get("lower_bound"):
        fig.add_hline(
            y=ladder["lower_bound"],
            line_color=RED,
            line_dash="dash",
            line_width=1.6,
            annotation_text=f"LOWER BREAK ${ladder['lower_bound']}" if not is_zh else f"破网警戒线 ${ladder['lower_bound']}",
            annotation_position="bottom right",
            annotation_font=dict(color=RED, size=10.5),
        )

    fig.update_xaxes(rangeslider_visible=False)
    title = (
        f"阶段四 · {symbol} 实盘网格挂单可视化 (绿=低吸挂单 / 黄=高抛挂单 / 红=破网线)"
        if is_zh
        else f"STAGE 4 · {symbol} LIVE GRID LIMIT LADDER (GREEN=BUY / AMBER=SELL / RED=BREAK)"
    )
    return _apply_broker_layout(fig, title, height=470, right_yaxis=True)
