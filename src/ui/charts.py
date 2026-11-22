"""Broker-grade dark Plotly financial charts (TradingView/Webull style, right-axis price, bilingual EN/ZH)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Authentic Broker Carbon Dark Palette (Webull / TradingView / Binance Pro)
DARK_BG = "#090C10"
PANEL_BG = "#0F141C"
GRID_COLOR = "#1C2430"
TEXT_COLOR = "#D9E1EC"
MUTED_TEXT = "#7F8EA3"
GREEN = "#0ECB81"
RED = "#F6465D"
AMBER = "#F0B90B"
BLUE = "#2962FF"
PURPLE = "#A371F7"


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


def make_candlestick_trade_chart(
    symbol: str,
    df: pd.DataFrame,
    ticket: dict[str, Any] | None = None,
    lookback_bars: int = 160,
    lang: str = "en",
) -> go.Figure:
    """Pro broker candlestick chart with right-hand price scale, crosshair, SMA 20/50/200 & order zones."""
    sub = df.tail(min(lookback_bars, len(df))).copy()
    close = df["Close"]
    sma20 = close.rolling(20).mean().reindex(sub.index)
    sma50 = close.rolling(50).mean().reindex(sub.index)
    sma200 = close.rolling(200).mean().reindex(sub.index)

    is_zh = lang == "zh"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.78, 0.22],
    )

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
            whiskerwidth=0.4,
        ),
        row=1,
        col=1,
    )

    if sma20.notna().any():
        fig.add_trace(
            go.Scatter(x=sub.index, y=sma20, name="MA20", line=dict(color=AMBER, width=1.2)),
            row=1,
            col=1,
        )
    if sma50.notna().any():
        fig.add_trace(
            go.Scatter(x=sub.index, y=sma50, name="MA50", line=dict(color=BLUE, width=1.4)),
            row=1,
            col=1,
        )
    if sma200.notna().any():
        fig.add_trace(
            go.Scatter(x=sub.index, y=sma200, name="MA200", line=dict(color=PURPLE, width=1.5, dash="dot")),
            row=1,
            col=1,
        )

    # Current Last Price dashed horizontal guide (authentic broker mark)
    if not sub.empty:
        last_px = float(sub["Close"].iloc[-1])
        prev_px = float(sub["Close"].iloc[-2]) if len(sub) >= 2 else last_px
        last_col = GREEN if last_px >= prev_px else RED
        fig.add_hline(
            y=last_px,
            line_color=last_col,
            line_dash="dot",
            line_width=1.0,
            annotation_text=f" {last_px:.2f} ",
            annotation_position="right",
            annotation_bgcolor=last_col,
            annotation_font=dict(color="#090C10", size=10.5),
            row=1,
            col=1,
        )

    if ticket:
        e_low, e_high = ticket.get("entry_low"), ticket.get("entry_high")
        if e_low and e_high:
            lbl_e = f"入场区 ${e_low}–${e_high}" if is_zh else f"ENTRY ${e_low}–${e_high}"
            fig.add_hrect(
                y0=e_low,
                y1=e_high,
                fillcolor="rgba(41, 98, 255, 0.14)",
                line=dict(color="rgba(41, 98, 255, 0.45)", width=1),
                annotation_text=lbl_e,
                annotation_position="top left",
                annotation_font=dict(color="#6CA0FF", size=10.5),
                row=1,
                col=1,
            )
        if ticket.get("hard_stop"):
            lbl_s = (
                f"止损 ${ticket['hard_stop']} (-{ticket.get('stop_dist_pct')}%)"
                if is_zh
                else f"STOP ${ticket['hard_stop']} (-{ticket.get('stop_dist_pct')}%)"
            )
            fig.add_hline(
                y=ticket["hard_stop"],
                line_color=RED,
                line_dash="dash",
                line_width=1.3,
                annotation_text=lbl_s,
                annotation_position="bottom left",
                annotation_font=dict(color=RED, size=10.5),
                row=1,
                col=1,
            )
        if ticket.get("target_1"):
            lbl_t1 = (
                f"目标1 ${ticket['target_1']} ({ticket.get('rr_target_1')}R)"
                if is_zh
                else f"TP1 ${ticket['target_1']} ({ticket.get('rr_target_1')}R)"
            )
            fig.add_hline(
                y=ticket["target_1"],
                line_color=GREEN,
                line_dash="dash",
                line_width=1.2,
                annotation_text=lbl_t1,
                annotation_position="top left",
                annotation_font=dict(color=GREEN, size=10.5),
                row=1,
                col=1,
            )
        if ticket.get("target_2"):
            lbl_t2 = f"目标2 ${ticket['target_2']}" if is_zh else f"TP2 ${ticket['target_2']}"
            fig.add_hline(
                y=ticket["target_2"],
                line_color="#4ADE80",
                line_dash="dot",
                line_width=1.0,
                annotation_text=lbl_t2,
                annotation_position="top left",
                annotation_font=dict(color="#4ADE80", size=10),
                row=1,
                col=1,
            )

    vol_colors = [
        "rgba(14, 203, 129, 0.55)" if c >= o else "rgba(246, 70, 93, 0.55)"
        for o, c in zip(sub["Open"].values, sub["Close"].values)
    ]
    fig.add_trace(
        go.Bar(
            x=sub.index,
            y=sub["Volume"],
            name="成交量" if is_zh else "Vol",
            marker_color=vol_colors,
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(rangeslider_visible=False)
    title_txt = f"{symbol} · 1D OHLCV & 执行区间" if is_zh else f"{symbol} · 1D OHLCV & EXECUTION ZONES"
    _apply_broker_layout(fig, title_txt, height=500, right_yaxis=True)
    return fig


def make_regime_gauge(score: float, label: str) -> go.Figure:
    """Compact broker-style Market Regime radial dial with pointer and active-zone highlight."""
    clamped = max(0.0, min(100.0, float(score)))
    zones = [
        (0, 42, "246, 70, 93", RED),       # Risk-Off / Bearish
        (42, 58, "240, 185, 11", AMBER),   # Choppy / Defensive
        (58, 75, "41, 98, 255", BLUE),     # Selective Bull
        (75, 100, "14, 203, 129", GREEN),  # Bull Expansion
    ]

    active_color = RED
    steps = []
    for lo, hi, rgb, hex_col in zones:
        is_active = (lo <= clamped < hi) or (hi == 100 and clamped >= 75)
        if is_active:
            active_color = hex_col
            steps.append({"range": [lo, hi], "color": f"rgba({rgb}, 0.72)"})
        else:
            steps.append({"range": [lo, hi], "color": f"rgba({rgb}, 0.12)"})

    # Keep the radial pointer fully visible even when score sits right on the 0 or 100 boundary
    pointer_val = min(99.4, max(0.6, clamped))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=clamped,
            number={
                "suffix": " /100",
                "font": {"size": 26, "color": active_color, "family": "JetBrains Mono, monospace"},
            },
            title={"text": label, "font": {"size": 12.5, "color": TEXT_COLOR}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickmode": "array",
                    "tickvals": [0, 25, 42, 58, 75, 100],
                    "tickwidth": 1,
                    "tickcolor": MUTED_TEXT,
                    "tickfont": {"size": 10, "color": MUTED_TEXT},
                },
                "bar": {"color": active_color, "thickness": 0.22},
                "bgcolor": PANEL_BG,
                "borderwidth": 1,
                "bordercolor": GRID_COLOR,
                "steps": steps,
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 4},
                    "thickness": 1.0,
                    "value": pointer_val,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=230,
        margin=dict(l=22, r=22, t=40, b=10),
    )
    return fig


def make_mc_distribution_chart(risk_mc: dict[str, Any], lang: str = "en") -> go.Figure:
    """100k-path Student-t simulated 21-day return distribution with VaR95 & CVaR95."""
    is_zh = lang == "zh"
    hist = risk_mc.get("dist_hist_21d") or {}
    xs = hist.get("bin_centers") or []
    ys = hist.get("counts") or []
    h21 = risk_mc.get("horizon_21d") or {}
    var95 = float(h21.get("var_95_pct", -10.0))
    cvar95 = float(h21.get("cvar_95_pct", -15.0))
    med = float(h21.get("median_return_pct", 0.0))

    colors = [RED if x <= var95 else ("#3B82F6" if x < 0 else GREEN) for x in xs]
    fig = go.Figure(
        go.Bar(x=xs, y=ys, marker_color=colors, name="频次" if is_zh else "Paths", opacity=0.88)
    )
    fig.add_vline(
        x=var95,
        line_color=AMBER,
        line_dash="dash",
        line_width=1.8,
        annotation_text=f"95% VaR {var95:.1f}%",
        annotation_position="top left",
    )
    fig.add_vline(
        x=cvar95,
        line_color=RED,
        line_dash="dot",
        line_width=1.8,
        annotation_text=f"95% CVaR {cvar95:.1f}%",
        annotation_position="top left",
    )
    fig.add_vline(
        x=med,
        line_color=TEXT_COLOR,
        line_dash="solid",
        line_width=1.3,
        annotation_text=("中位 " if is_zh else "Median ") + f"{med:+.1f}%",
        annotation_position="top right",
    )
    fig.update_xaxes(title_text="未来 21 交易日组合收益分布 (%)" if is_zh else "Simulated 21-Day Portfolio Return (%)")
    fig.update_yaxes(title_text="路径数" if is_zh else "Path Count")
    title = (
        f"GPU Student-t 蒙特卡洛分布 ({risk_mc.get('n_simulations', 0):,} 路径)"
        if is_zh
        else f"GPU STUDENT-T MONTE CARLO ({risk_mc.get('n_simulations', 0):,} PATHS)"
    )
    return _apply_broker_layout(fig, title, height=360)


def make_correlation_heatmap(corr_data: dict[str, Any], lang: str = "en") -> go.Figure:
    is_zh = lang == "zh"
    syms = corr_data.get("symbols") or []
    vals = corr_data.get("values") or []
    fig = go.Figure(
        go.Heatmap(
            z=vals,
            x=syms,
            y=syms,
            zmin=-1.0,
            zmax=1.0,
            colorscale="RdBu_r",
            text=[[f"{v:.2f}" for v in row] for row in vals],
            texttemplate="%{text}",
            textfont={"size": 11.5},
        )
    )
    title = (
        "持仓相关系数矩阵 (红=高联动，蓝=对冲)"
        if is_zh
        else "PORTFOLIO CORRELATION MATRIX (RED = CONCENTRATED, BLUE = HEDGE)"
    )
    return _apply_broker_layout(fig, title, height=360)


def make_risk_contribution_chart(risk_breakdown: list[dict[str, Any]], lang: str = "en") -> go.Figure:
    is_zh = lang == "zh"
    syms = [r["symbol"] for r in risk_breakdown]
    wts = [r["weight_nav_pct"] for r in risk_breakdown]
    rcs = [r["risk_contribution_pct"] for r in risk_breakdown]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="资金权重 %" if is_zh else "Capital Weight %", x=syms, y=wts, marker_color=BLUE))
    fig.add_trace(go.Bar(name="真实风险贡献 %" if is_zh else "Risk Contribution %", x=syms, y=rcs, marker_color=RED))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="%" )
    title = (
        "资金权重 vs 边际风险贡献 (MCR)"
        if is_zh
        else "CAPITAL WEIGHT VS. MARGINAL RISK CONTRIBUTION (MCR)"
    )
    return _apply_broker_layout(fig, title, height=360)


def make_backtest_equity_chart(bt: dict[str, Any], lang: str = "en") -> go.Figure:
    is_zh = lang == "zh"
    ec = bt.get("equity_curve") or {}
    dates = ec.get("dates") or []
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ec.get("strategy") or [],
            name="波段策略(扣滑点佣金)" if is_zh else "Swing Strategy (Net Costs)",
            line=dict(color=GREEN, width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ec.get("qqq") or [],
            name="QQQ 基准" if is_zh else "QQQ Buy & Hold",
            line=dict(color=BLUE, width=1.6, dash="dash"),
        )
    )
    if ec.get("soxx"):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=ec.get("soxx"),
                name="SOXX 基准" if is_zh else "SOXX Buy & Hold",
                line=dict(color=AMBER, width=1.5, dash="dot"),
            )
        )
    fig.update_yaxes(title_text="净值倍数" if is_zh else "Equity Multiple (1.0x)")
    title = (
        "历史策略净值曲线 vs QQQ / SOXX"
        if is_zh
        else "STRATEGY EQUITY CURVE VS. QQQ / SOXX"
    )
    return _apply_broker_layout(fig, title, height=390, right_yaxis=True)


def make_backtest_heatmap(bt: dict[str, Any], lang: str = "en") -> go.Figure:
    is_zh = lang == "zh"
    hm = bt.get("heatmap") or {}
    xs = [f"{v}x ATR" for v in (hm.get("x_stop_atr") or [])]
    ys = [f"RS >= {v}" for v in (hm.get("y_rs_threshold") or [])]
    zs = hm.get("z_sharpe") or []

    fig = go.Figure(
        go.Heatmap(
            z=zs,
            x=xs,
            y=ys,
            colorscale="Viridis",
            text=[[f"{v:.2f}" for v in row] for row in zs],
            texttemplate="%{text}",
            textfont={"size": 11.5},
        )
    )
    fig.update_xaxes(title_text="止损宽度 (ATR 倍数)" if is_zh else "Stop Width (ATR Multiple)")
    fig.update_yaxes(title_text="相对强度门槛" if is_zh else "RS Entry Threshold")
    title = (
        "GPU 参数稳健性热力图 (夏普比率)"
        if is_zh
        else "GPU PARAMETER ROBUSTNESS HEATMAP (SHARPE RATIO)"
    )
    return _apply_broker_layout(fig, title, height=390)


def make_pillar_radar(symbol: str, pillars: dict[str, float], lang: str = "en") -> go.Figure:
    is_zh = lang == "zh"
    cats = (
        ["趋势与阶段", "机构量价", "相对强度", "GPU量化胜率", "基本面"]
        if is_zh
        else ["Trend", "Inst. Flow", "Rel. Strength", "GPU Edge", "Fundamentals"]
    )
    vals = [
        float(pillars.get("trend", 50)),
        float(pillars.get("volume_flow", 50)),
        float(pillars.get("relative_strength", 50)),
        float(pillars.get("gpu_quant_edge", 50)),
        float(pillars.get("fundamentals", 50)),
    ]
    cats.append(cats[0])
    vals.append(vals[0])

    fig = go.Figure(
        go.Scatterpolar(
            r=vals,
            theta=cats,
            fill="toself",
            fillcolor="rgba(41, 98, 255, 0.24)",
            line=dict(color=BLUE, width=2),
            name=symbol,
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor=PANEL_BG,
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=GRID_COLOR, tickfont=dict(size=9)),
            angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=10.5)),
        ),
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        showlegend=False,
        height=265,
        margin=dict(l=34, r=34, t=28, b=22),
        title=dict(text=f"{symbol} · " + ("五维体检" if is_zh else "5-PILLAR SCORE"), font=dict(size=12.5)),
    )
    return fig
