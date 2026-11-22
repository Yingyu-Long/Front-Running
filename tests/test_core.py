"""Automated verification suite: AST syntax check across entire codebase + core unit tests.

Runs cleanly even on a bare Python stdlib environment (and runs full numerical tests when numpy/pandas are installed).
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.llm_client import extract_json_object, strip_thinking_tags
from src.ai.prompts import get_prompts
from src.storage.db import TradingDB
from src.ui.i18n import (
    STRINGS,
    t,
    tr_crash_badge,
    tr_earnings_flag,
    tr_holding_action,
    tr_regime_label,
    tr_signal,
)

try:
    import numpy as np
    import pandas as pd

    HAS_NUMPY_PANDAS = True
except ImportError:
    HAS_NUMPY_PANDAS = False


class TestCodebaseSyntax(unittest.TestCase):
    """Verify every single .py file in the repository parses cleanly with zero syntax errors."""

    def test_all_python_files_parse(self) -> None:
        py_files = sorted(
            p
            for p in PROJECT_ROOT.rglob("*.py")
            if ".venv" not in p.parts and "__pycache__" not in p.parts
        )
        self.assertGreaterEqual(len(py_files), 15, "Expected at least 15 Python modules")
        for path in py_files:
            source = path.read_text(encoding="utf-8")
            try:
                ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                self.fail(f"SyntaxError in {path}: {exc}")


class TestSQLitePersistence(unittest.TestCase):
    """Verify snapshot archive, week-over-week previous lookup, cache TTL, and trade journal."""

    def test_snapshots_cache_and_journal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = TradingDB(Path(tmp) / "test.db")

            # Snapshots
            id1 = db.save_snapshot(
                {"regime": {"regime_score": 62.5, "regime_label": "Selective Bull"}, "stocks": {"GOOG": {"composite_score": 70.0}}},
                run_label="week1",
                device_info="RTX 5070 Ti",
                llm_model="qwen3:14b",
            )
            id2 = db.save_snapshot(
                {"regime": {"regime_score": 45.0, "regime_label": "Choppy"}, "stocks": {"GOOG": {"composite_score": 55.0}}},
                run_label="week2",
                device_info="RTX 5070 Ti",
                llm_model="qwen3:14b",
            )
            self.assertGreater(id2, id1)
            latest = db.get_snapshot()
            self.assertIsNotNone(latest)
            self.assertEqual(latest["id"], id2)
            prev = db.get_previous_snapshot(id2)
            self.assertIsNotNone(prev)
            self.assertEqual(prev["id"], id1)

            # Cache
            db.set_cache("sec:GOOG", {"cik": "0001652044"})
            self.assertEqual(db.get_cache("sec:GOOG", max_age_hours=24)["cik"], "0001652044")

            # Journal
            jid = db.add_journal_entry(
                symbol="STX",
                action="BUY",
                shares=30,
                price=511.0,
                stop_loss=465.0,
                target_price=610.0,
                setup_tag="错杀低吸",
                conviction=4,
                notes="测试日志",
            )
            self.assertGreater(jid, 0)
            db.close_journal_entry(jid, outcome_pnl=1250.0, notes_append="按计划止盈")
            entries = db.list_journal_entries()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["status"], "CLOSED")
            self.assertAlmostEqual(entries[0]["outcome_pnl"], 1250.0)


class TestLLMJsonExtraction(unittest.TestCase):
    """Verify Qwen3 / DeepSeek-R1 <think> stripping & structured JSON extraction."""

    def test_strip_think_and_extract_json(self) -> None:
        raw = """<think>
Let's analyze ORCL near 52-week low and check volume climax...
</think>
```json
{
  "conviction_verdict": "择机回踩低吸",
  "confidence_score": 76,
  "one_line_summary": "情绪错杀但需等待放量企稳阳线"
}
```"""
        cleaned = strip_thinking_tags(raw)
        self.assertNotIn("<think>", cleaned)
        parsed = extract_json_object(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["conviction_verdict"], "择机回踩低吸")
        self.assertEqual(parsed["confidence_score"], 76)


class TestBilingualUIAndPrompts(unittest.TestCase):
    """Verify full EN/ZH key parity, dynamic broker badge translation, and bilingual LLM prompts."""

    def test_i18n_key_parity(self) -> None:
        en_keys = set(STRINGS["en"].keys())
        zh_keys = set(STRINGS["zh"].keys())
        self.assertEqual(en_keys, zh_keys, f"Mismatched i18n keys: {en_keys ^ zh_keys}")
        self.assertIn("Trade Desk", t("tab_desk", "en"))
        self.assertIn("交易工作台", t("tab_desk", "zh"))

    def test_dynamic_badges_and_prompts(self) -> None:
        self.assertIn("AVOID KNIFE", tr_signal("观望/禁接飞刀", "en"))
        self.assertIn("禁接飞刀", tr_signal("观望/禁接飞刀", "zh"))
        self.assertIn("Falling Knife", tr_crash_badge({"status": "FALLING_KNIFE", "badge": "警惕接飞刀"}, "en"))
        self.assertIn("CLEAR", tr_earnings_flag({"days_to_earnings": 35}, "en"))
        self.assertIn("STOP HIT", tr_holding_action("触发止损线 · 建议减仓/止损", "en"))
        self.assertIn("BULL EXPANSION", tr_regime_label(80.0, "强势进攻期", "en"))

        from src.ui.i18n import (
            tr_crash_badge_str,
            tr_earnings_flag_str,
            tr_pair_interpretation,
            tr_penalty,
            tr_sector_label,
            tr_skip_reason,
            tr_stage_label_str,
            tr_stress_scenario,
        )

        self.assertIn("SOXX -15%", tr_stress_scenario({"scenario": "AI / 半导体板块回调 (SOXX -15%)"}, "en"))
        self.assertIn("半导体", tr_stress_scenario({"scenario": "AI / 半导体板块回调 (SOXX -15%)"}, "zh"))
        self.assertEqual("Cash Reserve", tr_sector_label("现金储备 (Cash)", "en"))
        self.assertIn("Tier 1", tr_sector_label("上游 · 云巨头与 AI 基础设施 (CapEx 金主)", "en"))
        self.assertIn("Earnings in 5d", tr_penalty("财报仅剩 5 天(-7分防赌财报)", "en"))
        self.assertEqual("Healthy Uptrend", tr_crash_badge_str("上升趋势/常规波段", "en"))
        self.assertEqual("Stage 2 · Advancing", tr_stage_label_str("Stage 2 · 上升主升浪", "en"))
        self.assertEqual("CLEAR (35d)", tr_earnings_flag_str("安全 (35天)", "en"))
        self.assertIn(
            "stretched vs",
            tr_pair_interpretation({"left": "STX", "right": "WDC", "zscore_60d": 2.1, "left_21d_pct": 10.0, "right_21d_pct": 2.0}, "en"),
        )
        self.assertIn("skipped", tr_skip_reason("本次跳过回测模块", "en").lower())

        sc_en, st_en, pm_en = get_prompts("en")
        sc_zh, st_zh, pm_zh = get_prompts("zh")
        self.assertIn("Chief Supply-Chain Strategist", sc_en)
        self.assertIn("产业链首席策略师", sc_zh)
        self.assertIn("Weekly Swing Execution Plan", pm_en)
        self.assertIn("本周波段交易执行计划", pm_zh)

    def test_bilingual_markdown_report_export(self) -> None:
        from src.engine.orchestrator import export_markdown_report, render_markdown_report

        sample_payload = {
            "generated_at": "2026-11-22 22:40:00",
            "elapsed_sec": 42.5,
            "device_profile": {"summary": "RTX 5070 Ti CUDA"},
            "regime": {
                "regime_score": 78.0,
                "regime_label": "强势进攻期 (Bull Expansion)",
                "suggested_max_exposure_pct": 90,
                "suggested_defensive_cash_pct": 10,
                "action_guide": "主线顺风",
                "metrics": {"breadth_above_50ma_pct": 75.0, "breadth_above_20ma_pct": 70.0, "VIX": 15.2},
                "signals": ["半导体相对大盘走强"],
            },
            "holdings_diagnosis": [
                {
                    "symbol": "GOOG",
                    "close": 200.0,
                    "weight_pct": 25.0,
                    "composite_score": 76.0,
                    "active_stop": 185.0,
                    "risk_contribution_pct": 20.0,
                    "earnings_flag": "安全 (35天)",
                    "action": "核心持有 · 逢回踩可加仓",
                }
            ],
            "stocks": {
                "GOOG": {
                    "symbol": "GOOG",
                    "sector": "cloud_ai",
                    "sector_label": "上游 · 云巨头",
                    "close": 200.0,
                    "composite_score": 76.0,
                    "analog": {"win_rate_21d_pct": 68.0},
                    "crash_analysis": {"status": "UPTREND_HEALTHY", "badge": "上升趋势/常规波段"},
                    "fundamentals": {"days_to_earnings": 35, "earnings_risk_flag": "安全 (35天)"},
                    "trade_ticket": {
                        "entry_low": 195.0,
                        "entry_high": 202.0,
                        "hard_stop": 185.0,
                        "target_1": 225.0,
                        "rr_target_1": 2.4,
                        "action_signal": "优先买入候选",
                    },
                }
            },
            "risk_mc": {
                "available": True,
                "total_nav_usd": 100000.0,
                "gross_exposure_pct": 85.0,
                "portfolio_vol_ann_pct": 24.5,
                "horizon_21d": {
                    "var_95_pct": -8.2,
                    "var_95_usd": -8200.0,
                    "cvar_95_pct": -11.4,
                    "cvar_95_usd": -11400.0,
                    "prob_loss_gt_10pct": 3.5,
                    "prob_gain_gt_10pct": 12.1,
                },
                "stress_tests": [
                    {
                        "scenario": "AI / 半导体板块回调 (SOXX -15%)",
                        "scenario_en": "AI / Semiconductor Pullback (SOXX -15%)",
                        "est_portfolio_impact_pct": -11.2,
                        "est_portfolio_impact_usd": -11200.0,
                        "largest_drag_symbol": "GOOG",
                    }
                ],
            },
            "llm": {"available": True, "model": "deepseek-chat", "supply_chain_report": {}, "pm_weekly_memo": ""},
        }

        md_en = render_markdown_report(sample_payload, lang="en")
        self.assertIn("US Equity Swing Research Report", md_en)
        self.assertIn("Active Holdings Diagnostics", md_en)
        self.assertIn("PRIORITY BUY", md_en)
        self.assertIn("CORE HOLD", md_en)
        self.assertIn("AI / Semiconductor Pullback (SOXX -15%)", md_en)
        self.assertNotIn("大盘与半导体温度计", md_en)

        md_zh = render_markdown_report(sample_payload, lang="zh")
        self.assertIn("美股波段投研周报", md_zh)
        self.assertIn("当前持仓诊断", md_zh)
        self.assertIn("优先买入候选", md_zh)

        with tempfile.TemporaryDirectory() as tmp:
            p_en = export_markdown_report(sample_payload, Path(tmp), lang="en")
            p_zh = export_markdown_report(sample_payload, Path(tmp), lang="zh")
            self.assertTrue(p_en.name.endswith("_en.md"))
            self.assertTrue(p_zh.name.endswith("_zh.md"))
            self.assertIn("US Equity Swing Research Report", p_en.read_text(encoding="utf-8"))
            self.assertIn("美股波段投研周报", p_zh.read_text(encoding="utf-8"))


class TestExtendedHoursQuotes(unittest.TestCase):
    """Verify pre-market / after-hours extended quote extraction from Yahoo info payloads."""

    def test_pre_post_regular_and_closed_selection(self) -> None:
        from src.data.market_data import extract_extended_quote_from_info

        # Pre-market session active
        q_pre = extract_extended_quote_from_info(
            "NVDA",
            {
                "marketState": "PRE",
                "regularMarketPrice": 140.0,
                "regularMarketPreviousClose": 138.0,
                "preMarketPrice": 143.5,
                "preMarketTime": 1700000200,
                "postMarketPrice": 140.2,
                "postMarketTime": 1700000100,
            },
        )
        self.assertEqual(q_pre["ext_session"], "PRE")
        self.assertAlmostEqual(q_pre["ext_price"], 143.5)
        self.assertAlmostEqual(q_pre["ext_change_pct"], 2.5)
        self.assertAlmostEqual(q_pre["latest_price"], 143.5)

        # After-hours session active
        q_post = extract_extended_quote_from_info(
            "GOOG",
            {
                "marketState": "POST",
                "regularMarketPrice": 200.0,
                "regularMarketPreviousClose": 198.0,
                "postMarketPrice": 196.0,
                "postMarketTime": 1700000500,
            },
        )
        self.assertEqual(q_post["ext_session"], "POST")
        self.assertAlmostEqual(q_post["ext_price"], 196.0)
        self.assertAlmostEqual(q_post["ext_change_pct"], -2.0)

        # Regular session live -> no stale extended session shown
        q_reg = extract_extended_quote_from_info(
            "STX",
            {
                "marketState": "REGULAR",
                "regularMarketPrice": 110.0,
                "regularMarketPreviousClose": 100.0,
                "preMarketPrice": 105.0,
            },
        )
        self.assertIsNone(q_reg["ext_session"])
        self.assertAlmostEqual(q_reg["regular_change_pct"], 10.0)
        self.assertAlmostEqual(q_reg["latest_price"], 110.0)

        # Closed market -> picks newer timestamp between pre and post
        q_closed = extract_extended_quote_from_info(
            "AMZN",
            {
                "marketState": "CLOSED",
                "regularMarketPrice": 200.0,
                "preMarketPrice": 202.0,
                "preMarketTime": 1700000900,
                "postMarketPrice": 199.0,
                "postMarketTime": 1700000400,
            },
        )
        self.assertEqual(q_closed["ext_session"], "PRE")
        self.assertAlmostEqual(q_closed["ext_price"], 202.0)

    def test_auto_total_account_equity(self) -> None:
        from src.engine.orchestrator import _compute_live_account_equity

        portfolio_cfg = {
            "account": {"cash_usd": 15000.0, "total_equity_usd": 99999.0},
            "holdings": [
                {"symbol": "GOOG", "shares": 50, "avg_cost": 180.0},
                {"symbol": "STX", "shares": 30, "avg_cost": 100.0},
            ],
        }
        live_quotes = {
            "GOOG": {"regular_price": 200.0},
            "STX": {"regular_price": 120.0},
        }
        # 50 * 200 + 30 * 120 + 15000 = 10000 + 3600 + 15000 = 28600
        self.assertAlmostEqual(_compute_live_account_equity(portfolio_cfg, {}, live_quotes), 28600.0)

    def test_auto_sector_classification(self) -> None:
        from src.data.market_data import classify_sector_from_info

        self.assertEqual(classify_sector_from_info({"sector": "Technology", "industry": "Semiconductors"}), "compute_optical")
        self.assertEqual(classify_sector_from_info({"sector": "Technology", "industry": "Computer Hardware & Data Storage"}), "memory_storage")
        self.assertEqual(classify_sector_from_info({"sector": "Technology", "industry": "Software - Infrastructure"}), "cloud_ai")
        self.assertEqual(classify_sector_from_info({"sector": "Consumer Defensive", "industry": "Beverages"}), "defensive")
        self.assertEqual(classify_sector_from_info({}), "other")


@unittest.skipUnless(HAS_NUMPY_PANDAS, "numpy/pandas not installed in current host Python")
class TestQuantNumericalEngines(unittest.TestCase):
    """Full synthetic-market numerical verification of indicators, GPU analog, MC risk & tickets."""

    def _make_synthetic_ohlcv(self, seed: int, n_bars: int = 400, drift: float = 0.0005) -> Any:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2024-01-02", periods=n_bars)
        rets = rng.normal(drift, 0.022, size=n_bars)
        close = 100.0 * np.cumprod(1.0 + rets)
        high = close * (1.0 + rng.uniform(0.002, 0.025, size=n_bars))
        low = close * (1.0 - rng.uniform(0.002, 0.025, size=n_bars))
        open_p = close * (1.0 + rng.normal(0.0, 0.008, size=n_bars))
        vol = rng.uniform(5e6, 25e6, size=n_bars)
        return pd.DataFrame(
            {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": vol},
            index=dates,
        )

    def test_end_to_end_quant_math(self) -> None:
        from src.quant.execution_plan import build_trade_ticket, compute_composite_swing_score
        from src.quant.gpu_analog import run_gpu_historical_analog_search
        from src.quant.gpu_backtest import run_gpu_grid_backtest
        from src.quant.gpu_risk_mc import run_gpu_portfolio_risk_mc
        from src.quant.indicators import analyze_stock_technicals
        from src.quant.regime import evaluate_market_regime

        ohlcv = {
            "SPY": self._make_synthetic_ohlcv(1, 500, 0.0004),
            "QQQ": self._make_synthetic_ohlcv(2, 500, 0.0006),
            "SOXX": self._make_synthetic_ohlcv(3, 500, 0.0007),
            "^VIX": self._make_synthetic_ohlcv(4, 500, 0.0) * 0.18,
            "GOOG": self._make_synthetic_ohlcv(10, 500, 0.0005),
            "STX": self._make_synthetic_ohlcv(11, 500, 0.0008),
            "MCD": self._make_synthetic_ohlcv(12, 500, 0.0002),
            "ORCL": self._make_synthetic_ohlcv(13, 500, -0.0002),
        }
        syms = ["GOOG", "STX", "MCD", "ORCL"]
        regime = evaluate_market_regime(ohlcv, syms)
        self.assertTrue(0.0 <= regime["regime_score"] <= 100.0)

        tech = analyze_stock_technicals("GOOG", ohlcv["GOOG"], ohlcv["QQQ"], ohlcv["QQQ"], 72.0)
        self.assertIn("crash_analysis", tech)

        analog = run_gpu_historical_analog_search(ohlcv, syms, window=30, top_k=20, device_pref="cpu")
        self.assertIn("GOOG", analog["per_symbol"])

        comp = compute_composite_swing_score(
            tech=tech,
            fundamentals={"fundamental_score": 72.0, "days_to_earnings": 35},
            analog=analog["per_symbol"]["GOOG"],
            ml_alpha=None,
            sec_info=None,
            regime_score=regime["regime_score"],
        )
        ticket = build_trade_ticket(
            symbol="GOOG",
            tech=tech,
            composite=comp,
            fundamentals={"days_to_earnings": 35},
            account_cfg={"total_equity_usd": 100000.0, "max_risk_per_trade_pct": 1.25},
            regime=regime,
            sector_key="cloud_ai",
        )
        self.assertLess(ticket["hard_stop"], ticket["ref_entry"])
        self.assertGreater(ticket["target_1"], ticket["ref_entry"])

        mc = run_gpu_portfolio_risk_mc(
            ohlcv_map=ohlcv,
            portfolio_cfg={
                "account": {"cash_usd": 15000.0},
                "holdings": [
                    {"symbol": "GOOG", "shares": 50},
                    {"symbol": "STX", "shares": 30},
                    {"symbol": "MCD", "shares": 45},
                ],
            },
            universe_cfg={"universe": {}, "sectors": {}},
            n_sims=2000,
            device_pref="cpu",
        )
        self.assertTrue(mc["available"])
        self.assertLess(mc["horizon_21d"]["cvar_95_pct"], mc["horizon_21d"]["var_95_pct"])

        bt = run_gpu_grid_backtest(ohlcv, syms, lookback_years=2, device_pref="cpu")
        self.assertTrue(bt["available"])

        # Verify 4-Stage GPU Grid Trading Lab (CPU fallback on host)
        from src.quant.gpu_grid_lab import (
            build_live_grid_order_ladder,
            run_complete_grid_lab_for_symbol,
            screen_watchlist_for_grid,
        )

        s1 = screen_watchlist_for_grid(ohlcv, syms, device_pref="cpu")
        self.assertEqual(len(s1["candidates"]), len(syms))
        for cand in s1["candidates"]:
            self.assertTrue(0.0 <= cand["grid_score"] <= 100.0)
            self.assertIn(cand["verdict_code"], ("SUITABLE", "CONDITIONAL", "NOT_SUITABLE"))

        deep = run_complete_grid_lab_for_symbol(
            symbol="GOOG",
            ohlcv_map=ohlcv,
            n_mc_paths=200,
            steps_per_day=2,
            high_density_gpu=False,
            device_pref="cpu",
        )
        self.assertTrue(deep["available"])
        self.assertEqual(len(deep["stage2_hypercube"]["top_5_plateaus"]), 5)
        self.assertTrue(deep["stage3_montecarlo"]["available"])
        self.assertTrue(deep["stage4_ml_switch"]["available"])

        ladder = build_live_grid_order_ladder(
            last_price=float(ohlcv["GOOG"]["Close"].iloc[-1]),
            live_atr=3.5,
            config=deep["champion_config"],
            grid_capital_usd=20000.0,
            current_shares_held=20.0,
            ml_state_code=deep["stage4_ml_switch"]["state_code"],
        )
        self.assertGreater(len(ladder["orders"]), 0)


if __name__ == "__main__":
    unittest.main()
