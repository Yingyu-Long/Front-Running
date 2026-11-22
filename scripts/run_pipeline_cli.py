#!/usr/bin/env python3
"""One-shot CLI runner: execute the full 8-stage 5070 Ti pipeline & export offline Markdown report.

Perfect for non-24/7 PC usage:
  1. Boot PC -> run `python scripts/run_pipeline_cli.py`
  2. RTX 5070 Ti runs full quant + GPU ML + 100k Monte Carlo + GPU Backtest + Local LLM research
  3. Read `reports/weekly_trading_report_YYYYMMDD_HHMM.md` (or open Streamlit anytime) & shut down PC!
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.orchestrator import run_full_analysis_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Front-Running - RTX 5070 Ti US equity swing research and offline report generator"
    )
    parser.add_argument("--force-refresh", action="store_true", help="Bypass local cache and fetch fresh market, SEC, and news data")
    parser.add_argument("--no-ml", action="store_true", help="Skip GPU Walk-Forward machine learning module")
    parser.add_argument("--no-backtest", action="store_true", help="Skip GPU 288-combo grid backtest")
    parser.add_argument("--no-llm", action="store_true", help="Run GPU quant only without calling the local Ollama LLM")
    parser.add_argument("--model", type=str, default=None, help="Temporarily override the local LLM model name (e.g. qwen3:14b)")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"], help="AI report language (en default / zh Chinese)")
    args = parser.parse_args()

    def _progress(pct: float, msg: str) -> None:
        print(f"[{pct * 100:5.1f}%] {msg}", flush=True)

    payload = run_full_analysis_pipeline(
        force_refresh_data=args.force_refresh,
        run_ml_alpha=not args.no_ml,
        run_backtest=not args.no_backtest,
        run_llm=not args.no_llm,
        llm_model_override=args.model,
        lang=args.lang,
        progress_cb=_progress,
    )

    print("\n" + "=" * 72)
    print(f"[DONE] Full analysis archived (snapshot #{payload.get('snapshot_id')}, elapsed {payload.get('elapsed_sec')}s)")
    print(f"Market & Semi Regime: {payload['regime']['regime_score']}/100 - {payload['regime']['regime_label']}")
    print(f"Offline Markdown report exported: {payload.get('report_path')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
