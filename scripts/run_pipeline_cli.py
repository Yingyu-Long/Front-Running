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
        description="Front-Running · RTX 5070 Ti 一键全量美股波段投研与离线周报生成器"
    )
    parser.add_argument("--force-refresh", action="store_true", help="忽略本地缓存，强制拉取最新行情与SEC/新闻")
    parser.add_argument("--no-ml", action="store_true", help="跳过 GPU Walk-Forward 机器学习")
    parser.add_argument("--no-backtest", action="store_true", help="跳过 GPU 288组网格回测")
    parser.add_argument("--no-llm", action="store_true", help="仅跑纯 GPU 量化，不调用本地 Ollama 大模型")
    parser.add_argument("--model", type=str, default=None, help="临时覆盖本地 LLM 模型名称 (例如 qwen3:14b)")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"], help="AI 报告语言 (en 默认英文 / zh 中文)")
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
    print(f"✅ 全量分析归档完成 (快照 #{payload.get('snapshot_id')}, 总耗时 {payload.get('elapsed_sec')}s)")
    print(f"🌡️ 大盘与半导体温度: {payload['regime']['regime_score']}/100 — {payload['regime']['regime_label']}")
    print(f"📄 离线中文报告已导出: {payload.get('report_path')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
