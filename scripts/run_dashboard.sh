#!/usr/bin/env bash
# ==============================================================================
# 一键启动 Front-Running 本地投研工作台 (Windows WSL2 + RTX 5070 Ti)
# 启动后直接在 Windows 浏览器访问: http://localhost:8501
# ==============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 若安装了 Ollama 但尚未在后台启动，自动拉起（随用随开，不占用常驻开机自启）
if command -v ollama >/dev/null 2>&1; then
    if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo "🤖 自动唤醒本地 Ollama (RTX 5070 Ti 推理引擎)..."
        nohup ollama serve > /tmp/ollama_trading.log 2>&1 &
        sleep 2
    fi
fi

echo "🚀 正在启动 Front-Running 投研终端..."
echo "👉 请在 Windows 浏览器打开: http://localhost:8501"
exec streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --theme.base dark
