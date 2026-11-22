#!/usr/bin/env bash
# ==============================================================================
# Launch Front-Running Local Research Workstation (Windows WSL2 + RTX 5070 Ti)
# Open http://localhost:8501 in your Windows browser once started.
# ==============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Auto-start local Ollama on demand if installed in WSL2 and not already running
if command -v ollama >/dev/null 2>&1; then
    if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo "Starting local Ollama inference engine (RTX 5070 Ti)..."
        nohup ollama serve > /tmp/ollama_trading.log 2>&1 &
        sleep 2
    fi
fi

echo "Starting Front-Running research terminal..."
echo "Open in your Windows browser: http://localhost:8501"
exec streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --theme.base dark
