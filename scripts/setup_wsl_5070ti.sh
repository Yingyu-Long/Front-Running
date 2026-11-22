#!/usr/bin/env bash
# ==============================================================================
# Windows WSL2 (Ubuntu) + NVIDIA RTX 5070 Ti (Blackwell sm_120) Setup Script
# Usage: bash scripts/setup_wsl_5070ti.sh
# ==============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "================================================================"
echo "Front-Running - RTX 5070 Ti (WSL2) Environment Setup"
echo "================================================================"

# 1. Check WSL2 NVIDIA GPU passthrough
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[OK] Detected NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "[WARN] nvidia-smi not found in WSL2."
    echo "       Make sure the latest NVIDIA driver (572+ for RTX 50 Blackwell) is installed on Windows host."
    echo "       Do NOT install a Linux NVIDIA driver inside WSL2."
fi

# 2. Select Python interpreter (Ubuntu 24.04 ships with Python 3.12)
PY_BIN="python3"
if command -v python3.12 >/dev/null 2>&1; then
    PY_BIN="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
    PY_BIN="python3.11"
fi
echo "Using Python: $($PY_BIN --version)"

# 3. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment .venv ..."
    $PY_BIN -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# 4. Install PyTorch CUDA 12.8 for RTX 5070 Ti (Blackwell sm_120)
echo "Installing PyTorch CUDA 12.8 (native Blackwell RTX 5070 Ti support)..."
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128

# 5. Install core dependencies (XGBoost GPU / yfinance / Streamlit / Plotly / OpenAI SDK)
echo "Installing quant, data, visualization, and local LLM dependencies..."
pip install -r requirements.txt

# 6. Verify RTX 5070 Ti CUDA kernel execution (guards against sm_120 incompatibility)
echo "Verifying RTX 5070 Ti CUDA matrix multiplication..."
python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda, "| Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / (1024**3):.1f} GB | CC: {props.major}.{props.minor}")
    a = torch.randn((1024, 1024), device="cuda", dtype=torch.float32)
    b = torch.randn((1024, 1024), device="cuda", dtype=torch.float32)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print("[OK] RTX 5070 Ti CUDA tensor verification passed! checksum =", float(c[0, 0].item()))
else:
    print("[WARN] CUDA is not enabled; running in CPU mode.")
PY

# 7. Initialize .env configuration
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created default .env configuration file."
fi

# 8. Check / install local Ollama inference engine (recommended for 16GB VRAM: qwen3:14b)
if command -v ollama >/dev/null 2>&1; then
    echo "[OK] Local Ollama already installed: $(ollama --version)"
else
    echo "Installing Ollama (for running local 14B/32B financial research LLMs on RTX 5070 Ti)..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo ""
echo "================================================================"
echo "Setup complete. Next steps:"
echo "  1) Pull the recommended 16GB VRAM model (~9GB on first download):"
echo "     ollama serve &"
echo "     ollama pull qwen3:14b"
echo "     (Alternatives: ollama pull deepseek-r1:14b or qwen2.5:14b)"
echo ""
echo "  2) Launch the local dark-mode research dashboard (open http://localhost:8501 in Windows browser):"
echo "     bash scripts/run_dashboard.sh"
echo ""
echo "  Or run the one-shot CLI pipeline to export an offline report before shutting down:"
echo "     source .venv/bin/activate && python scripts/run_pipeline_cli.py"
echo "================================================================"
