#!/usr/bin/env bash
# ==============================================================================
# Windows WSL2 (Ubuntu) + NVIDIA RTX 5070 Ti (Blackwell sm_120) 一键环境配置脚本
# 用法: bash scripts/setup_wsl_5070ti.sh
# ==============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "================================================================"
echo "⚡ Front-Running · RTX 5070 Ti (WSL2) 环境初始化"
echo "================================================================"

# 1. 检查 WSL2 NVIDIA GPU 透传
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✅ 检测到 NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "⚠️  未在 WSL2 中检测到 nvidia-smi！"
    echo "   请确保在 Windows 宿主机安装了最新版 NVIDIA 驱动 (572+ 支持 RTX 50 系列 Blackwell)，"
    echo "   切勿在 WSL2 Linux 内手动安装 NVIDIA Linux 驱动！"
fi

# 2. 选择 Python 解释器 (推荐 Ubuntu 24.04 自带 Python 3.12)
PY_BIN="python3"
if command -v python3.12 >/dev/null 2>&1; then
    PY_BIN="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
    PY_BIN="python3.11"
fi
echo "🐍 使用 Python: $($PY_BIN --version)"

# 3. 创建虚拟环境
if [ ! -d ".venv" ]; then
    echo "📦 创建虚拟环境 .venv ..."
    $PY_BIN -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# 4. 安装适配 RTX 5070 Ti (Blackwell sm_120) 的 PyTorch CUDA 12.8
echo "🔥 安装 PyTorch CUDA 12.8 (Blackwell RTX 5070 Ti 原生支持)..."
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128

# 5. 安装项目核心依赖 (XGBoost GPU / yfinance / Streamlit / Plotly / OpenAI SDK)
echo "📚 安装量化、数据、可视化与本地 LLM 依赖..."
pip install -r requirements.txt

# 6. 验证 RTX 5070 Ti CUDA 算子实际可执行 (排除 sm_120 不兼容问题)
echo "🧪 验证 RTX 5070 Ti CUDA 矩阵计算..."
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
    print("✅ RTX 5070 Ti CUDA Tensor 运算验证通过! checksum =", float(c[0, 0].item()))
else:
    print("⚠️  当前未启用 CUDA，将以 CPU 模式运行。")
PY

# 7. 初始化 .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 已生成默认 .env 配置文件"
fi

# 8. 检查 / 引导 Ollama 本地推理引擎 (16GB VRAM 黄金模型: qwen3:14b)
if command -v ollama >/dev/null 2>&1; then
    echo "✅ 检测到本地已安装 Ollama: $(ollama --version)"
else
    echo "🤖 正在安装 Ollama (用于 5070 Ti 本地免费跑 14B/32B 金融投研大模型)..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo ""
echo "================================================================"
echo "🎉 环境安装完成！下一步只需两步即可起飞："
echo "  1) 拉取 5070 Ti 16GB 显存黄金推荐模型 (首次约 9GB):"
echo "     ollama serve &"
echo "     ollama pull qwen3:14b"
echo "     (也可选备选深度思考模型: ollama pull deepseek-r1:14b 或 qwen2.5:14b)"
echo ""
echo "  2) 启动本地暗黑投研驾驶舱 (在 Windows 浏览器打开 http://localhost:8501):"
echo "     bash scripts/run_dashboard.sh"
echo ""
echo "  或命令行一键跑完生成离线周报后直接关机:"
echo "     source .venv/bin/activate && python scripts/run_pipeline_cli.py"
echo "================================================================"
