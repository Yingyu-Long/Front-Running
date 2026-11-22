# Front-Running

GPU-accelerated US equity swing research and grid trading workstation built with Streamlit, PyTorch CUDA, XGBoost/LightGBM, and an OpenAI-compatible LLM research pipeline (Cloud API or local Ollama).

## Prerequisites

- **Python**: 3.11 or 3.12 (`python3-venv` installed on Ubuntu/WSL2)
- **GPU (Optional, recommended)**: NVIDIA GPU with CUDA support
  - On Windows WSL2, install the latest NVIDIA Game Ready / Studio driver on the Windows host (572+ for RTX 50-series Blackwell). Do **not** install a Linux NVIDIA driver inside WSL2.
- **LLM Backend (Optional)**: Either a cloud OpenAI-compatible API key (DeepSeek, Qwen/DashScope, OpenAI, OpenRouter) or a local Ollama instance. Pure GPU quantitative mode works without an LLM.

## 1. Clone and Install

### Option A: Windows WSL2 (Ubuntu) + NVIDIA RTX 50-Series (One-Shot Script)

```bash
git clone https://github.com/Yingyu-Long/Front-Running.git
cd Front-Running
bash scripts/setup_wsl_5070ti.sh
```

> Tip: If you plan to use a Cloud API or already have Ollama installed on Windows, you can press `Ctrl+C` if the script prompts for `sudo` to install Linux Ollama at the final step.

### Option B: Generic Linux / macOS / CPU-Only Setup

```bash
git clone https://github.com/Yingyu-Long/Front-Running.git
cd Front-Running
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# Install PyTorch (pick the wheel matching your hardware; CPU fallback works everywhere)
pip install torch
pip install -r requirements.txt

cp .env.example .env
```

## 2. Configure `.env`

Edit `.env` (ignored by Git) to set your LLM backend and SEC EDGAR identifier:

### Using a Cloud API (Fastest Setup)

```env
# Example: DeepSeek
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-your-api-key
LLM_MODEL=deepseek-chat
```

Any OpenAI-compatible endpoint works (e.g. OpenAI `https://api.openai.com/v1`, Alibaba DashScope `https://dashscope.aliyuncs.com/compatible-mode/v1`, or OpenRouter `https://openrouter.ai/api/v1`).

### Using Local Ollama

```env
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen3:14b
```

Pull a model before scanning:

```bash
ollama serve &
ollama pull qwen3:14b
```

> If running the app in WSL2 while Ollama runs on the Windows host, either add `networkingMode=mirrored` under `[wsl2]` in `C:\Users\<user>\.wslconfig` (then `wsl --shutdown`), or expose Ollama via `OLLAMA_HOST=0.0.0.0:11434` and point `LLM_BASE_URL` to your Windows host IP.

### SEC EDGAR Header & Portfolio

- Update `SEC_USER_AGENT` in `.env` with a valid email-format identifier (required by SEC EDGAR, free, no API key needed).
- Optionally edit `config/portfolio.yaml` and `config/universe.yaml` (both can also be managed directly in the web UI).

## 3. Run the Application

### Launch the Web Dashboard

```bash
bash scripts/run_dashboard.sh
```

Open `http://localhost:8501` in your browser, verify `CUDA Active` and `LLM Online` in the left sidebar, and click **RUN FULL SCAN**.

### One-Shot CLI Report Export

```bash
source .venv/bin/activate
python scripts/run_pipeline_cli.py --lang en
```

- Use `--lang zh` for a Chinese report, or `--no-llm` for quant-only mode.
- Reports are saved to `reports/weekly_trading_report_YYYYMMDD_HHMM_<lang>.md`.

## 4. Verify Installation

```bash
source .venv/bin/activate
pytest -q
```
