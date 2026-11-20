# ⚡ Front-Running · AI & 存储产业链美股波段投研驾驶舱 (RTX 5070 Ti 专属版)

专为 **持仓周期 2 周 ~ 1~2 个月、只做美股现货 (Buy/Sell)、电脑非 24 小时开机** 的个人交易者打造的专业本地量化 + AI 投研工作站，深度针对 **Windows + WSL2 + NVIDIA RTX 5070 Ti (16GB GDDR7 / Blackwell `sm_120`)** 进行算力与显存协同优化。

---

## 🎯 为你的核心持仓与股票池量身定制

已预置你的完整自选池、6 只持仓与 5 只产业链风向标（可在 `config/universe.yaml` 和网页中随时增删）：

* **💼 当前持仓**：`GOOG` · `STX` · `MCD` · `AMZN` · `ORCL` · `SKHY`
* **👀 核心自选**：`AMD` · `ANT` · `MU` · `WDC` · `SNDK` · `NVDA` · `AVGO` · `INTC` · `COHR` · `SPCX`
* **📡 产业链风向标 & 宏观锚**：`TSM` · `MSFT` · `META` · `SPY` · `QQQ` · `SOXX` · `^VIX` · `^TNX`

---

## 🔥 五大 GPU 算力与专业交易核心模块

| 模块 | 底层引擎 (RTX 5070 Ti CUDA 加速) | 解决的真实交易痛点 |
| :--- | :--- | :--- |
| **1. 宏观与半导体温度计** | `SPY/QQQ/SOXX/VIX/US10Y` + 股票池内部宽度 | 定死每周**总仓位上限**与 **现金 / `MCD` 防御底仓比例** |
| **2. 错杀抄底 vs 接飞刀排雷** | Weinstein 周期 + 机构吸筹/派发日 + CMF + **5070 Ti 读 SEC 8-K/新闻归因** | 防止在主跌浪半山腰盲目补仓，识别真错杀黄金坑 |
| **3. GPU 10 年历史相似形态检索** | **PyTorch CUDA Tensor GEMM** 并行扫描全池 10 年量价/RS 切片 | 给出未来 21 日/42 日真实历史胜率、盈亏比 PF 与中位最大盘中回撤 MAE |
| **4. GPU Walk-Forward 机器学习** | **XGBoost CUDA + PyTorch ResMLP**（严格防泄露 Purged Expanding CV） | 预测未来 1 个月跑赢纳指概率，公开样本外 AUC 与超额差防忽悠 |
| **5. GPU 10 万路径肥尾风险模拟** | **PyTorch CUDA 100,000 路径 Student-t 相关蒙特卡洛** + 4 大极端压力测试 | 精准量化 VaR/CVaR、单股风险贡献 MCR 与 AI/存储集中度 |
| **6. GPU 288 组网格策略回测** | **PyTorch CUDA 向量化回测**（含 10bps/边滑点佣金 + 移动止盈） | 6 年历史真数据对比 `QQQ/SOXX`，输出夏普参数稳健性热力图 |
| **7. 5070 Ti 本地多角色 AI 投研团** | **Ollama 本地 14B/32B**（产业链策略师 → 逐股多空排雷 → 基金经理周报） | 零 API 费精读 SEC 公告与新闻，自动输出可执行中文《本周交易计划》 |
| **8. 交易执行单 & 复盘日志** | 结构+ATR 止损、Chandelier 移动止盈、相关性折减仓位计算 + SQLite 日志 | 每只票直接给出入场区、硬止损、目标 1/2 与建议股数 |

> 💡 **16GB GDDR7 显存防爆设计**：系统采用两段式算力接力——先运行 PyTorch CUDA 全部量化/ML/蒙特卡洛/回测 → 自动调用 `release_gpu_memory()` 清空显存 → 再交棒给本地 Ollama 大模型推理，绝不发生 OOM。

---

## 🚀 Windows + WSL2 (Ubuntu) + RTX 5070 Ti 快速部署指南

### 前置条件（仅需确认一次）
1. **Windows 宿主机**安装最新 **NVIDIA 驱动（572+，原生支持 RTX 5070 Ti Blackwell）**；
   > ⚠️ **切勿**在 WSL2 Ubuntu 里再安装 Linux 版 NVIDIA 显卡驱动，Windows 驱动会自动把 CUDA 透传进 WSL2。
2. 在 Windows 打开 **WSL2 (Ubuntu 22.04 / 24.04)**，进入本仓库目录：

```bash
cd /path/to/Front-Running
```

### 第 1 步：一键安装 PyTorch CUDA 12.8 (Blackwell `sm_120`) + 全部依赖
```bash
bash scripts/setup_wsl_5070ti.sh
```
> 为什么必须用 `cu128`？因为 RTX 5070 Ti 是最新 Blackwell (`sm_120`) 架构，老版本 PyTorch (`cu121/cu124`) 会报错无法执行内核，脚本已自动为你配置好官方 `cu128` 源并做完矩阵算子实测。

### 第 2 步：拉取 5070 Ti 16GB 显存最佳中文投研模型（首次约 9GB）
```bash
ollama serve &
ollama pull qwen3:14b
```
*(其他 16GB 显存可选强模型：`qwen2.5:14b`、`deepseek-r1:14b`、`qwen2.5:32b-instruct-q4_K_M`，安装后可在网页左侧下拉框随时切换。)*

### 第 3 步：启动暗黑投研终端（随用随开，跑完关机）
```bash
bash scripts/run_dashboard.sh
```
然后在 **Windows 浏览器** 打开 👉 `http://localhost:8501`

1. 先到 **「💼 持仓诊断 & 风险模拟」** 展开顶部编辑框，填入你真实的总资金、现金、以及 `GOOG / STX / MCD / AMZN / ORCL / SKHY` 的实际持股数与成本价并保存；
2. 点击左侧 **「⚡ 启动全量分析」**，等待约 3~6 分钟；
3. 浏览 7 大面板或点击右上角 **「📄 下载离线中文周报 (.md)」**，全部结果自动存入本地 `data/trading_desk.db`，随时关机，下次开机秒看历史快照与周度变化对比！

### （可选）极客模式：一条命令跑完生成离线报告直接关机
```bash
source .venv/bin/activate
python scripts/run_pipeline_cli.py
```
报告会自动输出在 `reports/weekly_trading_report_YYYYMMDD_HHMM.md`。

### （可选）在 Mac 上打开网页，远程调用家里 5070 Ti 台式机的 Ollama
1. 在 Windows/WSL2 上启动 Ollama 时监听局域网：`OLLAMA_HOST=0.0.0.0:11434 ollama serve`
2. 在 Mac 本项目的 `.env` 中设置 `LLM_BASE_URL=http://<你的台式机局域网IP>:11434/v1`

---

## 🗂️ 项目目录结构

```text
Front-Running/
├── app.py                         # Streamlit 暗黑专业投研终端 (7大核心标签页)
├── config/
│   ├── universe.yaml              # 16只自选 + 5只风向标 + 供应链图谱 + 同赛道配对
│   └── portfolio.yaml             # 账户风控参数与 6 只当前持仓 (可在UI直接改)
├── scripts/
│   ├── setup_wsl_5070ti.sh        # WSL2 + RTX 5070 Ti (CUDA 12.8 Blackwell) 一键安装
│   ├── run_dashboard.sh           # 一键启动工作台
│   └── run_pipeline_cli.py        # 命令行一键跑全量并生成离线报告
├── src/
│   ├── data/                      # 行情缓存、基本面/财报日历、SEC EDGAR 官方公告、多源新闻
│   ├── quant/                     # 大盘温度计、量价/排雷、GPU历史相似检索、GPU ML、10万路径MC、GPU回测、仓位执行单
│   ├── ai/                        # 本地 5070 Ti LLM 客户端 + 产业链/多空排雷/基金经理 Prompt 与编排
│   ├── engine/                    # 8阶段总控引擎 + 显存移交 + 离线 Markdown 报告导出
│   ├── storage/                   # SQLite 快照归档、缓存、交易复盘日志
│   └── ui/                        # Plotly 专业暗黑金融图表库
└── tests/
    └── test_core.py               # 语法校验与核心逻辑自动化测试
```

---

*免责声明：本系统为本地开源量化分析与投研效率工具，所有评分、历史统计与本地 AI 分析均不构成任何具体证券买卖投资建议。股市有风险，实盘请严格遵守个人风险承受能力与止损纪律。*
