# EoA

一个面向组合优化与启发式搜索的 **LLM + Evolution** 实验框架。  
项目通过“大语言模型生成候选算法代码 → 任务评估器真实打分 → 演化选择保留优质个体”的闭环，不断改进启发式函数或求解策略。

## 1. 项目概览

本项目的核心思路：

1. 为某个任务提供一个 `seed` 初始算法；
2. 使用 LLM 生成多样化初始种群；
3. 每一代基于不同演化策略（`modification / exploration / simplification`）生成新候选；
4. 调用任务绑定的评估器真实执行并打分；
5. 用精英保留 + 锦标赛选择组成下一代；
6. 维护历史最优 `archive`。

入口文件是 `main.py`，任务通过 `tasks/registry.py` 注册和切换。

---

## 2. 目录结构

```text
EoA/
├── main.py
├── README.md
├── .env.example
├── .gitignore
├── config/
│   ├── llm.example.json
│   ├── llm.deepseek.json
│   └── llm.vllm.qwen3-omni.json
├── develop_population_manager_module/
│   ├── initialize_population_module.py
│   ├── select_parents_and_offspring_module.py
│   └── archive_best_individuals_module.py
├── implement_evolutionary_operators_module/
│   ├── design_offspring_generation_controller.py
│   └── develop_prompt_strategy_generator.py
├── implement_llm_interaction_module/
│   ├── env_loader.py
│   ├── llm_config.py
│   ├── develop_api_wrapper.py
│   └── implement_response_parser.py
├── integrate_fitness_evaluator_module/
├── run_output_recorder.py
├── tasks/
│   ├── base.py
│   ├── registry.py
│   ├── task_support/
│   ├── min_max_layout_16/
│   ├── tsp_aco/
│   ├── tsp_gls/
│   ├── tsp_constructive/
│   ├── tsp_pomo/
│   ├── tsp_lehd/
│   ├── cvrp_aco/
│   ├── cvrp_pomo/
│   ├── cvrp_lehd/
│   ├── op_aco/
│   ├── mkp_aco/
│   ├── bpp_offline_aco/
│   ├── bpp_online/
│   └── dpp_ga/
├── task_assets/
│   ├── problems/
│   ├── prompts/
│   └── utils/
└── output/
```

### 关键模块说明

- `main.py`：程序入口，负责参数解析、任务加载、主演化循环。
- `tasks/`：任务插件层；每个任务定义种子代码、提示词、评估器和任务配置。
- `task_assets/problems/`：任务相关的评估脚本、实例生成器、参考实现、模型脚本等资源。
- `implement_llm_interaction_module/`：LLM 配置、请求发送、响应解析。
- `develop_population_manager_module/`：种群初始化、父代/子代选择、archive 管理。
- `implement_evolutionary_operators_module/`：演化策略调度、prompt 构造、offspring 集成。
- `output/`：运行日志、每代快照、最终 archive 输出目录。

---

## 3. 环境准备

建议环境：

- Python 3.10+
- 推荐使用虚拟环境

### 安装基础依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy scipy requests
```

> 说明：  
> 某些任务（如 `tsp_pomo`、`cvrp_pomo`、`tsp_lehd`、`cvrp_lehd`）通常还需要 `torch` 及对应模型权重。  
> 如果你只想先验证主框架，建议先跑 `min_max_layout_16`。

---

## 4. LLM 配置方式

项目现在统一从项目根目录 `.env` 读取 LLM 配置。

### 4.1 创建本地配置

```bash
cp .env.example .env
```

编辑 `.env`，例如：

```env
LLM_PRESET=deepseek
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=deepseek-chat
LLM_TIMEOUT=120
```

### 4.2 配置优先级

优先级从高到低：

1. CLI 参数（如 `--llm-api-key`）
2. `.env` / 进程环境变量
3. `--llm-config` 指定的 JSON 配置
4. 内置 preset 默认值

通常只需要维护 `.env` 即可。

---

## 5. 运行方式

### 5.1 查看/确认支持的任务

当前注册任务包括：

- `min_max_layout_16`
- `tsp_constructive`
- `tsp_aco`
- `tsp_gls`
- `cvrp_aco`
- `op_aco`
- `mkp_aco`
- `bpp_offline_aco`
- `bpp_online`
- `dpp_ga`
- `tsp_pomo`
- `tsp_lehd`
- `cvrp_pomo`
- `cvrp_lehd`

### 5.2 推荐的最小可运行示例

```bash
python main.py --task min_max_layout_16 --llm-concurrency 1
```

如果网络稳定，也可以提高并发：

```bash
python main.py --task min_max_layout_16 --llm-concurrency 4
```

### 5.3 其他示例

```bash
python main.py --task tsp_aco --llm-concurrency 2
python main.py --task bpp_online --llm-concurrency 2
python main.py --task dpp_ga --llm-concurrency 2
```

### 5.4 常用参数

- `--task`：任务 ID
- `--llm-preset`：快速指定 LLM 预设，如 `ollama / openai / deepseek / vllm_qwen3`
- `--llm-base-url`：覆盖 `LLM_BASE_URL`
- `--llm-api-key`：覆盖 `LLM_API_KEY`
- `--llm-model`：覆盖 `LLM_MODEL`
- `--llm-timeout`：覆盖请求超时
- `--llm-concurrency`：并发请求数
- `--no-run-output`：不写入 `output/`

### 5.5 输出结果位置

默认输出目录：

```text
output/<task_id>/<timestamp>/
```

其中包含：

- `terminal.log`：终端日志
- `run_meta.json`：本次运行元信息
- `evolution/generation_*.json`：每代快照
- `final_archive.json`：最终 archive

---

## 6. 支持的任务 / 数据资源

这里的“数据集”更准确地说是 **任务资源、评测脚本、实例生成器与模型权重依赖**。

### 6.1 内置可直接运行或资源基本齐全的任务

| 任务 ID | 类型 | 资源位置 | 备注 |
|---|---|---|---|
| `min_max_layout_16` | 几何点布局 | `tasks/min_max_layout_16/` | 最容易跑通，适合作为 smoke test |
| `tsp_constructive` | TSP 构造式启发式 | `task_assets/problems/tsp_constructive/` | 含 `eval.py`、`gen_inst.py`、`gpt.py` |
| `tsp_aco` | TSP + ACO | `task_assets/problems/tsp_aco/` | 含 ACO 与评测脚本 |
| `tsp_gls` | TSP + GLS | `task_assets/problems/tsp_gls/` | 含 `gls.py` |
| `cvrp_aco` | CVRP + ACO | `task_assets/problems/cvrp_aco/` | 含 ACO 与评测脚本 |
| `op_aco` | Orienteering Problem + ACO | `task_assets/problems/op_aco/` | 含 ACO 与评测脚本 |
| `mkp_aco` | MKP + ACO | `task_assets/problems/mkp_aco/` | 含 ACO 与评测脚本 |
| `bpp_offline_aco` | Offline BPP + ACO | `task_assets/problems/bpp_offline_aco/` | 含 ACO 与评测脚本 |
| `bpp_online` | Online BPP | `task_assets/problems/bpp_online/` | 含 `eval.py`、`gen_inst.py` |
| `dpp_ga` | Decap Placement GA | `task_assets/problems/dpp_ga/` | 含 `eval.py`、`reward_functions.py` 等 |

### 6.2 需要额外下载权重/数据的任务

| 任务 ID | 类型 | 说明 |
|---|---|---|
| `tsp_pomo` | TSP + POMO | 需要下载官方 checkpoint |
| `cvrp_pomo` | CVRP + POMO | 需要下载官方 checkpoint |
| `tsp_lehd` | TSP + LEHD | 需要下载 LEHD checkpoint / data |
| `cvrp_lehd` | CVRP + LEHD | 需要下载 LEHD checkpoint / data |

相关说明见 `task_assets/problems/readme.md`：

- POMO：需要从官方仓库下载 checkpoint 并放入对应目录
- LEHD：需要下载 checkpoint 和数据并放入对应目录

---

## 7. 任务扩展方式

如果你想添加一个新任务，通常需要：

1. 在 `tasks/<your_task>/` 下实现任务类；
2. 继承 `EvolutionTask`；
3. 提供：
   - `seed_code`
   - `seed_thought`
   - `evaluate_raw()`
   - `prompt_strategies`
   - `initial_population_system_prompt`
4. 如有需要，在 `task_assets/problems/<your_task>/` 下放置评测脚本与任务资源；
5. 在 `tasks/registry.py` 中注册该任务。

---

## 8. 常见问题

### Q1. 为什么请求 DeepSeek 时会报 SSL / EOF / timeout？

通常是网络层问题，而不是任务逻辑错误。建议：

- 降低 `--llm-concurrency`
- 检查代理 / VPN / 网络稳定性
- 升级 `requests / urllib3 / certifi`

### Q2. 现在还需要传 `--llm-config` 吗？

通常不需要。  
推荐直接使用 `.env`。

### Q3. 运行前最推荐先跑哪个任务？

推荐先跑：

```bash
python main.py --task min_max_layout_16 --llm-concurrency 1
```

---

## 9. 参考说明

项目内 `task_assets/problems/readme.md` 中提到：

- ACO 实现参考 DeepACO
- POMO/LEHD 任务需要补充官方 checkpoint 或数据资源

如果你将本项目公开发布到 GitHub，建议同时提交：

- `README.md`
- `.env.example`
- `.gitignore`

并确保不要提交：

- `.env`
- `output/`
- 任意真实 API key

