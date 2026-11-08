# Agent Trace Theory: experiment code

Code and fixed task inputs for the experiments in *Agent Trace Theory: Information Coverage for Evaluating Stateful Tool-Using LLM Agents*. The repository contains no execution traces, model responses, metric tables, figures, or experimental result files.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/toolsim/` | Persistent state, tool execution, backends, agent policies, and evaluators |
| `scripts/run_experiment{1..4}_llm_<provider>.py` | One entry point per experiment and provider |
| `scripts/run_paper_experiment.ps1` | Common Windows launcher for the four experiments |
| `scripts/audit_same_trace_projections.py` | Offline audit of complete local traces |
| `data/benchmark/` | Fixed task definitions used by the paper |
| `tests/` | Unit and input contract checks |
| `outputs/` | Local results and logs; ignored by Git |

## Experimental protocol

| Experiment | Fixed tasks | Conditions | Entry point |
| --- | ---: | --- | --- |
| 1: Stateful versus stateless | 100 original ToolSandbox scenarios | 2 execution modes | `run_experiment1_llm_<provider>.py` |
| 2: Cross-tool dependency | 100 ToolSandbox cases | Direct, CoT, ReAct, Self-Refine | `run_experiment2_llm_<provider>.py` |
| 3: Interaction noise | 100 ToolSandbox cases | Four strategies, one perturbed attempt per case and strategy | `run_experiment3_llm_<provider>.py` |
| 4: Backend migration | 30 MCP-Bench-derived cases | Four strategies across mock, sandbox, local MCP | `run_experiment4_llm_<provider>.py` |

Experiments 1 and 2 share the fixed `data/benchmark/experiment2_toolsandbox_original_100.jsonl` task manifest. Experiment 1 reconstructs its stateful/stateless cases from this file, so the original `Toolsandbox/` folder is not required for the paper runs and is excluded from the repository.

The paper calls the third backend “Live MCP”; the implementation uses real MCP servers over local stdio transport. The provider endpoints are DeepSeek `deepseek-v4-flash` and SiliconFlow `openai/gpt-oss-120b`, `google/gemma-4-31B-it`, and `Qwen/Qwen3.6-35B-A3B`. Provider-side temperature, top-p, and output-token defaults are retained unless explicitly overridden. API timeouts and request retries default to 60 s / 2 for Experiments 1–2, 120 s / 3 for Experiment 3, and 90 s / 3 for Experiment 4. The runners apply task and strategy step budgets.

## Setup on Windows

Use Python 3.12 and install the main dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt pytest
```

Set `DEEPSEEK_API_KEY` or `SILICONFLOW_API_KEY` in your environment. `.env.example` documents the variables; local `.env` files are ignored by Git. For Experiment 4, install Node.js and set up the five MCP servers:

```powershell
.\scripts\setup_experiment4_mcp.ps1
python scripts/preflight_experiment4_core_local.py
```

The setup script checks out [Accenture/mcp-bench](https://github.com/Accenture/mcp-bench) at revision `7a8eaeae83a842a2949080acc5473f65e1569daf` under ignored `external/`. Its third-party source and installed packages are not committed here.

## Validate inputs without model calls

```powershell
python scripts/validate_paper_inputs.py
python scripts/run_experiment4_llm_deepseek.py --validate-only
python -m pytest tests -q
```

## Run the experiments

The PowerShell launcher chooses the official experiment entry point. Examples:

```powershell
.\scripts\run_paper_experiment.ps1 -Experiment 1 -Provider deepseek
.\scripts\run_paper_experiment.ps1 -Experiment 2 -Provider siliconflow -Model openai/gpt-oss-120b
.\scripts\run_paper_experiment.ps1 -Experiment 3 -Provider siliconflow -Model google/gemma-4-31B-it
.\scripts\run_paper_experiment.ps1 -Experiment 4 -Provider deepseek
```

For all four backbones, run each experiment once per model. The Python entry points also accept `--help`, `--resume`, and an optional output suffix. Runs write their traces, summaries, reports, and logs only under `outputs/`.

After the necessary local traces exist, run the offline same-trace audit:

```powershell
python scripts/audit_same_trace_projections.py
```

The audit expects the canonical Experiment 2 summaries, Experiment 3 traces, and Experiment 4 summaries for all four models in `outputs/`. It performs no model calls. The local output directory remains outside version control.

## Data sources

Experiments 1–3 use fixed tasks derived from the 100 unaugmented ToolSandbox scenarios. The upstream `Toolsandbox/` folder is not uploaded. Tests that directly inspect its original JSON or CSV files skip when those local source files are absent. Experiment 4 uses 30 fixed cases derived from MCP-Bench local server tasks. The manifest and dataset card are in `data/benchmark/` and `docs/`. Cite the original ToolSandbox and MCP-Bench publications when reusing these tasks. See `LICENSE` for this repository's code license and the upstream repositories for their respective licenses.
