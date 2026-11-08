# Experiment 4 MCP-Bench Core Local 30

## Purpose

This dataset evaluates backend migration without depending on internet access:

```text
mock -> strict sandbox -> real local MCP (stdio)
```

It is a local adaptation of the task and dependency patterns in
`external/mcp-bench/tasks`. It is not a verbatim 30-row sample of MCP-Bench.
The prompts are smaller, deterministic probes with rule-based goals.

## Scope

- 30 cases
- 5 offline MCP-Bench servers
- 18 single-server, 8 two-server, and 4 three-server cases
- 8 schema cases, 8 sequential-dependency cases, 6 cross-server handoffs,
  5 recovery/conditional cases, and 3 long-horizon cases
- no API keys, public APIs, URLs, or external network access

The included servers are:

| Server ID | MCP-Bench server | Why included |
|---|---|---|
| `unit_converter` | Unit Converter | strict enums and deterministic conversions |
| `time` | Time MCP | strict time/IANA timezone schemas |
| `math` | Math MCP | deterministic arithmetic and schema quirks |
| `scientific_computing` | Scientific Computing | stateful tensor workflows |
| `medical_calculator` | Medical Calculator | deterministic formulas and conditional chains |

## Files

- `data/benchmark/experiment4_mcpbench_core_local_30.jsonl`: case records
- `data/benchmark/experiment4_mcpbench_core_local_30_manifest.json`: counts and hash
- `scripts/build_experiment4_core_local_dataset.py`: deterministic builder
- `tests/test_experiment4_core_local_dataset.py`: structural validation

Regenerate with:

```powershell
python scripts\build_experiment4_core_local_dataset.py
```

## Record format

Every JSONL record contains:

- a backend-blind `user_prompt`;
- source MCP-Bench task IDs;
- server topology and difficulty labels;
- an `oracle_plan` with explicit data dependencies;
- rule-based `goals` and tolerances;
- qualified required tool names;
- backend-discrimination metadata;
- an explicit offline/runtime policy.

Oracle arguments can reference an earlier step:

```json
{"$ref": "convert.output.converted_value"}
```

The optional `transform` field applies a deterministic adapter, such as
extracting an hour from an ISO datetime. References must always point backward
in the oracle DAG.

## Backend semantics

The prompt must be identical across backends and must not disclose which
backend is under test.

- `mock`: permissive aliases and simplified state/workflow behavior.
- `sandbox`: strict deterministic schema, enum, state, and ordering checks.
- `local_mcp`: the real local MCP-Bench server over stdio.

The primary result must report first-attempt transfer separately from
closed-loop recovery. Infrastructure errors must not be counted as agent
failures.

## Runtime status

Ten cases use Scientific Computing's in-memory tensor store. The Core Local
runtime now keeps one persistent stdio MCP session per server for the lifetime
of a case, then closes every process deterministically.

Validated commands:

```powershell
# Verify all five real local MCP servers.
.\.venv_mcp_live\Scripts\python.exe scripts\preflight_experiment4_core_local.py

# Qualify all 30 cases on mock, sandbox, and real local MCP (90 runs).
.\.venv_mcp_live\Scripts\python.exe scripts\run_experiment4_core_local_oracle.py --backends mock,sandbox,local_mcp --output-suffix _full

# Validate the complete LLM experiment matrix without spending API tokens.
.\.venv_mcp_live\Scripts\python.exe scripts\run_experiment4_llm_deepseek.py --validate-only

# Run/resume 30 cases x 4 strategies x 3 backends (360 runs).
# The launcher securely prompts for the API key and does not save it.
.\scripts\run_experiment4_core_local.ps1
```

The five MCP tool servers are offline and require no outbound network. Only the
DeepSeek policy call requires internet access and an API key. Checkpoints are
appended after every LLM run, so a network interruption can be resumed safely.

## Evaluation

Rule-based goals are primary. Supported goal concepts include:

- exact and approximate numeric equality;
- approximate matrix/array equality;
- unordered numeric arrays;
- time-of-day comparison within ISO datetimes;
- expected error substring checks.

Dependency compliance is evaluated separately from final correctness. A case
may reach the right number while still failing required tool-order or
cross-server handoff checks.

## Known limitations

- The DeepSeek API is outside the offline MCP boundary; a configured key and
  reachable API endpoint are required for the 360-run LLM experiment.
- FastMCP and two upstream MCP-Bench servers emit deprecation/duplicate-tool
  warnings during startup. These warnings do not fail preflight or tool calls.
- Time MCP uses the current date internally. Cases only evaluate fixed-offset
  Asia/UTC time-of-day fields, not the date.
- Expected backend gaps remain model-dependent. Cases should not be removed
  solely because one model solves all three backends.

