#!/usr/bin/env python3
"""Same-trace projection audit for ATT experiments.

This script reuses existing execution logs and does not call any LLM.  For each
complete trace, it constructs five field-locked projections of the *same*
trace and compares their deterministic audit verdicts with the full
trajectory-level verdict.  Projection collisions are computed within the same
non-execution context K, not across unrelated tasks or protocols.

Outputs:
  outputs/metrics/same_trace_projection_audit.json
  outputs/metrics/same_trace_projection_audit_summary.csv
  outputs/reports/same_trace_projection_audit.md
  outputs/reports/same_trace_projection_audit_table.tex
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_METRICS = ROOT / "outputs" / "metrics"
OUT_REPORTS = ROOT / "outputs" / "reports"


MODEL_FILES_EXP2 = [
    ("DeepSeek-V4-Flash", OUT_METRICS / "experiment2_llm_deepseek_summary_toolsandbox_original_100.json"),
    ("GPT-OSS-120B", OUT_METRICS / "experiment2_llm_siliconflow_summary_gpt_oss_120b_toolsandbox_original_100.json"),
    ("Gemma-4-31B-it", OUT_METRICS / "experiment2_llm_siliconflow_summary_gemma_4_31b_it_toolsandbox_original_100.json"),
    ("Qwen3.6-35B-A3B", OUT_METRICS / "experiment2_llm_siliconflow_summary_qwen3_6_35b_a3b_toolsandbox_original_100.json"),
]

MODEL_FILES_EXP3 = [
    ("DeepSeek-V4-Flash", ROOT / "outputs" / "cases" / "experiment3_llm_deepseek_traces_toolsandbox_original_100_noise_single_attempt.jsonl"),
    ("GPT-OSS-120B", ROOT / "outputs" / "cases" / "experiment3_llm_siliconflow_traces_gpt_oss_120b_toolsandbox_original_100_noise_single_attempt.jsonl"),
    ("Gemma-4-31B-it", ROOT / "outputs" / "cases" / "experiment3_llm_siliconflow_traces_gemma_4_31b_it_toolsandbox_original_100_noise_single_attempt.jsonl"),
    ("Qwen3.6-35B-A3B", ROOT / "outputs" / "cases" / "experiment3_llm_siliconflow_traces_qwen3_6_35b_a3b_toolsandbox_original_100_noise_single_attempt_no_thinking.jsonl"),
]

MODEL_FILES_EXP4 = [
    ("DeepSeek-V4-Flash", ROOT / "outputs" / "core_local" / "experiment4_core_local_llm_full_summary.json"),
    ("GPT-OSS-120B", ROOT / "outputs" / "core_local" / "experiment4_core_local_llm_gpt_oss_120b_full_summary.json"),
    ("Gemma-4-31B-it", ROOT / "outputs" / "core_local" / "experiment4_core_local_llm_gemma_4_31b_it_full_summary.json"),
    ("Qwen3.6-35B-A3B", ROOT / "outputs" / "core_local" / "experiment4_core_local_llm_qwen3_6_35b_a3b_full_summary.json"),
]


PROJECTION_ORDER = [
    "call_only",
    "action_observation",
    "outcome_only",
    "state_aware",
    "full_trace",
]

PROJECTION_LABELS = {
    "call_only": "Call-only",
    "action_observation": "Action--Observation",
    "outcome_only": "Outcome-only",
    "state_aware": "State-aware",
    "full_trace": "Full audit",
}

PROJECTION_FIELDS = {
    "call_only": "A_{1:H}",
    "action_observation": "(A_{1:H}, O_{1:H})",
    "outcome_only": "S_H",
    "state_aware": "(A_{1:H}, O_{1:H}, DeltaS_{1:H})",
    "full_trace": "(S_0, A_{1:H}, O_{1:H}, DeltaS_{1:H}, B_{1:H})",
}


@dataclass
class AuditTrace:
    trace_id: str
    source: str
    model: str
    strategy: str
    full_verdict: bool
    terminal_state_correct: bool
    dependency_compliant: bool | None
    recovery_ok: bool | None
    backend: str | None
    run: dict[str, Any]
    metadata: dict[str, Any]
    context: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _safe_get(d: dict[str, Any], path: list[Any], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
            cur = cur[key]
        else:
            return default
    return cur


def _state_correct_from_run(run: dict[str, Any]) -> bool:
    return bool(_safe_get(run, ["experiment_result", "state_metrics", "all_passed"], False))


def _trace_calls(run: dict[str, Any]) -> list[dict[str, Any]]:
    return list(_safe_get(run, ["experiment_result", "trace"], []) or [])


def _action(call: dict[str, Any]) -> dict[str, Any]:
    """Return A_t: only the selected tool and its arguments."""
    return {
        "tool": call.get("tool_name"),
        "args": call.get("args"),
    }


def _agent_observation(call: dict[str, Any]) -> dict[str, Any]:
    """Return O_t exactly as exposed in the policy history.

    The runner exposes success, error, and observation together after a tool
    attempt.  Executor-only status and condition checks remain in B_t.
    """
    return {
        "success": call.get("success"),
        "error": call.get("error"),
        "observation": call.get("observation"),
    }


def _state_transition(call: dict[str, Any]) -> dict[str, Any]:
    """Return Delta S_t from paired state snapshots and transition markers."""
    return {
        "pre_state_hash": call.get("pre_state_hash"),
        "post_state_hash": call.get("post_state_hash"),
        "hash_changed": call.get("hash_changed"),
        "state_changed": call.get("state_changed"),
        "pre_state_snapshot": call.get("pre_state_snapshot"),
        "post_state_snapshot": call.get("post_state_snapshot"),
    }


def _backend_condition(call: dict[str, Any]) -> dict[str, Any]:
    """Return B_t: realized execution conditions omitted by reduced views."""
    return {
        "tool_version": call.get("tool_version"),
        "status": call.get("status"),
        "partial": call.get("partial"),
        "async_pending": call.get("async_pending"),
        "precondition_results": call.get("precondition_results"),
        "postcondition_results": call.get("postcondition_results"),
        "permission_results": call.get("permission_results"),
        "attempt_count": call.get("attempt_count"),
        "duration_ms": call.get("duration_ms"),
        "timeout_ms": call.get("timeout_ms"),
        "scheduled_effect_ids": call.get("scheduled_effect_ids"),
        "applied_effect_ids": call.get("applied_effect_ids"),
        "applied_effect_count": call.get("applied_effect_count"),
        "consistency_warning": call.get("consistency_warning"),
        "backend_name": call.get("backend_name"),
        "metadata": call.get("metadata"),
    }


def _projection_payload(name: str, item: AuditTrace) -> dict[str, Any]:
    run = item.run
    trace = _trace_calls(run)
    if name == "call_only":
        return {"actions": [_action(c) for c in trace]}
    if name == "action_observation":
        return {
            "interactions": [
                {"action": _action(c), "observation": _agent_observation(c)}
                for c in trace
            ],
        }
    if name == "outcome_only":
        return {"terminal_state": _safe_get(run, ["experiment_result", "final_state"], {})}
    if name == "state_aware":
        return {
            "interactions": [
                {
                    "action": _action(c),
                    "observation": _agent_observation(c),
                    "state_transition": _state_transition(c),
                }
                for c in trace
            ]
        }
    if name == "full_trace":
        initial_state = trace[0].get("pre_state_snapshot") if trace else {}
        return {
            "initial_state": initial_state,
            "interactions": [
                {
                    "action": _action(c),
                    "observation": _agent_observation(c),
                    "state_transition": _state_transition(c),
                    "backend_condition": _backend_condition(c),
                }
                for c in trace
            ],
        }
    raise KeyError(name)


def _projection_key(name: str, item: AuditTrace) -> str:
    payload = _projection_payload(name, item)
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _context_projection_key(name: str, item: AuditTrace) -> str:
    """Hash (K, Xi_phi), the equivalence class used for collisions."""
    raw = json.dumps(
        {"context": item.context, "projection": _projection_payload(name, item)},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _projection_cost(name: str, item: AuditTrace) -> int:
    payload = _projection_payload(name, item)
    return len(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))


def _validate_field_lock(rows: list[AuditTrace]) -> None:
    """Fail fast if a future edit breaks the declared projection nesting."""
    for item in rows:
        call = _projection_payload("call_only", item)
        ao = _projection_payload("action_observation", item)
        outcome = _projection_payload("outcome_only", item)
        state = _projection_payload("state_aware", item)
        full = _projection_payload("full_trace", item)

        ao_actions = [step["action"] for step in ao["interactions"]]
        if call["actions"] != ao_actions:
            raise ValueError(f"Call-only is not a deterministic projection of AO: {item.trace_id}")

        state_ao = [
            {"action": step["action"], "observation": step["observation"]}
            for step in state["interactions"]
        ]
        if ao["interactions"] != state_ao:
            raise ValueError(f"AO is not a deterministic projection of state-aware: {item.trace_id}")

        full_state = [
            {
                "action": step["action"],
                "observation": step["observation"],
                "state_transition": step["state_transition"],
            }
            for step in full["interactions"]
        ]
        if state["interactions"] != full_state:
            raise ValueError(f"State-aware is not a deterministic projection of full audit: {item.trace_id}")

        terminal_from_state = (
            state["interactions"][-1]["state_transition"]["post_state_snapshot"]
            if state["interactions"]
            else {}
        )
        if outcome["terminal_state"] != terminal_from_state:
            raise ValueError(f"S_H cannot be reconstructed from the state-aware view: {item.trace_id}")


def _actions_are_parseable(payload: dict[str, Any]) -> bool:
    actions = payload.get("actions")
    if actions is None:
        actions = [step.get("action") for step in payload.get("interactions", [])]
    return bool(actions) and all(
        isinstance(action, dict)
        and isinstance(action.get("tool"), str)
        and bool(action.get("tool"))
        and isinstance(action.get("args"), dict)
        for action in actions
    )


def _observations_show_success(payload: dict[str, Any]) -> bool:
    interactions = payload.get("interactions", [])
    if not interactions:
        return False
    for step in interactions:
        observation = step.get("observation") or {}
        if observation.get("success") is not True:
            return False
        if observation.get("error") not in (None, "", {}):
            return False
        value = observation.get("observation")
        if isinstance(value, dict) and value.get("exception") not in (None, "", {}):
            return False
    return True


def _state_transitions_are_consistent(payload: dict[str, Any]) -> bool:
    interactions = payload.get("interactions", [])
    if not interactions:
        return False
    for step in interactions:
        delta = step.get("state_transition") or {}
        pre_hash = delta.get("pre_state_hash")
        post_hash = delta.get("post_state_hash")
        pre_snapshot = delta.get("pre_state_snapshot")
        post_snapshot = delta.get("post_state_snapshot")
        if not isinstance(pre_snapshot, dict) or not isinstance(post_snapshot, dict):
            return False
        if not isinstance(pre_hash, str) or not isinstance(post_hash, str):
            return False
        observed_hash_change = pre_hash != post_hash
        if delta.get("hash_changed") is not observed_hash_change:
            return False
        declared_change = delta.get("state_changed")
        if isinstance(declared_change, bool) and declared_change != observed_hash_change:
            return False
    return True


def _projection_verdict(name: str, item: AuditTrace) -> bool:
    payload = _projection_payload(name, item)
    if name == "call_only":
        return _actions_are_parseable(payload)
    if name == "action_observation":
        return _actions_are_parseable(payload) and _observations_show_success(payload)
    if name == "outcome_only":
        # The evaluator applies the task oracle from K to the retained S_H.
        return item.terminal_state_correct
    if name == "state_aware":
        return (
            item.terminal_state_correct
            and _actions_are_parseable(payload)
            and _observations_show_success(payload)
            and _state_transitions_are_consistent(payload)
        )
    if name == "full_trace":
        return item.full_verdict
    raise KeyError(name)


def load_exp2() -> list[AuditTrace]:
    rows: list[AuditTrace] = []
    for model, path in MODEL_FILES_EXP2:
        data = _read_json(path)
        for idx, rec in enumerate(data.get("case_results", [])):
            run = rec["run"]
            trace = _trace_calls(run)
            terminal = bool(rec.get("final_state_correct", _state_correct_from_run(run)))
            dependency = bool(rec.get("dependency_correct", terminal))
            rows.append(
                AuditTrace(
                    trace_id=f"exp2::{model}::{idx}::{rec.get('strategy')}::{rec.get('case_name')}",
                    source="dependency",
                    model=model,
                    strategy=str(rec.get("strategy", "")),
                    full_verdict=dependency,
                    terminal_state_correct=terminal,
                    dependency_compliant=dependency,
                    recovery_ok=None,
                    backend=_safe_get(run, ["experiment_result", "trace", 0, "backend_name"], None),
                    run=run,
                    metadata={
                        "difficulty_level": rec.get("difficulty_level"),
                        "missing_prerequisite_count": rec.get("missing_prerequisite_count", 0),
                        "wrong_order_call_count": rec.get("wrong_order_call_count", 0),
                    },
                    context={
                        "task_family": "ToolSandbox-dependency",
                        "task_key": rec.get("case_name") or run.get("task_name"),
                        "oracle_scope": {
                            "difficulty_level": rec.get("difficulty_level"),
                            "dependency_type": rec.get("dependency_type"),
                        },
                        "initial_state_hash": trace[0].get("pre_state_hash") if trace else None,
                        "evaluation_protocol": "dependency",
                    },
                )
            )
    return rows


def load_exp3() -> list[AuditTrace]:
    rows: list[AuditTrace] = []
    for model, path in MODEL_FILES_EXP3:
        for idx, rec in enumerate(_iter_jsonl(path)):
            attempts = rec.get("attempts") or []
            if not attempts:
                continue
            attempt = attempts[0]
            run = attempt["run"]
            trace = _trace_calls(run)
            terminal = bool(attempt.get("state_correct", _state_correct_from_run(run)))
            full = bool(rec.get("success_at_1", attempt.get("final_correct", False)))
            recovery_ok = None
            if attempt.get("fault_triggered"):
                recovery_ok = bool(attempt.get("recovery_detected", False))
            rows.append(
                AuditTrace(
                    trace_id=f"exp3::{model}::{idx}::{rec.get('strategy')}::{rec.get('noise_type')}::{rec.get('case_name')}",
                    source="recovery",
                    model=model,
                    strategy=str(rec.get("strategy", "")),
                    full_verdict=full,
                    terminal_state_correct=terminal,
                    dependency_compliant=None,
                    recovery_ok=recovery_ok,
                    backend=_safe_get(run, ["experiment_result", "trace", 0, "backend_name"], None),
                    run=run,
                    metadata={
                        "noise_type": rec.get("noise_type"),
                        "fault_triggered": bool(attempt.get("fault_triggered", False)),
                        "final_correct": bool(attempt.get("final_correct", False)),
                    },
                    context={
                        "task_family": "ToolSandbox-recovery",
                        "task_key": rec.get("canonical_task_key") or rec.get("base_case_name") or run.get("task_name"),
                        "oracle_scope": {
                            "difficulty_level": rec.get("difficulty_level"),
                            "noise_type": rec.get("noise_type"),
                            "noise_intensity": rec.get("noise_intensity"),
                            "noise_target_tool": rec.get("noise_target_tool"),
                        },
                        "initial_state_hash": trace[0].get("pre_state_hash") if trace else None,
                        "evaluation_protocol": {
                            "study": "recovery",
                            "step_budget": attempt.get("step_budget"),
                            "single_attempt": True,
                        },
                    },
                )
            )
    return rows


def load_exp4() -> list[AuditTrace]:
    rows: list[AuditTrace] = []
    for model, path in MODEL_FILES_EXP4:
        data = _read_json(path)
        for idx, rec in enumerate(data.get("case_results", [])):
            run = rec["run"]
            trace = _trace_calls(run)
            terminal = bool(rec.get("final_state_correct", _state_correct_from_run(run)))
            compliant = bool(rec.get("dependency_compliant", terminal))
            # In the backend study, the full audit requires both terminal state
            # success and prerequisite-tool compliance.
            full = terminal and compliant
            rows.append(
                AuditTrace(
                    trace_id=f"exp4::{model}::{idx}::{rec.get('strategy')}::{rec.get('backend')}::{rec.get('case_id')}",
                    source="backend",
                    model=model,
                    strategy=str(rec.get("strategy", "")),
                    full_verdict=full,
                    terminal_state_correct=terminal,
                    dependency_compliant=compliant,
                    recovery_ok=bool(rec.get("recovery_detected", False)),
                    backend=str(rec.get("backend", "")),
                    run=run,
                    metadata={
                        "backend": rec.get("backend"),
                        "case_id": rec.get("case_id"),
                        "raw_final_state_correct": terminal,
                        "raw_dependency_compliant": compliant,
                    },
                    context={
                        "task_family": "MCP-Bench-derived-backend",
                        "task_key": rec.get("case_id") or run.get("task_name"),
                        "task_title": rec.get("title"),
                        "oracle_scope": {
                            "difficulty": rec.get("difficulty"),
                            "topology": rec.get("topology"),
                        },
                        "initial_state_hash": trace[0].get("pre_state_hash") if trace else None,
                        "evaluation_protocol": {
                            "study": "backend",
                            "configured_backend": rec.get("backend"),
                        },
                    },
                )
            )
    return rows


def _failure_types(item: AuditTrace) -> set[str]:
    out: set[str] = set()
    if not item.terminal_state_correct:
        out.add("state")
    if item.dependency_compliant is False:
        out.add("dependency")
    if (
        item.source == "recovery"
        and item.metadata.get("fault_triggered")
        and not item.full_verdict
        and item.recovery_ok is False
    ):
        out.add("recovery")
    if item.source == "backend" and item.backend and item.backend != "mock" and not item.full_verdict:
        out.add("backend")
    return out


def summarize(rows: list[AuditTrace]) -> dict[str, Any]:
    overall: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = {}
    for subset_name, subset in [("all", rows)] + sorted((k, v) for k, v in _group_by(rows, lambda x: x.source).items()):
        # A collision is defined within the current reporting subset by a
        # verdict-conflicting equivalence class of (K, Xi_phi).
        key_labels: dict[str, dict[str, set[bool]]] = {p: defaultdict(set) for p in PROJECTION_ORDER}
        for p in PROJECTION_ORDER:
            for r in subset:
                key_labels[p][_context_projection_key(p, r)].add(r.full_verdict)

        table: list[dict[str, Any]] = []
        subset_full_costs = [_projection_cost("full_trace", r) for r in subset]
        avg_full_cost = sum(subset_full_costs) / len(subset_full_costs)
        for p in PROJECTION_ORDER:
            counts = Counter()
            failure_totals = Counter()
            failure_detected = Counter()
            costs = []
            collision_items = 0
            for r in subset:
                proj_v = _projection_verdict(p, r)
                full_v = r.full_verdict
                counts["n"] += 1
                counts["full_pass"] += int(full_v)
                counts["projection_pass"] += int(proj_v)
                counts["false_accept"] += int(proj_v and not full_v)
                counts["false_reject"] += int((not proj_v) and full_v)
                counts["disagree"] += int(proj_v != full_v)
                costs.append(_projection_cost(p, r))
                if len(key_labels[p][_context_projection_key(p, r)]) > 1:
                    collision_items += 1
                for ft in _failure_types(r):
                    failure_totals[ft] += 1
                    if not proj_v:
                        failure_detected[ft] += 1
            n = counts["n"]
            component_recalls = {
                "state_failure_recall": _optional_rate(failure_detected["state"], failure_totals["state"]),
                "dependency_failure_recall": _optional_rate(failure_detected["dependency"], failure_totals["dependency"]),
                "recovery_failure_recall": _optional_rate(failure_detected["recovery"], failure_totals["recovery"]),
                "backend_failure_recall": _optional_rate(failure_detected["backend"], failure_totals["backend"]),
            }
            available_recalls = [value for value in component_recalls.values() if value is not None]
            row = {
                "subset": subset_name,
                "projection": p,
                "projection_label": PROJECTION_LABELS[p],
                "n": n,
                "full_pass_rate": _rate(counts["full_pass"], n),
                "projection_pass_rate": _rate(counts["projection_pass"], n),
                "false_acceptance": _rate(counts["false_accept"], n),
                "false_rejection": _rate(counts["false_reject"], n),
                "disagreement": _rate(counts["disagree"], n),
                "collision_rate": _rate(collision_items, n),
                **component_recalls,
                "trajectory_failure_recall": (
                    sum(available_recalls) / len(available_recalls) if available_recalls else None
                ),
                "avg_log_bytes": sum(costs) / len(costs),
                "relative_cost": (sum(costs) / len(costs)) / avg_full_cost,
            }
            table.append(row)
        if subset_name == "all":
            overall = table
        else:
            by_source[subset_name] = table

    return {
        "n_traces": len(rows),
        "n_contexts": len(
            {
                json.dumps(r.context, sort_keys=True, ensure_ascii=False, default=str)
                for r in rows
            }
        ),
        "sources": dict(Counter(r.source for r in rows)),
        "models": dict(Counter(r.model for r in rows)),
        "strategies": dict(Counter(r.strategy for r in rows)),
        "projection_fields": PROJECTION_FIELDS,
        "collision_key": "(K, Xi_phi) within each reporting subset",
        "trajectory_failure_recall": "macro-average over available state, dependency, recovery, and backend failure recalls",
        "overall": overall,
        "by_source": by_source,
    }


def _group_by(rows: list[AuditTrace], fn):
    groups: dict[str, list[AuditTrace]] = defaultdict(list)
    for row in rows:
        groups[str(fn(row))].append(row)
    return groups


def _rate(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _optional_rate(num: float, den: float) -> float | None:
    return float(num) / float(den) if den else None


def _pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{100*x:.1f}%"


def write_csv(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "subset",
        "projection",
        "projection_label",
        "n",
        "full_pass_rate",
        "projection_pass_rate",
        "false_acceptance",
        "false_rejection",
        "disagreement",
        "collision_rate",
        "state_failure_recall",
        "dependency_failure_recall",
        "recovery_failure_recall",
        "backend_failure_recall",
        "trajectory_failure_recall",
        "avg_log_bytes",
        "relative_cost",
    ]
    rows = list(summary["overall"])
    for table in summary["by_source"].values():
        rows.extend(table)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    def table(rows: list[dict[str, Any]]) -> list[str]:
        lines = [
            "| Projection | False acceptance | False rejection | Disagreement | Collision | Trajectory-failure recall | State recall | Dependency recall | Recovery recall | Backend recall | Rel. cost |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            lines.append(
                "| {projection_label} | {fa} | {fr} | {dis} | {col} | {tfr} | {sr} | {dr} | {rr} | {br} | {cost:.3f} |".format(
                    projection_label=r["projection_label"],
                    fa=_pct(r["false_acceptance"]),
                    fr=_pct(r["false_rejection"]),
                    dis=_pct(r["disagreement"]),
                    col=_pct(r["collision_rate"]),
                    tfr=_pct(r["trajectory_failure_recall"]),
                    sr=_pct(r["state_failure_recall"]),
                    dr=_pct(r["dependency_failure_recall"]),
                    rr=_pct(r["recovery_failure_recall"]),
                    br=_pct(r["backend_failure_recall"]),
                    cost=r["relative_cost"],
                )
            )
        return lines

    lines = [
        "# Same-Trace Projection Audit",
        "",
        f"- Total traces: {summary['n_traces']}",
        f"- Evaluation contexts K: {summary['n_contexts']}",
        f"- Sources: {summary['sources']}",
        "- Collision key: `(K, Xi_phi)` within each reporting subset",
        "- Trajectory-failure recall: macro-average over available state, dependency, recovery, and backend failure recalls",
        "- Relative cost: average serialized projection size relative to the field-locked full audit trace",
        "",
        "## Locked projection fields",
        "",
        *[f"- {PROJECTION_LABELS[name]}: `{fields}`" for name, fields in PROJECTION_FIELDS.items()],
        "",
        "## Overall",
        "",
        *table(summary["overall"]),
    ]
    for name, rows in summary["by_source"].items():
        lines.extend(["", f"## Source: {name}", "", *table(rows)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(summary: dict[str, Any], path: Path) -> None:
    rows = summary["overall"]

    def fmt(x: float | None) -> str:
        if x is None:
            return "n/a"
        return f"{100*x:.1f}\\%"

    lines = [
        "\\begin{table}[tb]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\caption{Same-trace projection audit over existing execution logs. Each field-locked projection is computed offline from the same complete trajectory and compared with the full trajectory-level audit verdict. Trajectory-Failure Recall is the macro-average over available state, dependency, recovery, and backend failure recalls. Rel. cost reports average serialized projection size relative to the full audit trace.}",
        "\\label{tab:same_trace_projection_audit}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Projection &",
        "\\shortstack{False\\\\Accept.} &",
        "\\shortstack{False\\\\Reject.} &",
        "\\shortstack{Disagree.} &",
        "\\shortstack{Collision} &",
        "\\shortstack{Trajectory-Failure\\\\Recall} &",
        "\\shortstack{Rel.\\\\Cost} \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['projection_label']} & "
            f"{fmt(r['false_acceptance'])} & "
            f"{fmt(r['false_rejection'])} & "
            f"{fmt(r['disagreement'])} & "
            f"{fmt(r['collision_rate'])} & "
            f"{fmt(r['trajectory_failure_recall'])} & "
            f"{r['relative_cost']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(load_exp2())
    rows.extend(load_exp3())
    rows.extend(load_exp4())
    _validate_field_lock(rows)
    summary = summarize(rows)
    full_row = next(row for row in summary["overall"] if row["projection"] == "full_trace")
    if any(
        full_row[name] != 0.0
        for name in ("false_acceptance", "false_rejection", "disagreement", "collision_rate")
    ):
        raise ValueError("Full audit must reproduce the complete trajectory-level verdict exactly")
    json_path = OUT_METRICS / "same_trace_projection_audit.json"
    csv_path = OUT_METRICS / "same_trace_projection_audit_summary.csv"
    md_path = OUT_REPORTS / "same_trace_projection_audit.md"
    tex_path = OUT_REPORTS / "same_trace_projection_audit_table.tex"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(summary, csv_path)
    write_markdown(summary, md_path)
    write_latex(summary, tex_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")
    print(f"Total traces: {summary['n_traces']}")


if __name__ == "__main__":
    main()
