"""Local-only E0 extraction and audit of the published same-trace sample.

This prepares data for review; it does not implement the proposed E1/E2 rules.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import audit_same_trace_projections as audit  # noqa: E402

DEST = Path(__file__).resolve().parent
VIEWS = tuple(audit.PROJECTION_ORDER)
FORBIDDEN = {"oracle_verdict", "oracle_components", "ground_truth", "dependency_satisfied", "recovery_success", "final_correct"}


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bad_keys(value: object, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN:
                found.append(path)
            found.extend(bad_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(bad_keys(child, f"{prefix}[{index}]"))
    return found


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    files = [*audit.MODEL_FILES_EXP2, *audit.MODEL_FILES_EXP3, *audit.MODEL_FILES_EXP4]
    source_manifest = []
    for model, path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        source_manifest.append({"model": model, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    counts = Counter()
    finish_flag_counts = Counter()
    view_counts = {view: Counter() for view in VIEWS}
    issues = Counter()
    issue_examples: dict[str, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()
    contexts: dict[str, dict] = {}
    predictions_path = DEST / "original_predictions.csv"
    evidence_path = DEST / "trajectories_clean.jsonl"
    labels_path = DEST / "oracle_labels.jsonl"
    context_path = DEST / "contexts.jsonl"
    columns = ["trajectory_id", "context_id", "source", "model", "strategy", "backend", "perturbation_type", "oracle_verdict", *VIEWS]

    with (evidence_path.open("w", encoding="utf-8") as evidence_file,
          labels_path.open("w", encoding="utf-8") as label_file,
          predictions_path.open("w", encoding="utf-8", newline="") as prediction_file):
        writer = csv.DictWriter(prediction_file, fieldnames=columns)
        writer.writeheader()
        for source, loader in (("dependency", audit.load_exp2), ("recovery", audit.load_exp3), ("backend", audit.load_exp4)):
            rows = loader()
            audit._validate_field_lock(rows)
            for item in rows:
                if item.trace_id in seen_ids:
                    raise ValueError(f"Duplicate trajectory ID: {item.trace_id}")
                seen_ids.add(item.trace_id)
                context_id = hashlib.sha256(canonical(item.context).encode("utf-8")).hexdigest()
                contexts.setdefault(context_id, item.context)
                trace = audit._trace_calls(item.run)
                final_state = audit._safe_get(item.run, ["experiment_result", "final_state"], {})
                initial_state = trace[0].get("pre_state_snapshot") if trace else None
                last_state = trace[-1].get("post_state_snapshot") if trace else None
                if not trace:
                    add_issue(issues, issue_examples, "zero_tool_steps", item.trace_id)
                if initial_state is None:
                    add_issue(issues, issue_examples, "missing_initial_state", item.trace_id)
                if not isinstance(final_state, dict):
                    add_issue(issues, issue_examples, "missing_terminal_state", item.trace_id)
                if trace and final_state != last_state:
                    add_issue(issues, issue_examples, "terminal_reconstruction_mismatch", item.trace_id)
                for before, after in zip(trace, trace[1:]):
                    if before.get("post_state_hash") != after.get("pre_state_hash"):
                        add_issue(issues, issue_examples, "state_hash_chain_break", item.trace_id)
                        break
                steps = []
                for call in trace:
                    step = {
                        "action": audit._action(call),
                        "observation": audit._agent_observation(call),
                        "state_transition": audit._state_transition(call),
                        "backend_evidence": audit._backend_condition(call),
                    }
                    if not isinstance(step["state_transition"].get("pre_state_snapshot"), dict) or not isinstance(step["state_transition"].get("post_state_snapshot"), dict):
                        add_issue(issues, issue_examples, "missing_transition_snapshot", item.trace_id)
                    if step["observation"].get("success") is None:
                        add_issue(issues, issue_examples, "missing_visible_success_flag", item.trace_id)
                    leakage = bad_keys(step)
                    if leakage:
                        add_issue(issues, issue_examples, "oracle_like_key_in_evidence", item.trace_id + ": " + ",".join(leakage[:3]))
                    steps.append(step)
                if bad_keys(item.context):
                    add_issue(issues, issue_examples, "oracle_like_key_in_context", item.trace_id)

                llm_steps = item.run.get("llm_steps") or []
                last_llm_step = llm_steps[-1] if llm_steps else {}
                last_parsed_action = last_llm_step.get("parsed_action")
                last_finish_flag = last_parsed_action.get("finish") if isinstance(last_parsed_action, dict) else None
                finish_flag_counts[f"{type(last_finish_flag).__name__}:{last_finish_flag}"] += 1
                if last_finish_flag is not None and not isinstance(last_finish_flag, bool):
                    add_issue(issues, issue_examples, "nonboolean_last_finish_flag", item.trace_id)
                termination = {
                    "run_mode": item.run.get("mode"),
                    "llm_step_count": len(llm_steps),
                    "last_parsed_action": last_parsed_action.get("tool_name") if isinstance(last_parsed_action, dict) else None,
                    "last_finish_flag": last_finish_flag,
                    "last_parse_error": last_llm_step.get("parse_error"),
                    "parse_error_count": item.run.get("parse_error_count"),
                    "tool_step_count": len(trace),
                }
                clean = {
                    "trajectory_id": item.trace_id,
                    "context_id": context_id,
                    "source": item.source,
                    "model": item.model,
                    "strategy": item.strategy,
                    "backend": item.backend,
                    "perturbation_type": item.metadata.get("noise_type"),
                    "K_public": item.context,
                    "task_description_agent_visible": item.run.get("task_description"),
                    "initial_state": initial_state,
                    "steps": steps,
                    "terminal_state": final_state,
                    "termination": termination,
                }
                labels = {
                    "trajectory_id": item.trace_id,
                    "context_id": context_id,
                    "oracle_verdict": int(item.full_verdict),
                    "oracle_components": {
                        "terminal_state_correct": item.terminal_state_correct,
                        "dependency_compliant": item.dependency_compliant,
                        "recovery_ok": item.recovery_ok,
                    },
                    "original_metadata_for_scoring": item.metadata,
                }
                evidence_file.write(canonical(clean) + "\n")
                label_file.write(canonical(labels) + "\n")
                prediction = {"trajectory_id": item.trace_id, "context_id": context_id, "source": item.source, "model": item.model,
                              "strategy": item.strategy, "backend": item.backend, "perturbation_type": item.metadata.get("noise_type"),
                              "oracle_verdict": int(item.full_verdict)}
                for view in VIEWS:
                    predicted = int(audit._projection_verdict(view, item))
                    prediction[view] = predicted
                    view_counts[view]["n"] += 1
                    view_counts[view]["false_acceptance"] += int(predicted == 1 and not item.full_verdict)
                    view_counts[view]["false_rejection"] += int(predicted == 0 and item.full_verdict)
                    view_counts[view]["disagreement"] += int(predicted != int(item.full_verdict))
                writer.writerow(prediction)
                counts[(source, item.model, item.strategy)] += 1
            print(f"Prepared {source}: {len(rows)} trajectories", flush=True)
            del rows
            gc.collect()

    with context_path.open("w", encoding="utf-8") as handle:
        for context_id, context in sorted(contexts.items()):
            handle.write(canonical({"context_id": context_id, "K_public": context}) + "\n")
    n = len(seen_ids)
    if n != 4640 or len(contexts) != 590:
        raise ValueError(f"Unexpected audit sample: {n} trajectories, {len(contexts)} contexts")
    original = json.loads((ROOT / "outputs" / "metrics" / "same_trace_projection_audit.json").read_text(encoding="utf-8"))
    comparisons = []
    for view in VIEWS:
        old = next(row for row in original["overall"] if row["projection"] == view)
        row = {"view": view, "n": n}
        for key in ("false_acceptance", "false_rejection", "disagreement"):
            count = view_counts[view][key]
            row[key + "_count"] = count
            row[key + "_rate"] = count / n
            row[key + "_matches_original"] = abs(count / n - old[key]) < 1e-12
        comparisons.append(row)
    if not all(row[key] for row in comparisons for key in row if key.endswith("_matches_original")):
        raise ValueError("Original Table 14 rates were not reproduced")
    report = {
        "n_trajectories": n,
        "n_contexts": len(contexts),
        "source_counts": dict(Counter({source: sum(value for (s, _, _), value in counts.items() if s == source) for source in ("dependency", "recovery", "backend")})),
        "source_model_strategy_counts": [{"source": s, "model": m, "strategy": p, "n": v} for (s, m, p), v in sorted(counts.items())],
        "last_finish_flag_counts": dict(finish_flag_counts),
        "table14_reproduction": comparisons,
        "issues": dict(issues),
        "issue_examples": dict(issue_examples),
        "source_manifest": source_manifest,
        "notes": ["Trajectory IDs are deterministic identifiers assigned by the original audit loader from model, source row index, strategy and task; source logs contain no standalone UUID.",
                  "Context IDs are SHA-256 hashes of the original audit context K; K is an implementation-specific context summary, not a full task-oracle formula.",
                  "Original Table 14 verdict rates were reproduced from the fixed source records. Collision and byte-cost columns are documented in the existing audit summary and are not recomputed here."]
    }
    (DEST / "qa_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DEST / "source_manifest.json").write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_trajectories": n, "n_contexts": len(contexts), "issues": dict(issues), "table14": comparisons}, indent=2), flush=True)


def add_issue(issues: Counter, examples: dict[str, list[str]], name: str, trace_id: str) -> None:
    issues[name] += 1
    if len(examples[name]) < 10:
        examples[name].append(trace_id)


if __name__ == "__main__":
    main()
