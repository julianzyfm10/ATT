"""Enumerate the declared Exp3 support and evaluate candidate log projections."""

import argparse
from collections import defaultdict
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from math import log2
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from audit_same_trace_projections import PROJECTION_ORDER, _projection_payload
from toolsim.core.world_state import WorldState
from toolsim.execution.stateful_executor import ExecutorConfig, StatefulExecutor
from toolsim.tools.file_tools import FILE_TOOLS

DESIGN = HERE / "design.json"
ATOMS = ("payload_content", "marker_content", "tool_names", "file_order", "call_success", "actions", "observations")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def file_content(state, file_id):
    return state["entities"].get("file", {}).get(file_id, {}).get("content")


def execute(context, branch):
    state = WorldState()
    profile = {"transient_failures": {"file.write": 1}} if branch.get("first_write_fault") else None
    executor = StatefulExecutor(FILE_TOOLS, config=ExecutorConfig(fault_profile=profile))
    trace = [executor.execute("file.write", state, {"file_id": file_id, "content": content}).to_dict()
             for file_id, content in branch["writes"]]
    expected_success = [not branch.get("first_write_fault") or index > 0 for index in range(len(trace))]
    if [step["success"] for step in trace] != expected_success:
        raise AssertionError((context["id"], branch["id"], "unexpected execution outcome"))
    terminal = state.to_dict()
    required_order = context["required_write_order"]
    actual_order = [step["args"]["file_id"] for step in trace if step["success"]]
    compliant = not required_order or [file_id for file_id in actual_order if file_id in required_order] == required_order
    state_correct = all(file_content(terminal, key) == value for key, value in context["required_contents"].items())
    source = SimpleNamespace(run={"experiment_result": {"trace": trace, "final_state": terminal}})
    views = {name: _projection_payload(name, source) for name in PROJECTION_ORDER}
    atoms = {"payload_content": file_content(terminal, "payload"), "marker_content": file_content(terminal, "marker"),
             "tool_names": [step["tool_name"] for step in trace],
             "file_order": [step["args"]["file_id"] for step in trace],
             "call_success": [step["success"] for step in trace], "actions": views["call_only"]["actions"],
             "observations": [step["observation"] for step in views["action_observation"]["interactions"]]}
    return {"context": context["id"], "branch": branch["id"],
            "weight": str(Fraction(context["weight"]) * Fraction(branch["weight"])),
            "oracle_verdict": int(state_correct and compliant), "state_correct": state_correct,
            "order_compliant": compliant, "initial_state": views["full_trace"]["initial_state"],
            "terminal_state": terminal, "trace": trace, "historical_views": views, "atoms": atoms}


def verify_pairs(rows):
    indexed = {(row["context"], row["branch"]): row for row in rows}
    specs = [("same_calls_different_outcomes", "normal", "fault", "call_only", "call_success", True),
             ("same_terminal_different_order", "compliant", "reversed", "outcome_only", "file_order", True),
             ("terminal_is_sufficient", "approved_forward", "approved_reverse", "outcome_only", "file_order", False),
             ("terminal_is_sufficient", "draft_forward", "draft_reverse", "outcome_only", "file_order", False)]
    findings = []
    for context, first, second, view, added, opposite in specs:
        left, right = indexed[context, first], indexed[context, second]
        evidence = left["historical_views"][view]
        labels = [left["oracle_verdict"], right["oracle_verdict"]]
        added_values = [left["atoms"][added], right["atoms"][added]]
        if (canonical(evidence) != canonical(right["historical_views"][view])
                or canonical(left["initial_state"]) != canonical(right["initial_state"])
                or (labels[0] != labels[1]) != opposite or added_values[0] == added_values[1]):
            raise AssertionError((context, first, second, "pair does not match the design"))
        findings.append({"context": context, "branches": [first, second], "same_view": view,
                         "same_evidence": evidence, "verdicts": labels, "added_field": added,
                         "added_values": added_values})
    if [indexed["terminal_is_sufficient", name]["oracle_verdict"] for name in
            ("approved_forward", "approved_reverse", "draft_forward", "draft_reverse")] != [1, 1, 0, 0]:
        raise AssertionError("Terminal control must contain both verdicts")
    return findings


def statistics(rows, project):
    groups = defaultdict(lambda: [Fraction(0), Fraction(0)])
    total_bytes = Fraction(0)
    for row in rows:
        evidence = canonical(project(row))
        weight = Fraction(row["weight"])
        groups[row["context"], evidence][row["oracle_verdict"]] += weight
        total_bytes += weight * len(evidence)
    total = sum(sum(group) for group in groups.values())
    entropy = sum(-float(value / total) * log2(float(value / sum(group)))
                  for group in groups.values() for value in group if value)
    error = sum(min(group) for group in groups.values()) / total
    return {"entropy_bits": entropy, "bayes_error": float(error), "bayes_error_exact": str(error),
            "avg_bytes": float(total_bytes / total), "avg_bytes_exact": str(total_bytes / total),
            "mixed_label_groups": sum(all(group) for group in groups.values())}


def evaluate(rows, epsilon):
    prior = statistics(rows, lambda row: {})["entropy_bits"]
    full = statistics(rows, lambda row: row["historical_views"]["full_trace"])["entropy_bits"]
    candidates = []

    def add(name, fields, project):
        metrics = statistics(rows, project)
        entropy = metrics["entropy_bits"]
        candidates.append({"name": name, "fields": list(fields), **metrics,
                           "coverage": (prior - entropy) / prior if prior else None,
                           "satisfies_eq20": entropy <= full + epsilon + 1e-12})

    for view in PROJECTION_ORDER:
        add(view, (), lambda row: row["historical_views"][view])
    for count in range(len(ATOMS) + 1):
        for fields in combinations(ATOMS, count):
            add("+".join(fields) or "empty", fields, lambda row: {key: row["atoms"][key] for key in fields})
    feasible = [item for item in candidates if item["satisfies_eq20"]]
    minimum = min(Fraction(item["avg_bytes_exact"]) for item in feasible)
    winners = [item for item in feasible if Fraction(item["avg_bytes_exact"]) == minimum]
    by_fields = {tuple(item["fields"]): item for item in candidates if item["name"] not in PROJECTION_ORDER}
    ablations = {winner["name"]: [{"removed": field, **by_fields[tuple(item for item in winner["fields"] if item != field)]}
                                  for field in winner["fields"]] for winner in winners if winner["fields"]}
    return {"prior_entropy_bits": prior, "full_entropy_bits": full, "epsilon_bits": epsilon,
            "winners": winners, "winner_field_ablations": ablations, "candidates": candidates}


def run(design):
    contexts = design["contexts"]
    if sum(Fraction(item["weight"]) for item in contexts) != 1 or design["epsilon_bits"] < 0:
        raise ValueError("Invalid context weights or epsilon")
    for context in contexts:
        if sum(Fraction(item["weight"]) for item in context["branches"]) != 1:
            raise ValueError(f"Branch weights do not sum to one: {context['id']}")
    rows = [execute(context, branch) for context in contexts for branch in context["branches"]]
    pairs = verify_pairs(rows)
    analysis = evaluate(rows, design["epsilon_bits"])
    by_context = {context["id"]: evaluate([row for row in rows if row["context"] == context["id"]], design["epsilon_bits"])
                  for context in contexts}
    return rows, {"pairs": pairs, "pooled": analysis, "by_context": by_context}


def write_report(path, summary):
    lines = ["# Exp3 日志选择结果", "", "## 轨迹对", "",
             "| 任务 | 分支 | 完全相同的视图 | Oracle 判决 | 区分字段 |", "|---|---|---|---|---|"]
    for pair in summary["pairs"]:
        lines.append(f"| {pair['context']} | {' / '.join(pair['branches'])} | {pair['same_view']} | "
                     f"{pair['verdicts']} | {pair['added_field']} |")
    for name, section in (("合并分布", summary["pooled"]), *summary["by_context"].items()):
        lines.extend(["", f"## {name}", "", "最小候选 `" + "`, `".join(row["name"] for row in section["winners"]) + "`", "",
                      "| 候选 | 条件熵（bit） | Coverage | 最低错误率 | 平均字节 | Eq. 20 |",
                      "|---|---:|---:|---:|---:|---|"])
        displayed = {*PROJECTION_ORDER, "empty", "tool_names", "payload_content", "file_order",
                     *(row["name"] for row in section["winners"]),
                     *(row["name"] for rows in section["winner_field_ablations"].values() for row in rows)}
        for row in section["candidates"]:
            if row["name"] in displayed:
                coverage = "N/A" if row["coverage"] is None else f"{row['coverage']:.1%}"
                lines.append(f"| {row['name']} | {row['entropy_bits']:.4f} | {coverage} | {row['bayes_error']:.1%} | "
                             f"{row['avg_bytes']:.2f} | {'是' if row['satisfies_eq20'] else '否'} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Exercise the full design in memory without saving formal outputs")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    design_bytes = DESIGN.read_bytes()
    traces, summary = run(json.loads(design_bytes))
    print(f"Verified {len(traces)} branches and {len(summary['pairs'])} exact pairs; "
          f"evaluated {len(summary['pooled']['candidates'])} candidates.")
    if not args.preflight:
        args.output.mkdir(parents=True, exist_ok=False)
        summary["design_sha256"] = sha256(design_bytes).hexdigest()
        with (args.output / "support.jsonl").open("wb") as stream:
            for row in traces:
                stream.write(canonical(row) + b"\n")
        (args.output / "log_selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_report(args.output / "report.md", summary)
        print(args.output)
