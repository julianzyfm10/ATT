"""Freeze task groups, samples, and label-free Table 7 inputs for Exp1."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent
OUTPUT = HERE / "common_judge"
VIEWS = ("call_only", "action_observation", "outcome_only", "state_aware", "full_trace")
SEED = "ATT-Exp1-common-judge"
DATASETS = {
    "dependency": "data/benchmark/experiment2_toolsandbox_original_100.jsonl",
    "recovery": "outputs/cases/experiment3_llm_deepseek_cases_toolsandbox_original_100_noise_v3.jsonl",
    "backend": "data/benchmark/experiment4_mcpbench_core_local_30.jsonl",
}
FORBIDDEN = {
    "oracle_verdict", "oracle_label", "oracle_components", "full_verdict", "ground_truth",
    "final_correct", "final_state_correct", "terminal_state_correct", "dependency_correct",
    "dependency_compliant", "dependency_satisfied", "recovery_ok", "recovery_detected",
    "recovery_success", "success_at_1", "state_correct", "semantic_observation_correct",
    "original_metadata_for_scoring",
}


def read_rows(path):
    with path.open(encoding="utf-8") as source:
        yield from map(json.loads, source)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def rank(value):
    return sha256(f"{SEED}:{value}".encode()).hexdigest()


def normalize(text):
    return " ".join(text.casefold().split())


def case_key(row):
    source = row["source"]
    return source, row["trajectory_id"].split("::", 4 if source == "dependency" else 5)[-1]


def case_catalog():
    return {(source, case["case_id" if source == "backend" else "case_name"]): case
            for source, path in DATASETS.items() for case in read_rows(ROOT / path)}


def task_groups(cases):
    aliases = defaultdict(set)
    for key, case in cases.items():
        if key[0] == "backend":
            names = [("backend_request", normalize(case["user_prompt"]))]
        else:
            scenario = re.sub(r"(?:_multiple_user_turn|_alt\d*)+$", "", case["scenario_name"])
            names = [("scenario", scenario), ("user_request", normalize(case["canonical_task_key"]))]
        for name in names:
            aliases[name].add(key)
    parent = {key: key for key in cases}

    def root(key):
        while key != parent[key]:
            key = parent[key]
        return key

    for members in aliases.values():
        first, *rest = sorted(members)
        for member in rest:
            parent[root(member)] = root(first)
    return {key: "/".join(root(key)) for key in cases}


def task_context(source, case):
    if source == "backend":
        plan = {step["step_id"]: f"{step['server_id']}.{step['tool_name']}" for step in case["oracle_plan"]}
        return {"task": case["user_prompt"], "dependency_constraints": [
            [plan.get(before, before), plan.get(after, after)] for before, after in case["dependency_edges"]]}
    return {"task": case["user_request"],
            "state_requirements": case["goals" if source == "dependency" else "state_goals"],
            "dependency_constraints": case.get("required_edges", []),
            "required_observation_tool": case.get("semantic_observation_tool")
            if case.get("semantic_observation_required") else None}


def projections(row):
    interactions = [{
        "action": {key: step["action"].get(key) for key in ("tool", "args")},
        "observation": {key: step["observation"].get(key) for key in ("success", "error", "observation")},
    } for step in row["steps"]]
    state_aware = [{**interaction, "state_transition": step["state_transition"]}
                   for interaction, step in zip(interactions, row["steps"], strict=True)]
    return {
        "call_only": {"actions": [interaction["action"] for interaction in interactions]},
        "action_observation": {"interactions": interactions},
        "outcome_only": {"terminal_state": row["terminal_state"]},
        "state_aware": {"interactions": state_aware},
        "full_trace": {"initial_state": row["initial_state"], "interactions": [
            {**interaction, "backend_condition": step["backend_evidence"]}
            for interaction, step in zip(state_aware, row["steps"], strict=True)]},
    }


def check_no_labels(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.casefold() in FORBIDDEN:
                raise ValueError(f"Computed scoring field in judge input: {key}")
            check_no_labels(child)
    elif isinstance(value, list):
        for child in value:
            check_no_labels(child)


def proportional_sample(records, size, salt):
    strata = defaultdict(list)
    for row in records:
        strata[row["source"], row["label"]].append(row["trajectory_id"])
    if size > len(records) or size < len(strata):
        raise ValueError(f"Sample size {size} incompatible with pool of {len(records)}")
    for members in strata.values():
        members.sort(key=lambda item: rank(f"{salt}:{item}"))
    quotas = {key: size * len(members) / len(records) for key, members in strata.items()}
    counts = {key: max(1, int(quota)) for key, quota in quotas.items()}
    while sum(counts.values()) != size:
        if sum(counts.values()) < size:
            key = max((key for key in sorted(strata) if counts[key] < len(strata[key])),
                      key=lambda key: quotas[key] - counts[key])
            counts[key] += 1
        else:
            key = max((key for key in sorted(strata) if counts[key] > 1),
                      key=lambda key: counts[key] - quotas[key])
            counts[key] -= 1
    return sorted(item for key, members in strata.items() for item in members[:counts[key]])


def prepare(output, test_size, development_size, review_size):
    cases = case_catalog()
    groups = task_groups(cases)
    members = defaultdict(set)
    for key, group in groups.items():
        members[key[0]].add(group)
    test_groups = {group for source, choices in members.items()
                   for group in sorted(choices, key=rank)[:max(1, round(len(choices) * 0.30))]}
    labels = {row["trajectory_id"]: row["oracle_verdict"] for row in read_rows(BASE / "oracle_labels.jsonl")}
    traces_path = BASE / "trajectories_clean.jsonl"
    pools = defaultdict(list)
    assignments = {}
    for row in read_rows(traces_path):
        key = case_key(row)
        group = groups[key]
        split = "test" if group in test_groups else "development"
        tid = row["trajectory_id"]
        assignments[tid] = {"group": group, "split": split, "source": key[0]}
        pools[split].append({"trajectory_id": tid, "source": key[0], "label": labels[tid]})
    if len(assignments) != 4640 or set(assignments) != set(labels):
        raise ValueError("Frozen trajectory and label sets differ")
    if any({row["source"] for row in pools[split]} != set(DATASETS) for split in ("development", "test")):
        raise ValueError("Both splits must contain all three sources")
    samples = {"development": proportional_sample(pools["development"], development_size, "development"),
               "test": proportional_sample(pools["test"], test_size, "test")}
    test_set = set(samples["test"])
    samples["review"] = proportional_sample([row for row in pools["test"] if row["trajectory_id"] in test_set],
                                            review_size, "review")
    selected = set(samples["development"]) | test_set
    manifest = {"seed": SEED, "assignments": assignments, "samples": samples, "views": VIEWS,
                "pools": {split: dict(Counter(row["source"] for row in records)) for split, records in pools.items()},
                "task_group_counts": {split: len({entry["group"] for entry in assignments.values() if entry["split"] == split})
                                      for split in pools},
                "sample_strata": {split: dict(Counter(f"{assignments[tid]['source']}/label={labels[tid]}" for tid in ids))
                                  for split, ids in samples.items()}}
    max_bytes = Counter()
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "inputs.jsonl.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for row in read_rows(traces_path):
            tid = row["trajectory_id"]
            if tid not in selected:
                continue
            key = case_key(row)
            context = task_context(key[0], cases[key])
            for view, evidence in projections(row).items():
                payload = {"task_context": context, "execution_evidence": evidence}
                check_no_labels(payload)
                max_bytes[view] = max(max_bytes[view], len(canonical(payload).encode("utf-8")))
                target.write(canonical({"trajectory_id": tid, "view": view, "input": payload}) + "\n")
    temporary.replace(output / "inputs.jsonl")
    manifest["max_input_bytes_by_view"] = dict(max_bytes)
    (output / "manifest.json").write_text(canonical(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"pools": manifest["pools"], "task_group_counts": manifest["task_group_counts"],
                      "sample_strata": manifest["sample_strata"],
                      "max_input_bytes_by_view": dict(max_bytes)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--development-size", type=int, default=40)
    parser.add_argument("--review-size", type=int, default=100)
    args = parser.parse_args()
    prepare(args.output, args.test_size, args.development_size, args.review_size)
