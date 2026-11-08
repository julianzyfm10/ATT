from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "build_experiment4_core_local_dataset.py"
DATASET_PATH = ROOT / "data" / "benchmark" / "experiment4_mcpbench_core_local_30.jsonl"
MANIFEST_PATH = ROOT / "data" / "benchmark" / "experiment4_mcpbench_core_local_30_manifest.json"


def _load_builder():
    spec = importlib.util.spec_from_file_location("experiment4_core_local_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _walk_refs(value: Any):
    if isinstance(value, dict):
        if "$ref" in value:
            yield value["$ref"]
        for nested in value.values():
            yield from _walk_refs(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_refs(nested)


def test_dataset_has_exact_scope_and_quotas():
    cases = _load_jsonl(DATASET_PATH)
    assert len(cases) == 30
    assert len({case["case_id"] for case in cases}) == 30
    assert Counter(case["topology"] for case in cases) == {
        "single_server": 18,
        "two_server": 8,
        "three_server": 4,
    }
    assert Counter(case["challenge_type"] for case in cases) == {
        "schema_strictness": 8,
        "sequential_dependency": 8,
        "cross_server_handoff": 6,
        "error_recovery": 4,
        "conditional_recovery": 1,
        "long_horizon": 3,
    }


def test_dataset_is_strictly_offline_and_backend_blind():
    cases = _load_jsonl(DATASET_PATH)
    expected_servers = {
        "unit_converter",
        "time",
        "math",
        "scientific_computing",
        "medical_calculator",
    }
    observed_servers = set()
    for case in cases:
        observed_servers.update(case["servers"])
        assert case["offline"] is True
        assert case["external_network_required"] is False
        assert case["credentials_required"] == []
        assert case["runtime_requirements"]["outbound_network"] is False
        assert case["runtime_requirements"]["backend_blind_prompt"] is True
        assert "http://" not in case["user_prompt"]
        assert "https://" not in case["user_prompt"]
    assert observed_servers == expected_servers


def test_oracle_steps_are_topologically_valid():
    for case in _load_jsonl(DATASET_PATH):
        seen: set[str] = set()
        for item in case["oracle_plan"]:
            step_id = item["step_id"]
            assert step_id not in seen
            assert item["server_id"] in case["servers"]
            assert set(item["depends_on"]).issubset(seen)
            for reference in _walk_refs(item["args"]):
                referenced_step = reference.split(".", 1)[0]
                assert referenced_step in seen, (case["case_id"], step_id, reference)
            seen.add(step_id)
        assert case["goals"]
        assert all(item["source"] in seen for item in case["goals"])


def test_required_tools_match_successful_oracle_steps():
    for case in _load_jsonl(DATASET_PATH):
        expected = []
        for item in case["oracle_plan"]:
            qualified = f"{item['server_id']}.{item['tool_name']}"
            if not item.get("expected_failure") and qualified not in expected:
                expected.append(qualified)
        assert case["required_tool_names"] == expected


def test_generated_artifacts_match_builder_and_manifest_hash():
    builder = _load_builder()
    assert _load_jsonl(DATASET_PATH) == builder.build_cases()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest == builder.build_manifest(builder.build_cases(), DATASET_PATH.read_bytes())
    assert manifest["case_count"] == 30
    assert manifest["server_count"] == 5

