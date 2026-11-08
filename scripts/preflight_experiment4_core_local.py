"""Preflight the five offline MCP servers used by Experiment 4 Core Local."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from toolsim.mcp_live.persistent_worker import PersistentMCPClient  # noqa: E402


EXPECTED = {
    "unit_converter": {"convert_length", "convert_temperature", "convert_mass", "convert_speed", "convert_time", "list_supported_units"},
    "time": {"convert_time"},
    "math": {"add", "multiply", "division", "sum", "mean", "round"},
    "scientific_computing": {"create_tensor", "view_tensor", "determinant", "matrix_inverse", "multiply_matrices", "compute_eigen"},
    "medical_calculator": {"egfr_epi", "egfr_epi_cr_cys", "bmi_bsa_calculator", "ibw_abw_calculator", "corrected_calcium", "crcl_cockcroft_gault"},
}

SMOKE_CALLS = {
    "unit_converter": ("convert_length", {"value": 10, "from_unit": "foot", "to_unit": "meter"}),
    "time": ("convert_time", {"source_timezone": "Asia/Shanghai", "time": "09:00", "target_timezone": "UTC"}),
    "math": ("multiply", {"firstNumber": 7, "SecondNumber": 8}),
    "scientific_computing": ("create_tensor", {"shape": [2, 2], "values": [1, 2, 3, 4], "name": "preflight_A"}),
    "medical_calculator": ("egfr_epi", {"scr": 1.2, "age": 60, "male": True}),
}


def main() -> None:
    report = {"servers": {}, "all_passed": True}
    with PersistentMCPClient(timeout=90.0) as client:
        for server_id, expected in EXPECTED.items():
            entry = {"initialize": False, "tools_ok": False, "smoke_ok": False, "error": None}
            try:
                tools = client.list_tools(server_id)
                names = {tool["name"] for tool in tools}
                entry["initialize"] = True
                entry["tool_count"] = len(names)
                entry["missing_tools"] = sorted(expected - names)
                entry["tools_ok"] = not entry["missing_tools"]
                tool_name, args = SMOKE_CALLS[server_id]
                result = client.call_tool(server_id, tool_name, args)
                entry["smoke_result"] = result
                entry["smoke_ok"] = not result.get("is_error", False)
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["passed"] = bool(entry["initialize"] and entry["tools_ok"] and entry["smoke_ok"])
            report["servers"][server_id] = entry
            report["all_passed"] = report["all_passed"] and entry["passed"]
            print(f"{server_id}: {'PASS' if entry['passed'] else 'FAIL'}", flush=True)
    output = ROOT / "outputs" / "preflight" / "experiment4_core_local_servers.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()



