"""Align generated Core Local cases with the actual Medical MCP entrypoint.

MCP-Bench's Medical Calculator contains an older split-tool module and the
actual ``medcalc.__main__`` server.  The experiment targets the latter.
"""

from __future__ import annotations

import math
from typing import Any


def _step(step_id: str, server: str, tool: str, args: dict[str, Any], depends: list[str] | None = None, condition: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"step_id": step_id, "server_id": server, "tool_name": tool, "args": args, "depends_on": depends or []}
    if condition:
        value["condition"] = condition
    return value


def _ref(step_id: str, path: str = "output") -> dict[str, str]:
    return {"$ref": f"{step_id}.{path}"}


def _goal(source: str, path: str, expected: Any, tolerance: float = 1e-6) -> dict[str, Any]:
    return {"source": source, "path": path, "op": "approx", "expected": expected, "tolerance": tolerance}


def _bmi(weight: float, height_cm: float) -> float:
    return weight / ((height_cm / 100) ** 2)


def _bsa(weight: float, height_cm: float) -> float:
    return math.sqrt(height_cm * weight / 3600)


def _ibw(weight: float, height_inches: float, male: bool) -> tuple[float, float]:
    ideal = (50 if male else 45.5) + 2.3 * (height_inches - 60)
    adjusted = ideal + 0.4 * (weight - ideal) if weight > 1.2 * ideal else ideal
    return ideal, adjusted


def _crcl(age: int, weight: float, height_inches: float, scr: float, male: bool) -> float:
    ideal = (50 if male else 45.5) + 2.3 * (height_inches - 60)
    adjusted = ideal + 0.4 * (weight - ideal)
    patient_bmi = weight / ((height_inches * 0.0254) ** 2)
    selected = weight if patient_bmi < 18.5 else min(ideal, weight) if patient_bmi < 25 else adjusted
    return (140 - age) * selected / (72 * scr) * (1 if male else 0.85)


def _case(cases: list[dict[str, Any]], number: int) -> dict[str, Any]:
    prefix = f"mcpbench_core_local_{number:03d}_"
    return next(case for case in cases if case["case_id"].startswith(prefix))


def _refresh(case: dict[str, Any]) -> None:
    required: list[str] = []
    edges: list[list[str]] = []
    for item in case["oracle_plan"]:
        qualified = f"{item['server_id']}.{item['tool_name']}"
        if not item.get("expected_failure") and qualified not in required:
            required.append(qualified)
        edges.extend([[dependency, item["step_id"]] for dependency in item.get("depends_on", [])])
    case["required_tool_names"] = required
    case["dependency_edges"] = edges


def align_medical_contracts(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    height_170_in = 170 / 2.54
    ideal95, adjusted95 = _ibw(95, height_170_in, True)
    case15 = _case(cases, 15)
    case15["oracle_plan"] = [
        _step("metrics", "medical_calculator", "bmi_bsa_calculator", {"weight": 95, "height": 170}),
        _step("weights", "medical_calculator", "ibw_abw_calculator", {"weight_kg": 95, "height_inches": height_170_in, "male": True}, ["metrics"]),
    ]
    case15["goals"] = [
        _goal("metrics", "output.bmi", round(_bmi(95, 170), 1)),
        _goal("metrics", "output.bsa", round(_bsa(95, 170), 2)),
        _goal("weights", "output.ideal_body_weight_kg", round(ideal95, 1)),
        _goal("weights", "output.adjusted_body_weight_kg", round(adjusted95, 1)),
    ]

    case16 = _case(cases, 16)
    combined_goal = next(goal for goal in case16["goals"] if goal["source"] == "combined")
    combined_goal["path"] = "output.egfr"
    combined_goal["expected"] = round(float(combined_goal["expected"]), 1)

    case19 = _case(cases, 19)
    case19["oracle_plan"][1] = _step("bmi", "medical_calculator", "bmi_bsa_calculator", {"weight": 80, "height": _ref("height", "output.converted_value")}, ["height"])
    bmi_goal = next(goal for goal in case19["goals"] if goal["source"] == "bmi")
    bmi_goal["path"] = "output.bmi"
    bmi_goal["expected"] = round(_bmi(80, 177.8), 1)

    case21 = _case(cases, 21)
    case21["oracle_plan"][1]["args"]["firstNumber"] = _ref("calcium", "output.corrected_calcium")
    calcium_goal = next(goal for goal in case21["goals"] if goal["source"] == "calcium")
    calcium_goal["path"] = "output.corrected_calcium"

    case26 = _case(cases, 26)
    case26["oracle_plan"][2]["args"]["number"] = _ref("combined", "output.egfr")
    combined_goal = next(goal for goal in case26["goals"] if goal["source"] == "combined")
    combined_goal["path"] = "output.egfr"
    combined_goal["expected"] = round(float(combined_goal["expected"]), 1)
    round_goal = next(goal for goal in case26["goals"] if goal["source"] == "round")
    round_goal["expected"] = round(combined_goal["expected"])

    case27 = _case(cases, 27)
    case27["oracle_plan"][3] = _step("bmi", "medical_calculator", "bmi_bsa_calculator", {"weight": _ref("weight", "output.converted_value"), "height": _ref("height", "output.converted_value")}, ["weight", "height"])
    case27["oracle_plan"][4]["args"]["number"] = _ref("bmi", "output.bmi")
    bmi_goal = next(goal for goal in case27["goals"] if goal["source"] == "bmi")
    bmi_goal["path"] = "output.bmi"
    bmi_goal["expected"] = round(_bmi(81.6466266, 172.72), 1)
    next(goal for goal in case27["goals"] if goal["source"] == "round")["expected"] = round(bmi_goal["expected"])

    case28 = _case(cases, 28)
    weight, height = 95.2543977, 177.8
    case28["oracle_plan"] = [
        _step("weight", "unit_converter", "convert_mass", {"value": 210, "from_unit": "pound", "to_unit": "kilogram"}),
        _step("height", "unit_converter", "convert_length", {"value": 70, "from_unit": "inch", "to_unit": "centimeter"}),
        _step("metrics", "medical_calculator", "bmi_bsa_calculator", {"weight": _ref("weight", "output.converted_value"), "height": _ref("height", "output.converted_value")}, ["weight", "height"]),
        _step("crcl", "medical_calculator", "crcl_cockcroft_gault", {"age": 65, "weight": _ref("weight", "output.converted_value"), "height": 70, "scr": 1.4, "sex": "male"}, ["weight"]),
        _step("round_bmi", "math", "round", {"number": _ref("metrics", "output.bmi")}, ["metrics"]),
        _step("round_bsa", "math", "round", {"number": _ref("metrics", "output.bsa")}, ["metrics"]),
        _step("round_crcl", "math", "round", {"number": _ref("crcl", "output.creatinine_clearance")}, ["crcl"]),
    ]
    expected_bmi, expected_bsa, expected_crcl = round(_bmi(weight, height), 1), round(_bsa(weight, height), 2), round(_crcl(65, weight, 70, 1.4, True), 1)
    case28["goals"] = [
        _goal("metrics", "output.bmi", expected_bmi),
        _goal("metrics", "output.bsa", expected_bsa),
        _goal("crcl", "output.creatinine_clearance", expected_crcl),
        {"source": "round_bmi", "path": "output", "op": "numeric_equals", "expected": round(expected_bmi)},
        {"source": "round_bsa", "path": "output", "op": "numeric_equals", "expected": round(expected_bsa)},
        {"source": "round_crcl", "path": "output", "op": "numeric_equals", "expected": round(expected_crcl)},
    ]

    # Expected failures are recovery diagnostics, not final-correctness goals.
    case23 = _case(cases, 23)
    case23["goals"] = [goal for goal in case23["goals"] if goal["source"] != "probe"]

    for number in (15, 16, 19, 21, 23, 26, 27, 28):
        _refresh(_case(cases, number))
    return cases

