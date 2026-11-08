"""Build the 30-case offline MCP-Bench Core Local dataset for Experiment 4.

The dataset is intentionally backend-blind and network-free.  It uses tools
from five MCP-Bench servers whose implementations are local and deterministic:
Unit Converter, Time MCP, Math MCP, Scientific Computing, and Medical
Calculator.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from core_local_contract_fixes import align_medical_contracts
except ModuleNotFoundError:
    from scripts.core_local_contract_fixes import align_medical_contracts


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "benchmark" / "experiment4_mcpbench_core_local_30.jsonl"
MANIFEST_PATH = ROOT / "data" / "benchmark" / "experiment4_mcpbench_core_local_30_manifest.json"

SERVER_NAMES = {
    "unit_converter": "Unit Converter",
    "time": "Time MCP",
    "math": "Math MCP",
    "scientific_computing": "Scientific Computing",
    "medical_calculator": "Medical Calculator",
}

SOURCE_TASKS = {
    "unit_converter": ["unit_converter_000", "unit_converter_001"],
    "time": ["time_mcp_000", "time_mcp_001"],
    "math": ["math_mcp_000", "math_mcp_001"],
    "scientific_computing": ["scientific_computing_000", "scientific_computing_001"],
    "medical_calculator": ["medical_calculator_000", "medical_calculator_001"],
}


def ref(step_id: str, path: str = "output", transform: str | None = None) -> dict[str, str]:
    value = {"$ref": f"{step_id}.{path}"}
    if transform:
        value["transform"] = transform
    return value


def step(
    step_id: str,
    server_id: str,
    tool_name: str,
    args: dict[str, Any],
    *,
    depends_on: list[str] | None = None,
    expected_failure: bool = False,
    condition: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "step_id": step_id,
        "server_id": server_id,
        "tool_name": tool_name,
        "args": args,
        "depends_on": depends_on or [],
    }
    if expected_failure:
        item["expected_failure"] = True
    if condition:
        item["condition"] = condition
    return item


def goal(
    source: str,
    path: str,
    op: str,
    expected: Any,
    *,
    tolerance: float | None = None,
) -> dict[str, Any]:
    item = {"source": source, "path": path, "op": op, "expected": expected}
    if tolerance is not None:
        item["tolerance"] = tolerance
    return item


def egfr_epi(scr: float, age: int, male: bool) -> float:
    k = 0.9 if male else 0.7
    alpha = -0.302 if male else -0.241
    sex_factor = 1.0 if male else 1.012
    return 142 * (0.9938**age) * (min(scr / k, 1) ** alpha) * (max(scr / k, 1) ** -1.2) * sex_factor


def egfr_epi_cr_cys(scr: float, scys: float, age: int, male: bool) -> float:
    k_cr = 0.9 if male else 0.7
    k_cys = 0.8
    alpha = (-0.144 if scr <= k_cr else -0.544) if male else (-0.219 if scr <= k_cr else -0.544)
    beta = -0.323 if scys <= k_cys else -0.778
    sex_factor = 1.0 if male else 0.963
    return 135 * (scr / k_cr) ** alpha * (scys / k_cys) ** beta * (0.9961**age) * sex_factor


def bmi(weight_kg: float, height_cm: float) -> float:
    return weight_kg / ((height_cm / 100.0) ** 2)


def bsa(weight_kg: float, height_cm: float) -> float:
    return math.sqrt(height_cm * weight_kg / 3600.0)


def ibw(weight_kg: float, height_cm: float, male: bool) -> float:
    del weight_kg
    height_inches = height_cm / 2.54
    base = 50.0 if male else 45.5
    return base + 2.3 * (height_inches - 60.0)


def abw(weight_kg: float, height_cm: float, male: bool) -> float:
    ideal = ibw(weight_kg, height_cm, male)
    return ideal + 0.4 * (weight_kg - ideal) if weight_kg > 1.2 * ideal else ideal


def crcl(age: int, weight_kg: float, height_inches: float, scr: float, male: bool) -> float:
    ideal = (50.0 if male else 45.5) + 2.3 * (height_inches - 60.0)
    adjusted = ideal + 0.4 * (weight_kg - ideal)
    height_m = height_inches * 0.0254
    patient_bmi = weight_kg / (height_m * height_m)
    if patient_bmi < 18.5:
        selected_weight = weight_kg
    elif patient_bmi < 25:
        selected_weight = min(ideal, weight_kg)
    else:
        selected_weight = adjusted
    value = (140 - age) * selected_weight / (72 * scr)
    return value if male else value * 0.85


def make_case(
    number: int,
    slug: str,
    title: str,
    prompt: str,
    servers: list[str],
    difficulty: str,
    challenge_type: str,
    oracle_plan: list[dict[str, Any]],
    goals: list[dict[str, Any]],
    discrimination_axis: str,
    *,
    persistent_session: bool = False,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    topology = {1: "single_server", 2: "two_server", 3: "three_server"}[len(servers)]
    required_tools = []
    for item in oracle_plan:
        qualified = f"{item['server_id']}.{item['tool_name']}"
        if qualified not in required_tools and not item.get("expected_failure"):
            required_tools.append(qualified)
    dependency_edges = [
        [dependency, item["step_id"]]
        for item in oracle_plan
        for dependency in item.get("depends_on", [])
    ]
    source_ids = sorted({source for server in servers for source in SOURCE_TASKS[server]})
    return {
        "case_id": f"mcpbench_core_local_{number:03d}_{slug}",
        "dataset_version": "1.0.0",
        "source": "Accenture/mcp-bench-local-adaptation",
        "source_task_ids": source_ids,
        "title": title,
        "user_prompt": prompt,
        "servers": servers,
        "server_names": [SERVER_NAMES[server] for server in servers],
        "topology": topology,
        "difficulty": difficulty,
        "challenge_type": challenge_type,
        "offline": True,
        "external_network_required": False,
        "credentials_required": [],
        "initial_state": {"tensor_store": {}} if "scientific_computing" in servers else {},
        "oracle_plan": oracle_plan,
        "required_tool_names": required_tools,
        "dependency_edges": dependency_edges,
        "goals": goals,
        "backend_probe": {
            "axis": discrimination_axis,
            "mock": "permissive local semantics and aliases",
            "sandbox": "strict deterministic schema and workflow validation",
            "local_mcp": "real local MCP stdio protocol and server schema",
        },
        "runtime_requirements": {
            "outbound_network": False,
            "persistent_mcp_session": persistent_session,
            "backend_blind_prompt": True,
        },
        "tags": tags or [],
    }


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    add = cases.append

    # 1-8: schema/interface strictness, all single-server.
    add(make_case(1, "length_plural_schema", "Plural length units", "Convert 10 feet to meters and return the numeric result.", ["unit_converter"], "L1", "schema_strictness", [step("convert", "unit_converter", "convert_length", {"value": 10, "from_unit": "foot", "to_unit": "meter"})], [goal("convert", "output.converted_value", "approx", 3.048, tolerance=1e-9)], "natural-language plurals versus canonical enum values", tags=["enum", "unit-alias"]))
    add(make_case(2, "temperature_symbol_schema", "Temperature symbol normalization", "Convert 212°F to °C and return the numeric result.", ["unit_converter"], "L1", "schema_strictness", [step("convert", "unit_converter", "convert_temperature", {"value": 212, "from_unit": "fahrenheit", "to_unit": "celsius"})], [goal("convert", "output.converted_value", "approx", 100.0, tolerance=1e-9)], "symbols and abbreviations versus canonical temperature enums", tags=["enum", "normalization"]))
    add(make_case(3, "mass_plural_schema", "Mass enum normalization", "Convert 5 pounds to kilograms and return the numeric result.", ["unit_converter"], "L1", "schema_strictness", [step("convert", "unit_converter", "convert_mass", {"value": 5, "from_unit": "pound", "to_unit": "kilogram"})], [goal("convert", "output.converted_value", "approx", 2.26796185, tolerance=1e-8)], "plural mass names versus strict singular enums", tags=["enum", "mass"]))
    add(make_case(4, "time_format_schema", "Strict 24-hour time", "Convert 9:00 AM in Asia/Shanghai to UTC.", ["time"], "L1", "schema_strictness", [step("convert", "time", "convert_time", {"source_timezone": "Asia/Shanghai", "time": "09:00", "target_timezone": "UTC"})], [goal("convert", "output.target.datetime", "time_equals", "01:00:00"), goal("convert", "output.time_difference", "equals", "-8.0h")], "informal time spelling versus strict HH:MM input", tags=["time-format", "iana-timezone"]))
    add(make_case(5, "math_parameter_case", "Case-sensitive multiplication schema", "Multiply 7 by 8 using the Math MCP server.", ["math"], "L1", "schema_strictness", [step("multiply", "math", "multiply", {"firstNumber": 7, "SecondNumber": 8})], [goal("multiply", "output", "numeric_equals", 56)], "the real server requires capitalized SecondNumber", tags=["parameter-case", "zod-schema"]))
    add(make_case(6, "math_tool_name", "Exact division tool name", "Divide 81 by 9 using the Math MCP server.", ["math"], "L1", "schema_strictness", [step("divide", "math", "division", {"numerator": 81, "denominator": 9})], [goal("divide", "output", "numeric_equals", 9)], "semantic tool aliases versus the exact division tool name", tags=["tool-selection", "schema"]))
    add(make_case(7, "tensor_flat_values", "Tensor shape and flat-value schema", "Create a 2×2 tensor named schema_A containing rows [1, 2] and [3, 4], then return it.", ["scientific_computing"], "L1", "schema_strictness", [step("create", "scientific_computing", "create_tensor", {"shape": [2, 2], "values": [1, 2, 3, 4], "name": "schema_A"}), step("view", "scientific_computing", "view_tensor", {"name": "schema_A"}, depends_on=["create"])], [goal("view", "output", "array_approx", [[1.0, 2.0], [3.0, 4.0]], tolerance=1e-9)], "nested user representation versus flat values required by create_tensor", persistent_session=True, tags=["stateful", "shape-validation"]))
    egfr1 = egfr_epi(1.2, 60, True)
    add(make_case(8, "medical_boolean_schema", "Boolean sex parameter", "For a 60-year-old male with serum creatinine 1.2 mg/dL, calculate 2021 CKD-EPI eGFR.", ["medical_calculator"], "L1", "schema_strictness", [step("egfr", "medical_calculator", "egfr_epi", {"scr": 1.2, "age": 60, "male": True})], [goal("egfr", "output", "approx", egfr1, tolerance=1e-6)], "natural-language sex value versus the required boolean field", tags=["boolean", "medical-schema"]))

    # 9-16: explicit dependencies, all single-server.
    add(make_case(9, "units_then_furlong", "Inspect units before uncommon conversion", "Inspect supported length units, then convert 1 furlong to meters.", ["unit_converter"], "L2", "sequential_dependency", [step("list", "unit_converter", "list_supported_units", {"unit_type": "length"}), step("convert", "unit_converter", "convert_length", {"value": 1, "from_unit": "furlong", "to_unit": "meter"}, depends_on=["list"])], [goal("convert", "output.converted_value", "approx", 201.168, tolerance=1e-9)], "required list-before-convert workflow", tags=["explicit-dependency", "uncommon-enum"]))
    matrix_a = [4, 7, 2, 6]
    add(make_case(10, "tensor_determinant", "Create then compute determinant", "Store matrix [[4,7],[2,6]] as det_A and return its determinant.", ["scientific_computing"], "L2", "sequential_dependency", [step("create", "scientific_computing", "create_tensor", {"shape": [2, 2], "values": matrix_a, "name": "det_A"}), step("det", "scientific_computing", "determinant", {"name": "det_A"}, depends_on=["create"])], [goal("det", "output", "approx", 10.0, tolerance=1e-9)], "state creation must precede determinant lookup", persistent_session=True, tags=["stateful", "linear-algebra"]))
    add(make_case(11, "tensor_inverse", "Create then invert matrix", "Store matrix [[4,7],[2,6]] as inv_A and return its inverse.", ["scientific_computing"], "L2", "sequential_dependency", [step("create", "scientific_computing", "create_tensor", {"shape": [2, 2], "values": matrix_a, "name": "inv_A"}), step("inverse", "scientific_computing", "matrix_inverse", {"name": "inv_A"}, depends_on=["create"])], [goal("inverse", "output", "array_approx", [[0.6, -0.7], [-0.2, 0.4]], tolerance=1e-8)], "state creation must precede inversion", persistent_session=True, tags=["stateful", "matrix-inverse"]))
    add(make_case(12, "tensor_multiply", "Two-state matrix multiplication", "Create A=[[1,2],[3,4]] and B=[[5,6],[7,8]], then return A×B.", ["scientific_computing"], "L2", "sequential_dependency", [step("create_a", "scientific_computing", "create_tensor", {"shape": [2, 2], "values": [1, 2, 3, 4], "name": "mul_A"}), step("create_b", "scientific_computing", "create_tensor", {"shape": [2, 2], "values": [5, 6, 7, 8], "name": "mul_B"}), step("multiply", "scientific_computing", "multiply_matrices", {"name_a": "mul_A", "name_b": "mul_B"}, depends_on=["create_a", "create_b"])], [goal("multiply", "output", "array_approx", [[19.0, 22.0], [43.0, 50.0]], tolerance=1e-9)], "both named tensors must exist before multiplication", persistent_session=True, tags=["stateful", "two-preconditions"]))
    add(make_case(13, "tensor_eigen", "Create then eigendecompose", "Create diagonal matrix [[2,0],[0,3]] as eig_A and return its eigenvalues.", ["scientific_computing"], "L2", "sequential_dependency", [step("create", "scientific_computing", "create_tensor", {"shape": [2, 2], "values": [2, 0, 0, 3], "name": "eig_A"}), step("eigen", "scientific_computing", "compute_eigen", {"name": "eig_A"}, depends_on=["create"])], [goal("eigen", "output.eigenvalues", "unordered_array_approx", [2.0, 3.0], tolerance=1e-8)], "eigendecomposition consumes a stored tensor name", persistent_session=True, tags=["stateful", "eigenvalues"]))
    add(make_case(14, "sum_then_divide", "Mean through explicit arithmetic chain", "Using only sum and division, compute the mean of [4,8,15,16,23,42].", ["math"], "L2", "sequential_dependency", [step("sum", "math", "sum", {"numbers": [4, 8, 15, 16, 23, 42]}), step("divide", "math", "division", {"numerator": ref("sum"), "denominator": 6}, depends_on=["sum"])], [goal("sum", "output", "numeric_equals", 108), goal("divide", "output", "numeric_equals", 18)], "the sum output must parameterize division", tags=["dataflow", "explicit-tools"]))
    ideal95 = ibw(95, 170, True)
    adjusted95 = abw(95, 170, True)
    add(make_case(15, "bmi_ibw_abw", "Conditional adjusted body weight", "For a 95 kg, 170 cm male, calculate BMI and IBW. If actual weight exceeds 120% of IBW, also calculate ABW.", ["medical_calculator"], "L2", "sequential_dependency", [step("bmi", "medical_calculator", "bmi_calculator", {"weight": 95, "height": 170}), step("ibw", "medical_calculator", "ibw_calculator", {"weight_kg": 95, "height_cm": 170, "male": True}), step("abw", "medical_calculator", "abw_calculator", {"weight_kg": 95, "height_cm": 170, "male": True}, depends_on=["ibw"], condition="95 > 1.2 * ibw.output")], [goal("bmi", "output", "approx", bmi(95, 170), tolerance=1e-6), goal("ibw", "output", "approx", ideal95, tolerance=1e-6), goal("abw", "output", "approx", adjusted95, tolerance=1e-6)], "IBW output controls whether ABW is required", tags=["conditional", "medical"]))
    renal_epi = egfr_epi(1.8, 70, True)
    renal_cys = egfr_epi_cr_cys(1.8, 1.6, 70, True)
    add(make_case(16, "egfr_conditional_cystatin", "Conditional cystatin-C refinement", "For a 70-year-old male with creatinine 1.8 mg/dL and cystatin C 1.6 mg/L, calculate creatinine eGFR; if it is below 60, calculate combined creatinine-cystatin C eGFR.", ["medical_calculator"], "L2", "sequential_dependency", [step("epi", "medical_calculator", "egfr_epi", {"scr": 1.8, "age": 70, "male": True}), step("combined", "medical_calculator", "egfr_epi_cr_cys", {"scr": 1.8, "scys": 1.6, "age": 70, "male": True}, depends_on=["epi"], condition="epi.output < 60")], [goal("epi", "output", "approx", renal_epi, tolerance=1e-6), goal("combined", "output", "approx", renal_cys, tolerance=1e-6)], "numeric threshold activates a second calculator", tags=["conditional", "renal"]))

    # 17-22: cross-server data handoff, all two-server.
    add(make_case(17, "distance_scale", "Convert then scale distance", "Convert 5280 feet to miles, then use Math MCP to multiply the converted value by 3.", ["unit_converter", "math"], "L3", "cross_server_handoff", [step("convert", "unit_converter", "convert_length", {"value": 5280, "from_unit": "foot", "to_unit": "mile"}), step("scale", "math", "multiply", {"firstNumber": ref("convert", "output.converted_value"), "SecondNumber": 3}, depends_on=["convert"])], [goal("scale", "output", "numeric_equals", 3.0)], "Unit Converter output becomes a Math MCP argument", tags=["cross-server", "numeric-ref"]))
    add(make_case(18, "temperature_mean", "Convert temperatures then average", "Convert 32°F and 212°F to Celsius, then use Math MCP to calculate their mean.", ["unit_converter", "math"], "L3", "cross_server_handoff", [step("cold", "unit_converter", "convert_temperature", {"value": 32, "from_unit": "fahrenheit", "to_unit": "celsius"}), step("hot", "unit_converter", "convert_temperature", {"value": 212, "from_unit": "fahrenheit", "to_unit": "celsius"}), step("mean", "math", "mean", {"numbers": [ref("cold", "output.converted_value"), ref("hot", "output.converted_value")]}, depends_on=["cold", "hot"])], [goal("mean", "output", "numeric_equals", 50.0)], "two conversion outputs converge into one statistics call", tags=["cross-server", "fan-in"]))
    height_cm = 177.8
    add(make_case(19, "height_to_bmi", "Convert height then calculate BMI", "A patient weighs 80 kg and is 70 inches tall. Convert height to centimeters, then calculate BMI with Medical Calculator.", ["unit_converter", "medical_calculator"], "L3", "cross_server_handoff", [step("height", "unit_converter", "convert_length", {"value": 70, "from_unit": "inch", "to_unit": "centimeter"}), step("bmi", "medical_calculator", "bmi_calculator", {"weight": 80, "height": ref("height", "output.converted_value")}, depends_on=["height"])], [goal("height", "output.converted_value", "approx", height_cm, tolerance=1e-9), goal("bmi", "output", "approx", bmi(80, height_cm), tolerance=1e-6)], "converted height feeds a calculator with strict centimeter semantics", tags=["cross-server", "medical"]))
    add(make_case(20, "determinant_scale", "Determinant then arithmetic scaling", "Create matrix [[1,2],[3,4]], compute its determinant, then multiply that result by -3 using Math MCP.", ["scientific_computing", "math"], "L3", "cross_server_handoff", [step("create", "scientific_computing", "create_tensor", {"shape": [2, 2], "values": [1, 2, 3, 4], "name": "cross_A"}), step("det", "scientific_computing", "determinant", {"name": "cross_A"}, depends_on=["create"]), step("scale", "math", "multiply", {"firstNumber": ref("det"), "SecondNumber": -3}, depends_on=["det"])], [goal("det", "output", "approx", -2.0, tolerance=1e-9), goal("scale", "output", "numeric_equals", 6.0)], "stateful scientific output crosses into Math MCP", persistent_session=True, tags=["cross-server", "stateful"]))
    add(make_case(21, "calcium_scale", "Correct calcium then scale", "Correct calcium 8.0 mg/dL for albumin 3.0 g/dL, then multiply the corrected value by 10 using Math MCP.", ["medical_calculator", "math"], "L3", "cross_server_handoff", [step("calcium", "medical_calculator", "corrected_calcium", {"serum_calcium": 8.0, "patient_albumin": 3.0, "normal_albumin": 4.0}), step("scale", "math", "multiply", {"firstNumber": ref("calcium"), "SecondNumber": 10}, depends_on=["calcium"])], [goal("calcium", "output", "approx", 8.8, tolerance=1e-9), goal("scale", "output", "numeric_equals", 88.0)], "medical output is reused without re-computation", tags=["cross-server", "medical"]))
    add(make_case(22, "converted_vector", "Conversions populate a tensor", "Convert 1 mile and 2 miles to kilometers, then store the two converted values as vector km_vector and return it.", ["unit_converter", "scientific_computing"], "L3", "cross_server_handoff", [step("one", "unit_converter", "convert_length", {"value": 1, "from_unit": "mile", "to_unit": "kilometer"}), step("two", "unit_converter", "convert_length", {"value": 2, "from_unit": "mile", "to_unit": "kilometer"}), step("create", "scientific_computing", "create_tensor", {"shape": [2], "values": [ref("one", "output.converted_value"), ref("two", "output.converted_value")], "name": "km_vector"}, depends_on=["one", "two"]), step("view", "scientific_computing", "view_tensor", {"name": "km_vector"}, depends_on=["create"])], [goal("view", "output", "array_approx", [1.609344, 3.218688], tolerance=1e-8)], "two external values initialize state in another server", persistent_session=True, tags=["cross-server", "stateful", "fan-in"]))

    # 23-27: conditional/error recovery (2 single, 2 double, 1 triple).
    add(make_case(23, "missing_tensor_recovery", "Recover from missing tensor", "A tensor named recover_A should be [[3,1],[1,2]]. Try its determinant first; if the tensor is absent, create it and retry. Return the determinant.", ["scientific_computing"], "L4", "error_recovery", [step("probe", "scientific_computing", "determinant", {"name": "recover_A"}, expected_failure=True), step("create", "scientific_computing", "create_tensor", {"shape": [2, 2], "values": [3, 1, 1, 2], "name": "recover_A"}, depends_on=["probe"]), step("det", "scientific_computing", "determinant", {"name": "recover_A"}, depends_on=["create"])], [goal("probe", "error", "contains", "not found"), goal("det", "output", "approx", 5.0, tolerance=1e-9)], "observation-driven recovery after a missing-state error", persistent_session=True, tags=["recovery", "stateful", "expected-error"]))
    add(make_case(24, "singular_inverse_recovery", "Recover from singular matrix", "Create S=[[1,2],[2,4]] and attempt to invert it. If singular, replace it with R=[[2,2],[2,5]] and return R's inverse.", ["scientific_computing"], "L4", "error_recovery", [step("create_s", "scientific_computing", "create_tensor", {"shape": [2, 2], "values": [1, 2, 2, 4], "name": "singular_S"}), step("inverse_s", "scientific_computing", "matrix_inverse", {"name": "singular_S"}, depends_on=["create_s"], expected_failure=True), step("create_r", "scientific_computing", "create_tensor", {"shape": [2, 2], "values": [2, 2, 2, 5], "name": "regular_R"}, depends_on=["inverse_s"]), step("inverse_r", "scientific_computing", "matrix_inverse", {"name": "regular_R"}, depends_on=["create_r"])], [goal("inverse_r", "output", "array_approx", [[0.8333333333, -0.3333333333], [-0.3333333333, 0.3333333333]], tolerance=1e-7)], "tool error must trigger a replacement plan", persistent_session=True, tags=["recovery", "singular-matrix"]))
    add(make_case(25, "speed_alias_recovery", "Recover from speed alias mismatch", "Convert 60 mph to meters per second. If the speed unit spelling is rejected, inspect supported speed units and retry; finally round the converted value with Math MCP.", ["unit_converter", "math"], "L4", "error_recovery", [step("list", "unit_converter", "list_supported_units", {"unit_type": "speed"}), step("convert", "unit_converter", "convert_speed", {"value": 60, "from_unit": "miles per hour", "to_unit": "meters per second"}, depends_on=["list"]), step("round", "math", "round", {"number": ref("convert", "output.converted_value")}, depends_on=["convert"])], [goal("convert", "output.converted_value", "approx", 26.8224, tolerance=1e-5), goal("round", "output", "numeric_equals", 27)], "an alias/schema failure should lead to schema inspection and retry", tags=["recovery", "enum", "cross-server"]))
    add(make_case(26, "renal_threshold_round", "Conditional renal refinement and rounding", "For a 70-year-old male with creatinine 1.8 and cystatin C 1.6, calculate creatinine eGFR. If below 60, calculate combined eGFR and round that final value using Math MCP.", ["medical_calculator", "math"], "L4", "conditional_recovery", [step("epi", "medical_calculator", "egfr_epi", {"scr": 1.8, "age": 70, "male": True}), step("combined", "medical_calculator", "egfr_epi_cr_cys", {"scr": 1.8, "scys": 1.6, "age": 70, "male": True}, depends_on=["epi"], condition="epi.output < 60"), step("round", "math", "round", {"number": ref("combined")}, depends_on=["combined"])], [goal("combined", "output", "approx", renal_cys, tolerance=1e-6), goal("round", "output", "numeric_equals", round(renal_cys))], "a threshold selects the value handed to another server", tags=["conditional", "cross-server", "medical"]))
    weight_lb_180 = 81.6466266
    height_in_68 = 172.72
    add(make_case(27, "imperial_bmi_recovery", "Normalize imperial units across three servers", "A patient is 180 lb and 68 in tall. Resolve any unit-name mismatch, convert both values to kg and cm, calculate BMI, then round BMI using Math MCP.", ["unit_converter", "medical_calculator", "math"], "L4", "error_recovery", [step("list_mass", "unit_converter", "list_supported_units", {"unit_type": "mass"}), step("weight", "unit_converter", "convert_mass", {"value": 180, "from_unit": "pound", "to_unit": "kilogram"}, depends_on=["list_mass"]), step("height", "unit_converter", "convert_length", {"value": 68, "from_unit": "inch", "to_unit": "centimeter"}), step("bmi", "medical_calculator", "bmi_calculator", {"weight": ref("weight", "output.converted_value"), "height": ref("height", "output.converted_value")}, depends_on=["weight", "height"]), step("round", "math", "round", {"number": ref("bmi")}, depends_on=["bmi"])], [goal("weight", "output.converted_value", "approx", weight_lb_180, tolerance=1e-7), goal("height", "output.converted_value", "approx", height_in_68, tolerance=1e-8), goal("bmi", "output", "approx", bmi(weight_lb_180, height_in_68), tolerance=1e-6), goal("round", "output", "numeric_equals", round(bmi(weight_lb_180, height_in_68)))], "schema recovery plus a three-server handoff chain", tags=["recovery", "three-server", "medical"]))

    # 28-30: long three-server workflows.
    weight_210 = 95.2543977
    height_70 = 177.8
    add(make_case(28, "cardiometabolic_pipeline", "Imperial-to-clinical calculation pipeline", "A 65-year-old male weighs 210 lb, is 70 in tall, and has serum creatinine 1.4 mg/dL. Convert weight and height, calculate BMI, BSA, and Cockcroft-Gault CrCl, then round all three outputs using Math MCP.", ["unit_converter", "medical_calculator", "math"], "L4", "long_horizon", [step("weight", "unit_converter", "convert_mass", {"value": 210, "from_unit": "pound", "to_unit": "kilogram"}), step("height", "unit_converter", "convert_length", {"value": 70, "from_unit": "inch", "to_unit": "centimeter"}), step("bmi", "medical_calculator", "bmi_calculator", {"weight": ref("weight", "output.converted_value"), "height": ref("height", "output.converted_value")}, depends_on=["weight", "height"]), step("bsa", "medical_calculator", "bsa_calculator", {"weight": ref("weight", "output.converted_value"), "height": ref("height", "output.converted_value")}, depends_on=["weight", "height"]), step("crcl", "medical_calculator", "crcl_cockcroft_gault", {"age": 65, "weight": ref("weight", "output.converted_value"), "height": 70, "scr": 1.4, "sex": "male"}, depends_on=["weight"]), step("round_bmi", "math", "round", {"number": ref("bmi")}, depends_on=["bmi"]), step("round_bsa", "math", "round", {"number": ref("bsa")}, depends_on=["bsa"]), step("round_crcl", "math", "round", {"number": ref("crcl")}, depends_on=["crcl"])], [goal("bmi", "output", "approx", bmi(weight_210, height_70), tolerance=1e-6), goal("bsa", "output", "approx", bsa(weight_210, height_70), tolerance=1e-6), goal("crcl", "output", "approx", crcl(65, weight_210, 70, 1.4, True), tolerance=1e-6), goal("round_bmi", "output", "numeric_equals", round(bmi(weight_210, height_70))), goal("round_bsa", "output", "numeric_equals", round(bsa(weight_210, height_70))), goal("round_crcl", "output", "numeric_equals", round(crcl(65, weight_210, 70, 1.4, True)))], "eight-step three-server dataflow with fan-out and convergence", tags=["long-horizon", "three-server", "medical"]))
    converted_sides = [0.9144, 1.2192, 1.524]
    det_diag = math.prod(converted_sides)
    add(make_case(29, "converted_matrix_pipeline", "Conversions build a scientific matrix", "Convert 3, 4, and 5 feet to meters; create a 3×3 diagonal tensor from those values, compute its determinant, then multiply the determinant by 1000 using Math MCP.", ["unit_converter", "scientific_computing", "math"], "L4", "long_horizon", [step("a", "unit_converter", "convert_length", {"value": 3, "from_unit": "foot", "to_unit": "meter"}), step("b", "unit_converter", "convert_length", {"value": 4, "from_unit": "foot", "to_unit": "meter"}), step("c", "unit_converter", "convert_length", {"value": 5, "from_unit": "foot", "to_unit": "meter"}), step("matrix", "scientific_computing", "create_tensor", {"shape": [3, 3], "values": [ref("a", "output.converted_value"), 0, 0, 0, ref("b", "output.converted_value"), 0, 0, 0, ref("c", "output.converted_value")], "name": "converted_diag"}, depends_on=["a", "b", "c"]), step("det", "scientific_computing", "determinant", {"name": "converted_diag"}, depends_on=["matrix"]), step("scale", "math", "multiply", {"firstNumber": ref("det"), "SecondNumber": 1000}, depends_on=["det"])], [goal("matrix", "output", "array_approx", [[0.9144, 0, 0], [0, 1.2192, 0], [0, 0, 1.524]], tolerance=1e-9), goal("det", "output", "approx", det_diag, tolerance=1e-8), goal("scale", "output", "approx", det_diag * 1000, tolerance=1e-6)], "three independent conversions feed stateful matrix construction", persistent_session=True, tags=["long-horizon", "three-server", "stateful"]))
    add(make_case(30, "meeting_end_pipeline", "Timezone and duration meeting pipeline", "A meeting starts at 09:00 in Asia/Shanghai. Convert the start to Asia/Tokyo. Convert a 150-minute duration to hours, then use Math MCP to add that duration to the Tokyo start hour and return the Tokyo end hour.", ["time", "unit_converter", "math"], "L4", "long_horizon", [step("start", "time", "convert_time", {"source_timezone": "Asia/Shanghai", "time": "09:00", "target_timezone": "Asia/Tokyo"}), step("duration", "unit_converter", "convert_time", {"value": 150, "from_unit": "minutes", "to_unit": "hours"}), step("end", "math", "add", {"firstNumber": ref("start", "output.target.datetime", transform="hour"), "secondNumber": ref("duration", "output.converted_value")}, depends_on=["start", "duration"])], [goal("start", "output.target.datetime", "time_equals", "10:00:00"), goal("start", "output.time_difference", "equals", "+1.0h"), goal("duration", "output.converted_value", "approx", 2.5, tolerance=1e-9), goal("end", "output", "numeric_equals", 12.5)], "time parsing and unit conversion converge in Math MCP", tags=["long-horizon", "three-server", "time"]))

    return align_medical_contracts(cases)


def build_manifest(cases: list[dict[str, Any]], dataset_bytes: bytes) -> dict[str, Any]:
    return {
        "dataset_name": "Experiment 4 MCP-Bench Core Local 30",
        "dataset_version": "1.0.0",
        "case_count": len(cases),
        "source": "Accenture/mcp-bench local server implementations and task metadata",
        "network_policy": "offline-only; outbound network is not required or permitted",
        "comparison_axis": ["mock", "sandbox", "local_mcp"],
        "server_count": len(SERVER_NAMES),
        "servers": SERVER_NAMES,
        "source_task_ids": SOURCE_TASKS,
        "topology_counts": dict(sorted(Counter(case["topology"] for case in cases).items())),
        "difficulty_counts": dict(sorted(Counter(case["difficulty"] for case in cases).items())),
        "challenge_counts": dict(sorted(Counter(case["challenge_type"] for case in cases).items())),
        "persistent_session_case_count": sum(case["runtime_requirements"]["persistent_mcp_session"] for case in cases),
        "sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "reference_syntax": {
            "$ref": "<step_id>.<path>; references must point to an earlier oracle step",
            "transform": "optional deterministic transform such as hour",
        },
        "evaluation_policy": "rule-based goals are primary; LLM judging is not required",
    }


def write_dataset() -> tuple[Path, Path]:
    cases = build_cases()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(case, ensure_ascii=False, sort_keys=True) for case in cases]
    dataset_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    DATASET_PATH.write_bytes(dataset_bytes)
    manifest = build_manifest(cases, dataset_bytes)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return DATASET_PATH, MANIFEST_PATH


if __name__ == "__main__":
    dataset_path, manifest_path = write_dataset()
    print(f"Wrote {dataset_path}")
    print(f"Wrote {manifest_path}")

