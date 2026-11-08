from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "benchmark" / "experiment4_mcpbench_core_local_30.jsonl"

TOOL_ARGUMENTS = {
    ("unit_converter", "convert_length"): {"value", "from_unit", "to_unit"},
    ("unit_converter", "convert_temperature"): {"value", "from_unit", "to_unit"},
    ("unit_converter", "convert_mass"): {"value", "from_unit", "to_unit"},
    ("unit_converter", "convert_speed"): {"value", "from_unit", "to_unit"},
    ("unit_converter", "convert_time"): {"value", "from_unit", "to_unit"},
    ("unit_converter", "list_supported_units"): {"unit_type"},
    ("time", "convert_time"): {"source_timezone", "time", "target_timezone"},
    ("math", "add"): {"firstNumber", "secondNumber"},
    ("math", "multiply"): {"firstNumber", "SecondNumber"},
    ("math", "division"): {"numerator", "denominator"},
    ("math", "sum"): {"numbers"},
    ("math", "mean"): {"numbers"},
    ("math", "round"): {"number"},
    ("scientific_computing", "create_tensor"): {"shape", "values", "name"},
    ("scientific_computing", "view_tensor"): {"name"},
    ("scientific_computing", "determinant"): {"name"},
    ("scientific_computing", "matrix_inverse"): {"name"},
    ("scientific_computing", "multiply_matrices"): {"name_a", "name_b"},
    ("scientific_computing", "compute_eigen"): {"name"},
    ("medical_calculator", "egfr_epi"): {"scr", "age", "male"},
    ("medical_calculator", "egfr_epi_cr_cys"): {"scr", "scys", "age", "male"},
    ("medical_calculator", "bmi_bsa_calculator"): {"weight", "height"},
    ("medical_calculator", "ibw_abw_calculator"): {"weight_kg", "height_inches", "male"},
    ("medical_calculator", "corrected_calcium"): {"serum_calcium", "patient_albumin", "normal_albumin"},
    ("medical_calculator", "crcl_cockcroft_gault"): {"age", "weight", "height", "scr", "sex"},
}


def test_oracle_tools_and_argument_names_match_local_server_sources():
    cases = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    used_tools = set()
    for case in cases:
        for item in case["oracle_plan"]:
            key = (item["server_id"], item["tool_name"])
            used_tools.add(key)
            assert key in TOOL_ARGUMENTS, (case["case_id"], key)
            assert set(item["args"]) == TOOL_ARGUMENTS[key], (case["case_id"], key, item["args"])
    assert used_tools == set(TOOL_ARGUMENTS)

