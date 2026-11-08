"""Offline MCP-Bench tools used by the Experiment 4 Core Local dataset."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from toolsim.core.tool_spec import ExecutionContext, ToolExecutionResult, ToolMetadata, ToolSpec


@dataclass(frozen=True)
class CoreLocalToolDefinition:
    server_id: str
    tool_name: str
    description: str
    properties: dict[str, dict[str, Any]]
    required: tuple[str, ...]

    @property
    def qualified_name(self) -> str:
        return f"mcp.{self.server_id}.{self.tool_name}"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": self.properties, "required": list(self.required)}


def _props(**items: str) -> dict[str, dict[str, Any]]:
    return {name: {"type": kind} for name, kind in items.items()}


CORE_LOCAL_TOOL_DEFINITIONS = [
    CoreLocalToolDefinition("unit_converter", "convert_length", "Convert length units.", _props(value="number", from_unit="string", to_unit="string"), ("value", "from_unit", "to_unit")),
    CoreLocalToolDefinition("unit_converter", "convert_temperature", "Convert temperature units.", _props(value="number", from_unit="string", to_unit="string"), ("value", "from_unit", "to_unit")),
    CoreLocalToolDefinition("unit_converter", "convert_mass", "Convert mass units.", _props(value="number", from_unit="string", to_unit="string"), ("value", "from_unit", "to_unit")),
    CoreLocalToolDefinition("unit_converter", "convert_speed", "Convert speed units.", _props(value="number", from_unit="string", to_unit="string"), ("value", "from_unit", "to_unit")),
    CoreLocalToolDefinition("unit_converter", "convert_time", "Convert duration units.", _props(value="number", from_unit="string", to_unit="string"), ("value", "from_unit", "to_unit")),
    CoreLocalToolDefinition("unit_converter", "list_supported_units", "List supported units.", {"unit_type": {"type": ["string", "null"]}}, ()),
    CoreLocalToolDefinition("time", "convert_time", "Convert a time between IANA timezones.", _props(source_timezone="string", time="string", target_timezone="string"), ("source_timezone", "time", "target_timezone")),
    CoreLocalToolDefinition("math", "add", "Add two numbers.", _props(firstNumber="number", secondNumber="number"), ("firstNumber", "secondNumber")),
    CoreLocalToolDefinition("math", "multiply", "Multiply two numbers. The second parameter is case-sensitive.", _props(firstNumber="number", SecondNumber="number"), ("firstNumber", "SecondNumber")),
    CoreLocalToolDefinition("math", "division", "Divide numerator by denominator.", _props(numerator="number", denominator="number"), ("numerator", "denominator")),
    CoreLocalToolDefinition("math", "sum", "Sum a non-empty list of numbers.", {"numbers": {"type": "array", "items": {"type": "number"}, "minItems": 1}}, ("numbers",)),
    CoreLocalToolDefinition("math", "mean", "Compute arithmetic mean.", {"numbers": {"type": "array", "items": {"type": "number"}, "minItems": 1}}, ("numbers",)),
    CoreLocalToolDefinition("math", "round", "Round to the nearest integer.", _props(number="number"), ("number",)),
    CoreLocalToolDefinition("scientific_computing", "create_tensor", "Create and store a tensor.", {"shape": {"type": "array", "items": {"type": "integer"}}, "values": {"type": "array", "items": {"type": "number"}}, "name": {"type": "string"}}, ("shape", "values", "name")),
    CoreLocalToolDefinition("scientific_computing", "view_tensor", "Return a stored tensor.", _props(name="string"), ("name",)),
    CoreLocalToolDefinition("scientific_computing", "determinant", "Compute determinant of a stored matrix.", _props(name="string"), ("name",)),
    CoreLocalToolDefinition("scientific_computing", "matrix_inverse", "Invert a stored matrix.", _props(name="string"), ("name",)),
    CoreLocalToolDefinition("scientific_computing", "multiply_matrices", "Multiply two stored matrices.", _props(name_a="string", name_b="string"), ("name_a", "name_b")),
    CoreLocalToolDefinition("scientific_computing", "compute_eigen", "Compute eigenvalues and eigenvectors.", _props(name="string"), ("name",)),
    CoreLocalToolDefinition("medical_calculator", "egfr_epi", "Calculate 2021 CKD-EPI creatinine eGFR.", _props(scr="number", age="integer", male="boolean"), ("scr", "age", "male")),
    CoreLocalToolDefinition("medical_calculator", "egfr_epi_cr_cys", "Calculate combined creatinine-cystatin C eGFR.", _props(scr="number", scys="number", age="integer", male="boolean"), ("scr", "scys", "age", "male")),
    CoreLocalToolDefinition("medical_calculator", "bmi_bsa_calculator", "Calculate BMI and BSA.", {"weight": {"type": "number"}, "height": {"type": "number"}, "height_unit": {"type": "string", "default": "cm"}}, ("weight", "height")),
    CoreLocalToolDefinition("medical_calculator", "ibw_abw_calculator", "Calculate ideal and adjusted body weight.", _props(weight_kg="number", height_inches="number", male="boolean"), ("weight_kg", "height_inches", "male")),
    CoreLocalToolDefinition("medical_calculator", "corrected_calcium", "Correct calcium for albumin.", _props(serum_calcium="number", patient_albumin="number", normal_albumin="number"), ("serum_calcium", "patient_albumin")),
    CoreLocalToolDefinition("medical_calculator", "crcl_cockcroft_gault", "Calculate Cockcroft-Gault creatinine clearance.", _props(age="integer", weight="number", height="number", scr="number", sex="string"), ("age", "weight", "height", "scr", "sex")),
]

DEFINITION_BY_NAME = {item.qualified_name: item for item in CORE_LOCAL_TOOL_DEFINITIONS}


class CoreLocalMCPTool(ToolSpec):
    def __init__(self, definition: CoreLocalToolDefinition) -> None:
        self.definition = definition
        self.tool_name = definition.qualified_name
        self.description = definition.description
        self.input_schema = definition.input_schema
        self.metadata = ToolMetadata(
            name=self.tool_name,
            version="1.0",
            domain=f"mcp.{definition.server_id}",
            description=definition.description,
            input_schema=definition.input_schema,
            idempotency="idempotent",
        )

    def execute(self, state_or_context: Any, args: dict[str, Any]) -> ToolExecutionResult:
        context = state_or_context if isinstance(state_or_context, ExecutionContext) else None
        state = context.state if context is not None else state_or_context
        backend = context.backend if context is not None else None
        backend_name = backend.get_backend_name() if backend is not None else "mock"
        clean_args = {key: value for key, value in args.items() if not key.startswith("_")}
        try:
            if backend is not None and hasattr(backend, "call_core_local_tool"):
                result = backend.call_core_local_tool(
                    state,
                    self.definition.server_id,
                    self.definition.tool_name,
                    clean_args,
                )
            else:
                result = execute_core_local_replica(
                    state,
                    self.definition,
                    clean_args,
                    strict=(backend_name == "sandbox"),
                )
        except Exception as exc:
            result = {"success": False, "output": None, "error": str(exc)}

        output = normalize_core_local_output(result.get("output"))
        success = bool(result.get("success"))
        error = result.get("error")
        sequence = int(state.resources.get("core_local_call_sequence", 0)) + 1
        state.resources["core_local_call_sequence"] = sequence
        call_entity = {
            "sequence": sequence,
            "server_id": self.definition.server_id,
            "tool_name": self.definition.tool_name,
            "qualified_tool_name": self.tool_name,
            "input": clean_args,
            "output": output,
            "success": success,
            "error": error,
            "backend": backend_name,
        }
        state.set_entity("core_local_call", f"call_{sequence:03d}", call_entity)
        observation = {
            "mcp": True,
            "server_id": self.definition.server_id,
            "tool_name": self.definition.tool_name,
            "input": clean_args,
            "output": output,
            "exception": None if success else error,
        }
        return ToolExecutionResult(
            success=success,
            observation=observation,
            error=None if success else str(error),
            state_changed=True,
            metadata={"core_local": True, "server_id": self.definition.server_id},
        )


CORE_LOCAL_MCP_TOOLS: dict[str, ToolSpec] = {
    definition.qualified_name: CoreLocalMCPTool(definition)
    for definition in CORE_LOCAL_TOOL_DEFINITIONS
}


def registry_for_servers(server_ids: list[str]) -> dict[str, ToolSpec]:
    selected = set(server_ids)
    return {
        name: tool
        for name, tool in CORE_LOCAL_MCP_TOOLS.items()
        if tool.definition.server_id in selected  # type: ignore[attr-defined]
    }


def execute_core_local_replica(
    state: Any,
    definition: CoreLocalToolDefinition,
    args: dict[str, Any],
    *,
    strict: bool,
) -> dict[str, Any]:
    normalized = _validate_and_normalize_args(definition, args, strict=strict)
    server = definition.server_id
    if server == "unit_converter":
        output = _unit_call(definition.tool_name, normalized, strict=strict)
    elif server == "time":
        output = _time_call(definition.tool_name, normalized)
    elif server == "math":
        output = _math_call(definition.tool_name, normalized)
    elif server == "scientific_computing":
        output = _scientific_call(state, definition.tool_name, normalized)
    elif server == "medical_calculator":
        output = _medical_call(definition.tool_name, normalized)
    else:
        raise ValueError(f"Unsupported Core Local server: {server}")
    return {"success": True, "output": output, "error": None}


def _validate_and_normalize_args(
    definition: CoreLocalToolDefinition,
    args: dict[str, Any],
    *,
    strict: bool,
) -> dict[str, Any]:
    values = dict(args)
    if not strict:
        aliases = {key.lower(): key for key in definition.properties}
        for key in list(values):
            canonical = aliases.get(key.lower())
            if canonical and canonical not in values:
                values[canonical] = values.pop(key)
        if definition.tool_name == "multiply" and "secondNumber" in values and "SecondNumber" not in values:
            values["SecondNumber"] = values.pop("secondNumber")
    missing = [name for name in definition.required if name not in values]
    if missing:
        raise ValueError(f"Missing required argument(s): {', '.join(missing)}")
    if strict:
        unknown = sorted(set(values) - set(definition.properties))
        if unknown:
            raise ValueError(f"Unexpected argument(s): {', '.join(unknown)}")
    return values


UNIT_LISTS = {
    "length": ["centimeter", "foot", "furlong", "inch", "kilometer", "meter", "mile"],
    "temperature": ["celsius", "fahrenheit", "kelvin"],
    "mass": ["gram", "kilogram", "pound"],
    "speed": ["meters per second", "miles per hour"],
    "time": ["minutes", "hours"],
}

UNIT_ALIASES = {
    "feet": "foot", "meters": "meter", "miles": "mile", "kilometers": "kilometer",
    "inches": "inch", "pounds": "pound", "kilograms": "kilogram", "mph": "miles per hour",
    "m/s": "meters per second", "minute": "minutes", "hour": "hours", "°f": "fahrenheit",
    "f": "fahrenheit", "°c": "celsius", "c": "celsius",
}


def _canonical_unit(value: Any, *, strict: bool) -> str:
    text = str(value).strip()
    lowered = text.lower()
    return lowered if strict else UNIT_ALIASES.get(lowered, lowered)


def _unit_call(tool_name: str, args: dict[str, Any], *, strict: bool) -> Any:
    if tool_name == "list_supported_units":
        unit_type = args.get("unit_type")
        return {unit_type: UNIT_LISTS[unit_type]} if unit_type else UNIT_LISTS
    from_unit = _canonical_unit(args["from_unit"], strict=strict)
    to_unit = _canonical_unit(args["to_unit"], strict=strict)
    value = float(args["value"])
    factors = {
        "convert_length": {"meter": 1.0, "centimeter": 0.01, "foot": 0.3048, "inch": 0.0254, "furlong": 201.168, "mile": 1609.344, "kilometer": 1000.0},
        "convert_mass": {"kilogram": 1.0, "gram": 0.001, "pound": 0.45359237},
        "convert_speed": {"meters per second": 1.0, "miles per hour": 0.44704},
        "convert_time": {"minutes": 60.0, "hours": 3600.0},
    }
    if tool_name == "convert_temperature":
        if from_unit not in UNIT_LISTS["temperature"] or to_unit not in UNIT_LISTS["temperature"]:
            raise ValueError(f"Unsupported temperature unit: {from_unit}->{to_unit}")
        celsius = {"celsius": value, "fahrenheit": (value - 32) * 5 / 9, "kelvin": value - 273.15}[from_unit]
        converted = {"celsius": celsius, "fahrenheit": celsius * 9 / 5 + 32, "kelvin": celsius + 273.15}[to_unit]
        conversion_type = "temperature"
    else:
        tool_factors = factors[tool_name]
        if from_unit not in tool_factors or to_unit not in tool_factors:
            raise ValueError(f"Unsupported unit: {from_unit}->{to_unit}")
        converted = value * tool_factors[from_unit] / tool_factors[to_unit]
        conversion_type = tool_name.removeprefix("convert_")
    return {
        "original_value": value,
        "original_unit": from_unit,
        "converted_value": converted,
        "converted_unit": to_unit,
        "conversion_type": conversion_type,
    }


def _time_call(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool_name != "convert_time":
        raise ValueError(f"Unsupported Time MCP tool: {tool_name}")
    try:
        parsed = datetime.strptime(str(args["time"]), "%H:%M")
    except ValueError as exc:
        raise ValueError("Invalid time format. Expected HH:MM [24-hour format]") from exc
    source_tz = ZoneInfo(str(args["source_timezone"]))
    target_tz = ZoneInfo(str(args["target_timezone"]))
    now = datetime.now(source_tz)
    source = datetime(now.year, now.month, now.day, parsed.hour, parsed.minute, tzinfo=source_tz)
    target = source.astimezone(target_tz)
    difference = (target.utcoffset().total_seconds() - source.utcoffset().total_seconds()) / 3600
    return {
        "source": {"timezone": str(args["source_timezone"]), "datetime": source.isoformat(timespec="seconds"), "is_dst": bool(source.dst())},
        "target": {"timezone": str(args["target_timezone"]), "datetime": target.isoformat(timespec="seconds"), "is_dst": bool(target.dst())},
        "time_difference": f"{difference:+.1f}h" if difference.is_integer() else f"{difference:+g}h",
    }


def _math_call(tool_name: str, args: dict[str, Any]) -> Any:
    if tool_name == "add":
        return float(args["firstNumber"]) + float(args["secondNumber"])
    if tool_name == "multiply":
        return float(args["firstNumber"]) * float(args["SecondNumber"])
    if tool_name == "division":
        denominator = float(args["denominator"])
        if denominator == 0:
            raise ValueError("Division by zero")
        return float(args["numerator"]) / denominator
    numbers = [float(value) for value in args.get("numbers", [])]
    if tool_name == "sum":
        return sum(numbers)
    if tool_name == "mean":
        if not numbers:
            raise ValueError("numbers must not be empty")
        return sum(numbers) / len(numbers)
    if tool_name == "round":
        return math.floor(float(args["number"]) + 0.5)
    raise ValueError(f"Unsupported Math MCP tool: {tool_name}")


def _tensor(state: Any, name: str) -> list[Any]:
    entity = state.get_entity("core_tensor", name)
    if entity is None:
        raise ValueError("The tensor name is not found in the store.")
    return entity["values"]


def _scientific_call(state: Any, tool_name: str, args: dict[str, Any]) -> Any:
    import numpy as np

    if tool_name == "create_tensor":
        shape = [int(value) for value in args["shape"]]
        flat = [float(value) for value in args["values"]]
        if len(flat) != math.prod(shape):
            raise ValueError("Shape does not match number of values.")
        array = np.array(flat).reshape(shape)
        values = array.tolist()
        state.set_entity("core_tensor", str(args["name"]), {"shape": shape, "values": values})
        return values
    if tool_name == "view_tensor":
        return _tensor(state, str(args["name"]))
    if tool_name == "determinant":
        return float(np.linalg.det(np.array(_tensor(state, str(args["name"])), dtype=float)))
    if tool_name == "matrix_inverse":
        return np.linalg.inv(np.array(_tensor(state, str(args["name"])), dtype=float)).tolist()
    if tool_name == "multiply_matrices":
        left = np.array(_tensor(state, str(args["name_a"])), dtype=float)
        right = np.array(_tensor(state, str(args["name_b"])), dtype=float)
        return np.matmul(left, right).tolist()
    if tool_name == "compute_eigen":
        values, vectors = np.linalg.eig(np.array(_tensor(state, str(args["name"])), dtype=float))
        return {"eigenvalues": values.tolist(), "eigenvectors": vectors.tolist()}
    raise ValueError(f"Unsupported Scientific Computing tool: {tool_name}")


def _medical_call(tool_name: str, args: dict[str, Any]) -> Any:
    if tool_name == "egfr_epi":
        scr, age, male = float(args["scr"]), int(args["age"]), bool(args["male"])
        k, alpha = (0.9, -0.302) if male else (0.7, -0.241)
        return 142 * (0.9938**age) * (min(scr / k, 1) ** alpha) * (max(scr / k, 1) ** -1.2) * (1 if male else 1.012)
    if tool_name == "egfr_epi_cr_cys":
        scr, scys, age, male = float(args["scr"]), float(args["scys"]), int(args["age"]), bool(args["male"])
        k_cr, k_cys = (0.9 if male else 0.7), 0.8
        alpha = (-0.144 if scr <= k_cr else -0.544) if male else (-0.219 if scr <= k_cr else -0.544)
        beta = -0.323 if scys <= k_cys else -0.778
        value = 135 * (scr / k_cr) ** alpha * (scys / k_cys) ** beta * (0.9961**age) * (1 if male else 0.963)
        return {"egfr": round(value, 1), "equation": "2021 CKD-EPI Creatinine-Cystatin C"}
    if tool_name == "bmi_bsa_calculator":
        weight, height = float(args["weight"]), float(args["height"])
        value_bmi = weight / ((height / 100) ** 2)
        value_bsa = math.sqrt(height * weight / 3600)
        category = "Underweight" if value_bmi < 18.5 else "Normal weight" if value_bmi < 25 else "Overweight" if value_bmi < 30 else "Obesity class I" if value_bmi < 35 else "Obesity class II" if value_bmi < 40 else "Obesity class III"
        return {"bmi": round(value_bmi, 1), "bmi_classification": category, "bsa": round(value_bsa, 2)}
    if tool_name == "ibw_abw_calculator":
        weight, height, male = float(args["weight_kg"]), float(args["height_inches"]), bool(args["male"])
        ideal = (50 if male else 45.5) + 2.3 * (height - 60)
        adjusted = ideal + 0.4 * (weight - ideal) if weight > ideal * 1.2 else ideal
        return {"actual_weight_kg": round(weight, 1), "ideal_body_weight_kg": round(ideal, 1), "adjusted_body_weight_kg": round(adjusted, 1), "percent_ibw": round(weight / ideal * 100, 1), "obesity_status": "Obese (>120% of IBW)" if weight > ideal * 1.2 else "Non-obese"}
    if tool_name == "corrected_calcium":
        measured, albumin = float(args["serum_calcium"]), float(args["patient_albumin"])
        normal = float(args.get("normal_albumin", 4.0))
        corrected = 0.8 * (normal - albumin) + measured
        return {"measured_calcium": round(measured, 2), "corrected_calcium": round(corrected, 2)}
    if tool_name == "crcl_cockcroft_gault":
        age, weight, height, scr = int(args["age"]), float(args["weight"]), float(args["height"]), float(args["scr"])
        male = str(args["sex"]).lower() == "male"
        ideal = (50 if male else 45.5) + 2.3 * (height - 60)
        adjusted = ideal + 0.4 * (weight - ideal)
        patient_bmi = weight / ((height * 0.0254) ** 2)
        selected = weight if patient_bmi < 18.5 else min(ideal, weight) if patient_bmi < 25 else adjusted
        value = (140 - age) * selected / (72 * scr) * (1 if male else 0.85)
        return {"creatinine_clearance": round(value, 1), "ibw": round(ideal, 1), "abw": round(adjusted, 1)}
    raise ValueError(f"Unsupported Medical Calculator tool: {tool_name}")


def normalize_core_local_output(value: Any) -> Any:
    if isinstance(value, complex):
        return value.real if abs(value.imag) < 1e-12 else {"real": value.real, "imag": value.imag}
    if isinstance(value, str):
        text = value.strip()
        try:
            return normalize_core_local_output(json.loads(text))
        except (json.JSONDecodeError, TypeError):
            numpy_array = _parse_numpy_array_text(text)
            if numpy_array is not None:
                return normalize_core_local_output(numpy_array)
            try:
                number = float(text)
                return int(number) if number.is_integer() else number
            except ValueError:
                return text
    if isinstance(value, dict):
        normalized = {str(key): normalize_core_local_output(item) for key, item in value.items()}
        # FastMCP serializes scalar structured results as {"result": value}.
        # Unwrap that transport-only envelope so all three backends expose the
        # same value and cross-server references remain type compatible.
        if set(normalized) == {"result"}:
            return normalized["result"]
        return normalized
    if isinstance(value, (list, tuple)):
        return [normalize_core_local_output(item) for item in value]
    if hasattr(value, "item"):
        return normalize_core_local_output(value.item())
    if hasattr(value, "tolist"):
        return normalize_core_local_output(value.tolist())
    return value

def _parse_numpy_array_text(text: str) -> list[Any] | None:
    """Parse the compact 1-D/2-D ndarray repr returned by scientific MCP."""
    if not (text.startswith("[") and text.endswith("]")):
        return None

    def parse_row(raw: str) -> list[Any] | None:
        tokens = raw.replace(",", " ").split()
        if not tokens:
            return []
        parsed: list[Any] = []
        try:
            for token in tokens:
                number = complex(token)
                if abs(number.imag) < 1e-12:
                    real = number.real
                    parsed.append(int(real) if real.is_integer() else real)
                else:
                    parsed.append(number)
        except ValueError:
            return None
        return parsed

    inner = text[1:-1].strip()
    if not inner:
        return []
    if not inner.startswith("["):
        return parse_row(inner)

    rows: list[Any] = []
    depth = 0
    start = -1
    for index, character in enumerate(inner):
        if character == "[":
            if depth == 0:
                start = index + 1
            depth += 1
        elif character == "]":
            depth -= 1
            if depth < 0:
                return None
            if depth == 0:
                row = parse_row(inner[start:index])
                if row is None:
                    return None
                rows.append(row)
        elif depth == 0 and not character.isspace() and character != ",":
            return None
    return rows if depth == 0 and rows else None
