"""MCP-Bench pilot tools for Experiment 4 live MCP migration."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from toolsim.core.tool_spec import ExecutionContext, ToolExecutionResult, ToolMetadata, ToolSpec


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


class MCPTool(ToolSpec):
    server_id = ""
    mcp_tool_name = ""
    tool_name = ""
    description = ""
    input_schema: dict[str, Any] = {}

    def execute(self, state_or_context: Any, args: dict[str, Any]) -> ToolExecutionResult:
        state = state_or_context.state if isinstance(state_or_context, ExecutionContext) else state_or_context
        backend = getattr(state_or_context, "backend", None)
        clean_args = {k: v for k, v in args.items() if not k.startswith("_")}
        result_id = str(args.get("_result_id") or self.tool_name)

        if backend is not None and hasattr(backend, "call_mcp_tool") and backend.get_backend_name() == "live_mcp":
            api_result = backend.call_mcp_tool(state, self.server_id, self.mcp_tool_name, clean_args)
        else:
            api_result = _local_replica_call(self.server_id, self.mcp_tool_name, clean_args, strict=bool(backend and backend.get_backend_name() == "sandbox"))

        observation = _normalize_observation(api_result["observation"])
        if api_result["success"]:
            state.set_entity("mcp_result", result_id, {
                "result_id": result_id,
                "server_id": self.server_id,
                "tool_name": self.mcp_tool_name,
                "output": observation.get("output"),
                "summary": _summary_value(observation.get("output")),
                "success": True,
            })
        return ToolExecutionResult(
            success=bool(api_result["success"]),
            observation=observation,
            error=api_result["error"],
            state_changed=bool(api_result["success"]),
            metadata={"mcp": True, "server_id": self.server_id, "mcp_tool_name": self.mcp_tool_name},
        )

    @classmethod
    def _metadata(cls) -> ToolMetadata:
        return ToolMetadata(
            name=cls.tool_name,
            version="0.1",
            domain=f"mcp.{cls.server_id}",
            description=cls.description,
            input_schema=cls.input_schema,
            idempotency="idempotent",
        )


class MCPConvertLengthTool(MCPTool):
    server_id = "unit_converter"
    mcp_tool_name = "convert_length"
    tool_name = "mcp.unit.convert_length"
    description = "Call MCP-Bench Unit Converter to convert length units."
    input_schema = _schema({
        "value": {"type": "number"},
        "from_unit": {"type": "string"},
        "to_unit": {"type": "string"},
        "_result_id": {"type": "string"},
    }, ["value", "from_unit", "to_unit"])


MCPConvertLengthTool.metadata = MCPConvertLengthTool._metadata()


class MCPConvertTemperatureTool(MCPTool):
    server_id = "unit_converter"
    mcp_tool_name = "convert_temperature"
    tool_name = "mcp.unit.convert_temperature"
    description = "Call MCP-Bench Unit Converter to convert temperature units."
    input_schema = _schema({
        "value": {"type": "number"},
        "from_unit": {"type": "string"},
        "to_unit": {"type": "string"},
        "_result_id": {"type": "string"},
    }, ["value", "from_unit", "to_unit"])


MCPConvertTemperatureTool.metadata = MCPConvertTemperatureTool._metadata()


class MCPListSupportedUnitsTool(MCPTool):
    server_id = "unit_converter"
    mcp_tool_name = "list_supported_units"
    tool_name = "mcp.unit.list_supported_units"
    description = "List supported units from MCP-Bench Unit Converter."
    input_schema = _schema({
        "unit_type": {"type": ["string", "null"]},
        "_result_id": {"type": "string"},
    }, [])


MCPListSupportedUnitsTool.metadata = MCPListSupportedUnitsTool._metadata()


class MCPTimeConvertTool(MCPTool):
    server_id = "time"
    mcp_tool_name = "convert_time"
    tool_name = "mcp.time.convert_time"
    description = "Call MCP-Bench Time MCP to convert time across IANA time zones."
    input_schema = _schema({
        "source_timezone": {"type": "string"},
        "time": {"type": "string"},
        "target_timezone": {"type": "string"},
        "_result_id": {"type": "string"},
    }, ["source_timezone", "time", "target_timezone"])


MCPTimeConvertTool.metadata = MCPTimeConvertTool._metadata()


class MCPWikipediaSearchTool(MCPTool):
    server_id = "wikipedia"
    mcp_tool_name = "search_wikipedia"
    tool_name = "mcp.wikipedia.search"
    description = "Search Wikipedia through the MCP-Bench Wikipedia server."
    input_schema = _schema({
        "query": {"type": "string"},
        "limit": {"type": "integer"},
        "_result_id": {"type": "string"},
    }, ["query"])


MCPWikipediaSearchTool.metadata = MCPWikipediaSearchTool._metadata()


MCP_TOOLS: dict[str, ToolSpec] = {
    tool.get_metadata().name: tool
    for tool in [
        MCPConvertLengthTool(),
        MCPConvertTemperatureTool(),
        MCPListSupportedUnitsTool(),
        MCPTimeConvertTool(),
        MCPWikipediaSearchTool(),
    ]
}


def _local_replica_call(server_id: str, tool_name: str, args: dict[str, Any], *, strict: bool) -> dict[str, Any]:
    try:
        output = _local_output(server_id, tool_name, args, strict=strict)
    except Exception as exc:
        return {
            "success": False,
            "state_changed": False,
            "observation": {
                "mcp": True,
                "server_id": server_id,
                "tool_name": tool_name,
                "input": dict(args),
                "output": None,
                "exception": str(exc),
            },
            "error": str(exc),
        }
    return {
        "success": True,
        "state_changed": False,
        "observation": {
            "mcp": True,
            "server_id": server_id,
            "tool_name": tool_name,
            "input": dict(args),
            "output": output,
            "exception": None,
        },
        "error": None,
    }


def _local_output(server_id: str, tool_name: str, args: dict[str, Any], *, strict: bool) -> Any:
    if server_id == "unit_converter":
        return _local_unit_converter(tool_name, args, strict=strict)
    if server_id == "time":
        return _local_time(tool_name, args)
    if server_id == "wikipedia":
        return _local_wikipedia(tool_name, args)
    raise ValueError(f"Unsupported MCP server: {server_id}")


def _local_unit_converter(tool_name: str, args: dict[str, Any], *, strict: bool) -> dict[str, Any]:
    if tool_name == "list_supported_units":
        units = {
            "length": ("foot", "meter", "mile", "kilometer"),
            "temperature": ("fahrenheit", "celsius", "kelvin"),
        }
        unit_type = args.get("unit_type")
        return {unit_type: units[unit_type]} if unit_type else units
    if tool_name == "convert_length":
        value = float(args["value"])
        from_unit = str(args["from_unit"]).lower()
        to_unit = str(args["to_unit"]).lower()
        strict_factors = {"foot": 0.3048, "meter": 1.0, "mile": 1609.344, "kilometer": 1000.0}
        lenient_factors = {
            **strict_factors,
            "feet": 0.3048,
            "meters": 1.0,
            "miles": 1609.344,
            "kilometers": 1000.0,
        }
        factors = strict_factors if strict else lenient_factors
        if from_unit not in factors or to_unit not in factors:
            raise ValueError(f"Unsupported length unit: {from_unit}->{to_unit}")
        converted = value * factors.get(from_unit, 1.0) / factors.get(to_unit, 1.0)
        return {
            "original_value": value,
            "original_unit": args["from_unit"],
            "converted_value": converted,
            "converted_unit": args["to_unit"],
            "conversion_type": "length",
        }
    if tool_name == "convert_temperature":
        value = float(args["value"])
        from_unit = str(args["from_unit"]).lower()
        to_unit = str(args["to_unit"]).lower()
        celsius = _to_celsius(value, from_unit)
        converted = _from_celsius(celsius, to_unit)
        return {
            "original_value": value,
            "original_unit": args["from_unit"],
            "converted_value": converted,
            "converted_unit": args["to_unit"],
            "conversion_type": "temperature",
        }
    raise ValueError(f"Unsupported Unit Converter tool: {tool_name}")


def _to_celsius(value: float, unit: str) -> float:
    if unit in {"celsius", "degree celsius"}:
        return value
    if unit in {"fahrenheit", "degree fahrenheit"}:
        return (value - 32.0) * 5.0 / 9.0
    if unit == "kelvin":
        return value - 273.15
    raise ValueError(f"Unsupported temperature unit: {unit}")


def _from_celsius(value: float, unit: str) -> float:
    if unit in {"celsius", "degree celsius"}:
        return value
    if unit in {"fahrenheit", "degree fahrenheit"}:
        return value * 9.0 / 5.0 + 32.0
    if unit == "kelvin":
        return value + 273.15
    raise ValueError(f"Unsupported temperature unit: {unit}")


def _local_time(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool_name != "convert_time":
        raise ValueError(f"Unsupported Time MCP tool: {tool_name}")
    source_tz = str(args["source_timezone"])
    target_tz = str(args["target_timezone"])
    hour, minute = [int(part) for part in str(args["time"]).split(":", 1)]
    now = datetime.now(ZoneInfo(source_tz))
    source = datetime(now.year, now.month, now.day, hour, minute, tzinfo=ZoneInfo(source_tz))
    target = source.astimezone(ZoneInfo(target_tz))
    source_offset = source.utcoffset() or timedelta()
    target_offset = target.utcoffset() or timedelta()
    hours_difference = (target_offset - source_offset).total_seconds() / 3600
    return {
        "source": {"timezone": source_tz, "datetime": source.isoformat(timespec="seconds"), "is_dst": bool(source.dst())},
        "target": {"timezone": target_tz, "datetime": target.isoformat(timespec="seconds"), "is_dst": bool(target.dst())},
        "time_difference": f"{hours_difference:+.1f}h",
    }


def _local_wikipedia(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool_name != "search_wikipedia":
        raise ValueError(f"Unsupported Wikipedia MCP tool: {tool_name}")
    query = str(args.get("query", ""))
    limit = int(args.get("limit", 3))
    return {
        "query": query,
        "results": [
            {"title": query, "snippet": f"Local sandbox placeholder for {query}", "pageid": idx + 1}
            for idx in range(max(0, limit))
        ],
    }


def _normalize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    output = observation.get("output")
    if isinstance(output, str) and output.startswith("Input validation error:"):
        return {**observation, "exception": output}
    return observation


def _summary_value(output: Any) -> Any:
    if isinstance(output, dict):
        if "converted_value" in output:
            return output["converted_value"]
        if "time_difference" in output:
            return output["time_difference"]
        if "results" in output:
            return len(output.get("results") or [])
    if isinstance(output, str):
        return re.sub(r"\s+", " ", output).strip()[:240]
    return output
