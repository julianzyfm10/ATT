"""Select the task-relevant tools for a closed-loop ToolSandbox episode."""

from __future__ import annotations

from typing import Any

from toolsim.core.tool_spec import ToolSpec
from toolsim.runners.llm_agent_runner import tool_registry_for_mode

def _tool_names_from_case(case: Any) -> set[str]:
    payload = case.payload
    names: set[str] = set()
    for key in ["tool_call", "search_call", "action_call", "verify_call"]:
        call = payload.get(key)
        if isinstance(call, dict) and call.get("tool_name"):
            names.add(str(call["tool_name"]))
    for key in ["warmup_calls", "mutation_calls", "oracle_tool_calls"]:
        for call in payload.get(key, []) or []:
            if call.get("tool_name"):
                names.add(str(call["tool_name"]))
    final_tool = payload.get("final_tool")
    if final_tool:
        names.add(str(final_tool))
    return names


def _goal_implied_tool_names(goals: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for goal in goals:
        goal_type = str(goal.get("type", ""))
        entity_type = str(goal.get("entity_type", ""))
        if goal_type in {"query_hits_file", "indexed_contains"}:
            names.update({"file.write", "file.read", "search.index", "search.query"})
        if entity_type == "file":
            names.update({"file.write", "file.read", "search.index", "search.query"})
        if goal_type.startswith("issue_") or entity_type == "issue":
            names.update({"issue.create", "issue.assign", "issue.comment", "issue.close", "issue.reopen"})
        if goal_type.startswith("event_") or entity_type == "event":
            names.update({"calendar.create_event", "calendar.update_event", "calendar.search_events", "calendar.delete_event"})
        if goal_type.startswith("toolsandbox_"):
            names.update(_toolsandbox_goal_tools(goal))
    return names


def _toolsandbox_goal_tools(goal: dict[str, Any]) -> set[str]:
    entity_type = str(goal.get("entity_type", ""))
    if entity_type == "contact":
        return {"add_contact", "modify_contact", "remove_contact", "search_contacts"}
    if entity_type == "reminder":
        return {"add_reminder", "modify_reminder", "remove_reminder", "search_reminder", "datetime_info_to_timestamp"}
    if entity_type == "messaging":
        return {"send_message_with_phone_number", "search_messages"}
    if entity_type == "sandbox":
        return {"end_conversation"}
    if goal.get("type") == "toolsandbox_setting_equals":
        field = str(goal.get("field", ""))
        return {
            "wifi": {"set_wifi_status", "get_wifi_status"},
            "cellular": {"set_cellular_service_status", "get_cellular_service_status"},
            "location_service": {"set_location_service_status", "get_location_service_status", "get_current_location"},
            "low_battery_mode": {"set_low_battery_mode_status", "get_low_battery_mode_status"},
        }.get(field, set())
    return set()


def _domain_companion_tools(tool_names: set[str]) -> set[str]:
    companions: set[str] = set()
    if tool_names & {"file.write", "file.read", "search.index", "search.query"}:
        companions.update({"file.write", "file.read", "search.index", "search.query"})
    if any(name.startswith("issue.") for name in tool_names):
        companions.update({"issue.create", "issue.assign", "issue.comment", "issue.close", "issue.reopen"})
    if any(name.startswith("calendar.") for name in tool_names):
        companions.update({"calendar.create_event", "calendar.update_event", "calendar.search_events", "calendar.delete_event"})
    if tool_names & {"add_contact", "modify_contact", "remove_contact", "search_contacts"}:
        companions.update({"add_contact", "modify_contact", "remove_contact", "search_contacts"})
    if tool_names & {"add_reminder", "modify_reminder", "remove_reminder", "search_reminder", "datetime_info_to_timestamp"}:
        companions.update({"add_reminder", "modify_reminder", "remove_reminder", "search_reminder", "datetime_info_to_timestamp"})
    if tool_names & {"send_message_with_phone_number", "search_messages"}:
        companions.update({"send_message_with_phone_number", "search_messages"})
    if tool_names & {"set_wifi_status", "get_wifi_status", "set_cellular_service_status", "get_cellular_service_status", "set_location_service_status", "get_location_service_status", "set_low_battery_mode_status", "get_low_battery_mode_status", "get_current_location"}:
        companions.update({"set_wifi_status", "get_wifi_status", "set_cellular_service_status", "get_cellular_service_status", "set_location_service_status", "get_location_service_status", "set_low_battery_mode_status", "get_low_battery_mode_status", "get_current_location"})
    if tool_names & {"end_conversation"}:
        companions.add("end_conversation")
    return companions


def policy_noise_tool_registry(case: Any) -> dict[str, ToolSpec]:
    base_registry = tool_registry_for_mode("stateful")
    requested = _tool_names_from_case(case)
    requested.update(_goal_implied_tool_names(case.goals))
    requested.update(_domain_companion_tools(requested))
    filtered = {
        tool_name: base_registry[tool_name]
        for tool_name in sorted(requested)
        if tool_name in base_registry
    }
    return filtered or base_registry
