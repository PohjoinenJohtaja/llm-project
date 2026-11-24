"""Validated knowledge and ERP tools (FR-6)."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
_PART = {"type": "string", "pattern": r"^P-\d{4}[A-Z]?$"}
_WAREHOUSE = {"type": "string", "enum": WAREHOUSES}
_MODEL = {"type": "string", "pattern": r"^(AR|BZ|GS)-\d{3}$"}


def _schema(name: str, description: str, properties: dict) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties), "additionalProperties": False},
    }}


TOOL_SCHEMAS: list[dict] = [
    _schema("search_knowledge", "Найти факты о моделях, ошибках и замене деталей с источниками.",
            {"question": {"type": "string"}}),
    _schema("summarize_topic", "Ответить на обзорный вопрос по сводкам сообществ графа.",
            {"question": {"type": "string"}}),
    _schema("check_stock", "Проверить остаток детали на складе через ERP.",
            {"part_id": _PART, "warehouse": _WAREHOUSE}),
    _schema("get_price", "Получить текущую цену детали из ERP.", {"part_id": _PART}),
    _schema("create_service_order", "Создать заказ только после подтверждения человеком.",
            {"part_id": _PART, "warehouse": _WAREHOUSE, "qty": {"type": "integer", "minimum": 1},
             "model": _MODEL}),
]
_PARAMETERS = {item["function"]["name"]: item["function"]["parameters"] for item in TOOL_SCHEMAS}


def validate_arguments(name: str, arguments: dict) -> str | None:
    """Use the same strict validation for direct tool calls and agent calls."""
    if name not in _PARAMETERS:
        return "Неизвестный инструмент."
    if not isinstance(arguments, dict):
        return "Аргументы должны быть объектом JSON."
    properties = _PARAMETERS[name]["properties"]
    if set(arguments) != set(properties):
        return "Неверный набор аргументов инструмента."
    for key, spec in properties.items():
        value = arguments[key]
        if spec["type"] == "integer":
            if type(value) is not int or value < spec.get("minimum", 1):
                return f"{key}: требуется положительное целое число."
        elif not isinstance(value, str) or not value.strip():
            return f"{key}: требуется непустая строка."
        elif "enum" in spec and value not in spec["enum"]:
            return f"{key}: неизвестное значение."
        elif "pattern" in spec and not re.fullmatch(spec["pattern"], value):
            return f"{key}: неверный формат кода."
    return None


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Keep missing data, rejected requests and unavailable ERP distinguishable."""
    def knowledge(question: str) -> dict:
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def topic(question: str) -> dict:
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    functions = {"search_knowledge": knowledge, "summarize_topic": topic,
                 "check_stock": erp.get_stock, "get_price": erp.get_price,
                 "create_service_order": erp.create_order}

    def wrap(name: str, function: Callable) -> Callable[..., dict]:
        def execute(**arguments) -> dict:
            error = validate_arguments(name, arguments)
            if error:
                return {"error": error}
            try:
                return function(**arguments)
            except ErpNotFound as exc:
                return {"error": f"Данных в ERP нет: {exc}", "error_type": "not_found"}
            except ErpRejected as exc:
                return {"error": f"ERP отклонила запрос: {exc}", "error_type": "rejected"}
        return execute

    return {name: wrap(name, function) for name, function in functions.items()}
