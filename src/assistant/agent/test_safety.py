"""Additional agent safety regressions; run with pytest on this file."""

from assistant.agent.agent import run_agent
from assistant.agent.tools import validate_arguments
from assistant.infra.llm import LLMResponse, ToolCall
from assistant.infra.testing import FakeLLM


def request(name, **args):
    return LLMResponse("", [ToolCall("call", name, args)])


def test_boolean_quantity_is_not_an_integer_quantity():
    args = {"part_id": "P-1042B", "warehouse": "Казань", "qty": True, "model": "AR-210"}
    assert validate_arguments("create_service_order", args)


def test_batch_cannot_bypass_step_limit():
    calls = []

    def price(**args):
        calls.append(args)
        return {"price_rub": 100}

    llm = FakeLLM([LLMResponse("", [ToolCall(str(i), "get_price", {"part_id": "P-1042B"}) for i in range(5)])])
    result = run_agent("Цена P-1042B?", llm, {"get_price": price}, max_steps=2)
    assert len(result.steps) == 2 and len(calls) == 1 and "прервана" in result.answer


def test_stock_quantity_cannot_be_used_as_a_price():
    llm = FakeLLM([request("check_stock", part_id="P-1042B", warehouse="Казань"), "Цена 3 руб.", "Цена 3 руб."])
    result = run_agent("Есть ли P-1042B в Казани?", llm, {"check_stock": lambda **args: {"qty": 3}})
    assert result.refused and "3 руб" not in result.answer


def test_code_digits_cannot_supply_an_order_quantity():
    calls = []
    args = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 210, "model": "AR-210"}
    result = run_agent(
        "Закажи P-1042B для AR-210 в Казань",
        FakeLLM([request("create_service_order", **args)]),
        {"create_service_order": lambda **kw: calls.append(kw)},
        confirm=True,
    )
    assert not calls and not result.needs_confirmation and "Сколько" in result.answer


def test_confirmed_duplicate_order_is_created_once():
    calls = []
    args = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 2, "model": "AR-210"}

    def order(**kw):
        calls.append(kw)
        return {"order_id": "SO-000001", **kw}

    llm = FakeLLM([request("create_service_order", **args), request("create_service_order", **args), "Готово."])
    result = run_agent("Закажи 2 штуки P-1042B для AR-210, Казань", llm, {"create_service_order": order}, confirm=True)
    assert len(calls) == 1 and "SO-000001" in result.answer


def test_batch_stops_at_unconfirmed_order():
    calls = []
    args = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 2, "model": "AR-210"}
    llm = FakeLLM(
        [
            LLMResponse(
                "",
                [
                    ToolCall("order", "create_service_order", args),
                    ToolCall("price", "get_price", {"part_id": "P-1042B"}),
                ],
            )
        ]
    )
    tools = {"create_service_order": lambda **kw: calls.append(kw), "get_price": lambda **kw: calls.append(kw)}
    result = run_agent("Закажи 2 штуки P-1042B для AR-210, Казань", llm, tools)
    assert result.needs_confirmation and not calls


def test_failed_tool_result_cannot_authorize_a_new_part():
    calls = []
    llm = FakeLLM(
        [
            request("search_knowledge", question="деталь"),
            request("get_price", part_id="P-7777"),
            "Нет данных.",
        ]
    )
    tools = {
        "search_knowledge": lambda **kw: {"error": "P-7777 неизвестна"},
        "get_price": lambda **kw: calls.append(kw),
    }
    result = run_agent("Найди деталь для AR-210", llm, tools)
    assert not calls and result.steps[1].error


def test_false_order_claim_is_corrected_to_confirmation_without_side_effects():
    calls = []
    args = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 2, "model": "AR-210"}
    llm = FakeLLM(["Заказ на 2 штуки оформлен.", request("create_service_order", **args)])
    result = run_agent(
        "Оформи заказ: 2 штуки P-1042B, AR-210, Казань", llm, {"create_service_order": lambda **kw: calls.append(kw)}
    )
    assert result.needs_confirmation and result.pending_action["args"] == args
    assert not calls and "оформлен" not in result.answer
    assert all(value in result.answer for value in ("P-1042B", "количество: 2", "AR-210", "Казань"))


def test_repeated_false_order_claim_is_not_presented_as_success():
    llm = FakeLLM(["Заказ создан.", "Заказ оформлен."])
    result = run_agent("Закажи 2 штуки P-1042B для AR-210, Казань", llm, {})
    assert result.answer.startswith("Заказ не создан")
    assert not result.needs_confirmation


def test_advice_request_cannot_become_an_order_via_a_false_claim():
    llm = FakeLLM(["Заказ оформлен.", "Заказ не создан."])
    result = run_agent("Какая цена P-1042B?", llm, {})
    assert result.answer == "Заказ не создан."
    assert not result.steps
