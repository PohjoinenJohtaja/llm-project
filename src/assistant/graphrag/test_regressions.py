"""Regressions for abstention and community boundary evidence."""

import networkx as nx

from assistant.agent.agent import run_agent
from assistant.graphrag.communities import summarize_community
from assistant.graphrag.search import global_search, local_search
from assistant.infra.llm import LLMResponse, ToolCall
from assistant.infra.schemas import Community, KnowledgeBase
from assistant.infra.testing import FakeLLM


def knowledge():
    graph = nx.DiGraph()
    for kind, name in [("Model", "AR-100"), ("Component", "компрессор"), ("ProductLine", "Арктика")]:
        graph.add_node(f"{kind}:{name}", type=kind, name=name, sources=["report.md#0"])
    graph.add_edge("Model:AR-100", "Component:компрессор", type="FAILED", weight=7, sources=["report.md#0"])
    graph.add_edge("Model:AR-100", "ProductLine:Арктика", type="BELONGS_TO", weight=1, sources=["manual.md#0"])
    return KnowledgeBase(graph, [Community(0, list(graph), "Отказ компрессора AR-100.")], {})


def test_local_model_abstention_has_refused_flag():
    answer = local_search("Есть данные о AR-100?", knowledge(), FakeLLM(["В базе знаний не найдено данных."]))
    assert answer.refused and answer.sources


def test_global_drops_high_scoring_abstention_without_reduce():
    llm = FakeLLM(['{"answer":"В базе знаний не найдено данных", "score":8}'])
    assert global_search("Расскажи об отказах", knowledge(), llm).refused
    assert len(llm.requests) == 1


def test_global_reduce_abstention_has_refused_flag():
    llm = FakeLLM(['{"answer":"Компрессор", "score":8}', "Недостаточно данных."])
    assert global_search("Какие отказы?", knowledge(), llm).refused


def test_global_unknown_code_never_calls_model():
    llm = FakeLLM([])
    assert global_search("Что произошло с AR-999?", knowledge(), llm).refused
    assert not llm.requests


def test_summary_counts_boundary_failures_for_owned_model():
    kb = knowledge()
    llm = FakeLLM(["Сводка"])
    summarize_community(Community(0, ["Model:AR-100"]), kb.graph, llm)
    import json

    data = json.loads(llm.requests[0]["messages"][1]["content"])
    assert data["failures_by_line"] == {"Арктика": {"компрессор": 7}}
    assert data["failures_by_model"] == {"AR-100": 7}
    other = FakeLLM(["Сводка"])
    summarize_community(Community(1, ["Component:компрессор"]), kb.graph, other)
    assert json.loads(other.requests[0]["messages"][1]["content"])["failures_by_model"] == {}


def test_missing_erp_record_is_refusal_not_zero():
    llm = FakeLLM([LLMResponse("", [ToolCall("a", "get_price", {"part_id": "P-9999"})]), "Деталь отсутствует."])
    result = run_agent("Цена P-9999?", llm, {"get_price": lambda **kw: {"error": "нет", "error_type": "not_found"}})
    assert result.refused and "в базе знаний не найдено" in result.answer.lower()
