"""Regressions for complete inventory extraction used by ERP lookups."""

import json

from assistant.graphrag.extraction import extract_chunk
from assistant.graphrag.graph_builder import build_graph
from assistant.infra.schemas import Chunk
from assistant.infra.testing import FakeLLM


def response(parts, extra=()):
    relations = [
        {"subject_type": "Model", "subject": "AR-210", "type": "HAS_PART", "object_type": "Part", "object": part}
        for part in parts
    ]
    relations.append(
        {
            "subject_type": "Model",
            "subject": "AR-210",
            "type": "BELONGS_TO",
            "object_type": "ProductLine",
            "object": "Арктика",
        }
    )
    return json.dumps({"entities": [{"type": "Part", "name": p} for p in extra], "relations": relations})


def test_inventory_batches_preserve_all_rows_sources_and_distinct_chunk_weights():
    parts = ["P-9001", "P-9002", "P-9003", "P-9004", "P-9005", "P-9006", "P-9007"]
    chunk = Chunk(
        "inventory.md#0",
        "inventory.md",
        "# AR-210\nЛинейка Арктика\n" + "\n".join(f"- компрессор — деталь {p}" for p in parts),
    )
    llm = FakeLLM([response(parts[:3]), response(parts[3:6]), response(parts[6:])])
    result = extract_chunk(chunk, llm)
    graph = build_graph([result])
    assert {r.object.name for r in result.relations if r.type == "HAS_PART"} == set(parts)
    assert result.chunk_id == chunk.chunk_id and result.doc_id == chunk.doc_id
    assert graph.edges["Model:AR-210", "ProductLine:Арктика"]["weight"] == 1
    assert graph.edges["Model:AR-210", "ProductLine:Арктика"]["sources"] == [chunk.chunk_id]
    for request in llm.requests:
        assert "# AR-210" in request["messages"][1]["content"]
        assert "Линейка Арктика" in request["messages"][1]["content"]


def test_inventory_batch_cannot_accept_code_from_a_different_batch():
    parts = ["P-9001", "P-9002", "P-9003", "P-9004"]
    chunk = Chunk(
        "inventory.md#0", "inventory.md", "# AR-210\n" + "\n".join(f"- компрессор — деталь {p}" for p in parts)
    )
    llm = FakeLLM([response(parts[:3], extra=[parts[3]]), response(parts[3:])])
    result = extract_chunk(chunk, llm)
    assert result.dropped == 1
    assert {r.object.name for r in result.relations if r.type == "HAS_PART"} == set(parts)


def test_twice_malformed_inventory_batch_discards_partial_chunk():
    parts = ["P-9001", "P-9002", "P-9003", "P-9004"]
    chunk = Chunk(
        "inventory.md#0", "inventory.md", "# AR-210\n" + "\n".join(f"- компрессор — деталь {p}" for p in parts)
    )
    llm = FakeLLM([response(parts[:3]), "broken", "still broken"])
    result = extract_chunk(chunk, llm)
    assert result.entities == [] and result.relations == []
    assert len(llm.requests) == 3


def test_inventory_wrong_direction_is_retried_then_validated():
    parts = ["P-9001", "P-9002", "P-9003", "P-9004"]
    chunk = Chunk(
        "inventory.md#0", "inventory.md", "# AR-210\n" + "\n".join(f"- компрессор — деталь {p}" for p in parts)
    )
    wrong = json.loads(response(parts[:3]))
    wrong["relations"][0] = {
        "subject_type": "Part",
        "subject": "P-9001",
        "type": "HAS_PART",
        "object_type": "Model",
        "object": "AR-210",
    }
    llm = FakeLLM([json.dumps(wrong), response(parts[:3]), response(parts[3:])])
    result = extract_chunk(chunk, llm)
    assert {r.object.name for r in result.relations if r.type == "HAS_PART"} == set(parts)
    assert result.dropped == 1
    assert len(llm.requests) == 3
    assert "Model -> Part" in llm.requests[1]["messages"][-1]["content"]


def test_inventory_second_wrong_direction_is_dropped_without_more_retries():
    parts = ["P-9001", "P-9002", "P-9003", "P-9004"]
    chunk = Chunk(
        "inventory.md#0", "inventory.md", "# AR-210\n" + "\n".join(f"- компрессор — деталь {p}" for p in parts)
    )
    wrong = json.loads(response(parts[:3]))
    wrong["relations"][0] = {
        "subject_type": "Part",
        "subject": "P-9001",
        "type": "HAS_PART",
        "object_type": "Model",
        "object": "AR-210",
    }
    llm = FakeLLM([json.dumps(wrong), json.dumps(wrong), response(parts[3:])])
    result = extract_chunk(chunk, llm)
    assert all(r.matches_schema() for r in result.relations)
    assert "P-9001" not in {r.object.name for r in result.relations if r.type == "HAS_PART"}
    assert len(llm.requests) == 3


def test_inventory_missing_model_links_get_one_completeness_retry():
    parts = ["P-9001", "P-9002", "P-9003", "P-9004"]
    chunk = Chunk(
        "inventory.md#0",
        "inventory.md",
        "# AR-210\n## Состав узлов\n" + "\n".join(f"- компрессор — деталь {p}" for p in parts),
    )
    llm = FakeLLM([response(parts[:1]), response(parts[:3]), response(parts[3:])])
    result = extract_chunk(chunk, llm)
    assert {r.object.name for r in result.relations if r.type == "HAS_PART"} == set(parts)
    assert len(llm.requests) == 3
    assert "P-9002" in llm.requests[1]["messages"][-1]["content"]
    assert "P-9004" not in llm.requests[1]["messages"][-1]["content"]
