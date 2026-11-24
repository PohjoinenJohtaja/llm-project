"""Build a provenance-preserving directed knowledge graph (FR-2)."""

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Merge facts, count distinct supporting chunks and resolve opposite replacements."""
    graph = nx.DiGraph(conflicts=0)

    def add_entity(entity: Entity, source: str) -> None:
        if entity.node_id not in graph:
            graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[])
        sources = graph.nodes[entity.node_id]["sources"]
        if source not in sources:
            sources.append(source)

    for result in results:
        for entity in result.entities:
            add_entity(entity, result.chunk_id)
        for relation in result.relations:
            if not relation.matches_schema():
                continue
            for entity in (relation.subject, relation.object):
                add_entity(entity, result.chunk_id)
            start, end = relation.subject.node_id, relation.object.node_id
            if not graph.has_edge(start, end):
                graph.add_edge(start, end, type=relation.type, weight=0, sources=[])
            edge = graph.edges[start, end]
            if result.chunk_id not in edge["sources"]:
                edge["sources"].append(result.chunk_id)
                edge["weight"] = len(edge["sources"])

    checked: set[frozenset[str]] = set()
    for start, end, edge in list(graph.edges(data=True)):
        pair = frozenset((start, end))
        if edge["type"] != "REPLACED_BY" or pair in checked:
            continue
        checked.add(pair)
        if start == end or not graph.has_edge(end, start):
            continue
        reverse = graph.edges[end, start]
        if reverse["type"] != "REPLACED_BY":
            continue
        graph.graph["conflicts"] += 1
        forward_weight, reverse_weight = edge["weight"], reverse["weight"]
        if forward_weight <= reverse_weight:
            graph.remove_edge(start, end)
        if reverse_weight <= forward_weight:
            graph.remove_edge(end, start)

    for _, attrs in graph.nodes(data=True):
        attrs["sources"].sort()
    for _, _, attrs in graph.edges(data=True):
        attrs["sources"].sort()
    return graph
