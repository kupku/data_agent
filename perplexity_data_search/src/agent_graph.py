"""Сборка LangGraph приложения для интеллектуального поиска."""

from functools import partial
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from src.nodes import (
    decompose_query,
    extract_filters,
    hybrid_search_node,
    rerank_products,
    synthesize_answer,
)
from src.utils import get_reranker, gigachat_complete


class AgentState(TypedDict):
    """Состояние графа поиска."""

    query: str
    sub_queries: List[str]
    filters: Dict[str, List[str]]
    candidates: List[Dict[str, Any]]
    final_answer: str


def build_app(bm25: Any, documents: List[Dict[str, Any]], collection: Any, catalog: List[Dict[str, Any]]) -> Any:
    """Собирает и компилирует LangGraph пайплайн."""
    reranker = get_reranker()

    graph = StateGraph(AgentState)
    graph.add_node("decompose", partial(decompose_query, llm_func=gigachat_complete))
    graph.add_node("extract_filters", partial(extract_filters, llm_func=gigachat_complete))
    graph.add_node(
        "hybrid_search",
        partial(hybrid_search_node, bm25=bm25, documents=documents, collection=collection),
    )
    graph.add_node("rerank", partial(rerank_products, reranker=reranker, catalog=catalog))
    graph.add_node("synthesize", partial(synthesize_answer, llm_func=gigachat_complete, catalog=catalog))

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "extract_filters")
    graph.add_edge("extract_filters", "hybrid_search")
    graph.add_edge("hybrid_search", "rerank")
    graph.add_edge("rerank", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
