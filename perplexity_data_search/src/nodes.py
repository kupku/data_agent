"""Узлы LangGraph для пайплайна поиска и синтеза ответа."""

import json
import time
from typing import Any, Callable, Dict, List

from src.config import settings
from src.logger import setup_logger
from src.retrievers import hybrid_search

logger = setup_logger(__name__)


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """Пытается распарсить JSON с fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def decompose_query(state: Dict[str, Any], llm_func: Callable[[str, float], str]) -> Dict[str, Any]:
    """Декомпозирует исходный запрос на подзапросы."""
    prompt = (
        "Ты — аналитик данных. Разбей следующий запрос на 3-5 простых подзапросов для поиска в каталоге данных.\n"
        "Ответ выведи в виде JSON-объекта с ключом sub_queries.\n\n"
        f"Запрос: {state['query']}\n\n"
        "Пример ответа: {\"sub_queries\": [\"таблицы с кредитным скорингом\", \"дата-продукты с образованием\", "
        "\"EDA где упоминается локация\"]}"
    )

    try:
        raw = llm_func(prompt, settings.LLM_TEMPERATURE)
        parsed = _safe_json_loads(raw)
        sub_queries = parsed.get("sub_queries", [])
        if not isinstance(sub_queries, list) or not sub_queries:
            raise ValueError("sub_queries missing")
        sub_queries = [str(item).strip() for item in sub_queries if str(item).strip()]
        if not sub_queries:
            raise ValueError("No valid sub_queries")
    except Exception:
        logger.warning("Failed to parse subqueries, using original query as fallback")
        sub_queries = [state["query"]]

    logger.info("Sub-queries: %s", sub_queries)
    return {"sub_queries": sub_queries}


def extract_filters(state: Dict[str, Any], llm_func: Callable[[str, float], str]) -> Dict[str, Any]:
    """Извлекает домены/теги/сущности для фильтрации."""
    prompt = (
        "Выдели из запроса ключевые сущности и теги для фильтрации дата-продуктов.\n"
        "Формат ответа: JSON-объект с полями domains, tags, entities.\n\n"
        f"Запрос: {state['query']}\n\n"
        "Пример: {\"domains\": [\"кредиты\", \"банки\"], \"tags\": [\"скоринг\", \"физлица\"], "
        "\"entities\": [\"образование\", \"регион\"]}"
    )

    default_filters = {"domains": [], "tags": [], "entities": []}
    try:
        raw = llm_func(prompt, settings.LLM_TEMPERATURE)
        parsed = _safe_json_loads(raw)
        filters = {
            "domains": [str(x) for x in parsed.get("domains", [])],
            "tags": [str(x) for x in parsed.get("tags", [])],
            "entities": [str(x) for x in parsed.get("entities", [])],
        }
    except Exception:
        logger.warning("Failed to parse filters, using empty filters")
        filters = default_filters

    logger.info("Extracted filters: %s", filters)
    return {"filters": filters}


def hybrid_search_node(
    state: Dict[str, Any],
    bm25: Any,
    documents: List[Dict[str, Any]],
    collection: Any,
) -> Dict[str, Any]:
    """Запускает hybrid_search по всем подзапросам и агрегирует кандидатов."""
    sub_queries = state.get("sub_queries") or [state["query"]]
    aggregate: Dict[str, Dict[str, Any]] = {}

    for sub_query in sub_queries:
        candidates = hybrid_search(
            sub_query=sub_query,
            bm25=bm25,
            documents=documents,
            collection=collection,
            k=max(settings.BM25_TOP_K, settings.VECTOR_TOP_K),
            weights=settings.rrf_weights,
        )
        for cand in candidates:
            pid = cand["product_id"]
            if pid not in aggregate:
                aggregate[pid] = {
                    "product_id": pid,
                    "product_name": cand.get("product_name", ""),
                    "rrf_score": 0.0,
                    "docs": [],
                }
            aggregate[pid]["rrf_score"] += cand["rrf_score"]
            aggregate[pid]["docs"].extend(cand.get("docs", []))

    merged = []
    for item in aggregate.values():
        unique_docs = {doc["doc_id"]: doc for doc in item["docs"]}
        item["docs"] = sorted(unique_docs.values(), key=lambda d: d["score"], reverse=True)[:3]
        merged.append(item)

    merged = sorted(merged, key=lambda x: x["rrf_score"], reverse=True)[: settings.FINAL_CANDIDATES_COUNT]

    logger.info("Hybrid search node: sub_queries=%s unique_products=%s", len(sub_queries), len(merged))
    logger.info("Top-3 products: %s", [(c["product_id"], round(c["rrf_score"], 4)) for c in merged[:3]])
    return {"candidates": merged}


def rerank_products(state: Dict[str, Any], reranker: Any, catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Переранжирует кандидаты с помощью cross-encoder."""
    catalog_map = {str(item.get("id")): item for item in catalog}
    candidates = state.get("candidates", [])

    pairs = []
    for cand in candidates:
        product = catalog_map.get(cand["product_id"], {})
        eda_chunks = [doc["text"] for doc in cand.get("docs", []) if doc.get("metadata", {}).get("type") == "eda_chunk"]
        context = (
            f"Name: {product.get('name', cand.get('product_name', ''))}\n"
            f"Description: {product.get('description', '')}\n"
            f"EDA: {' '.join(eda_chunks[:2])}"
        )
        pairs.append((state["query"], context))

    if not pairs:
        return {"candidates": []}

    scores = reranker.predict(pairs)
    before = [(c["product_id"], c["rrf_score"]) for c in candidates]

    for cand, score in zip(candidates, scores):
        cand["rerank_score"] = float(score)

    candidates = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)[: settings.RERANK_TOP_K]
    after = [(c["product_id"], c["rerank_score"]) for c in candidates]

    logger.info("Rerank before=%s", before[:5])
    logger.info("Rerank after=%s", after)
    return {"candidates": candidates}


def synthesize_answer(
    state: Dict[str, Any],
    llm_func: Callable[[str, float], str],
    catalog: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Синтезирует финальный ответ на основе топ-кандидатов."""
    catalog_map = {str(item.get("id")): item for item in catalog}

    products_context: List[str] = []
    for candidate in state.get("candidates", []):
        product = catalog_map.get(candidate["product_id"], {})
        eda_docs = [
            doc["text"]
            for doc in candidate.get("docs", [])
            if doc.get("metadata", {}).get("type") == "eda_chunk"
        ][: settings.MAX_EDA_CHUNKS_IN_CONTEXT]

        block = (
            f"--- Продукт: {product.get('name', candidate.get('product_name', ''))} (ID: {candidate['product_id']}) ---\n"
            f"Описание: {product.get('description', '')}\n"
            f"Релевантные фрагменты EDA:\n" + "\n".join(f"- {chunk}" for chunk in eda_docs) + "\n"
            f"Теги: {', '.join(product.get('tags', []))}\n"
        )
        products_context.append(block)

    prompt = (
        "Ты — эксперт по корпоративным данным. Пользователь задал вопрос:\n"
        f"\"{state['query']}\"\n\n"
        "На основе найденных дата-продуктов дай развёрнутый ответ. Для каждого продукта укажи:\n"
        "- Название и ID.\n"
        "- Почему он подходит (ссылайся на конкретные поля данных и выводы из EDA).\n"
        "- Какие именно атрибуты (колонки) можно использовать для анализа.\n\n"
        "Найденные продукты:\n"
        + "\n".join(products_context)
        + "\nСформулируй ответ в свободной форме, но структурированно. Не генерируй SQL. "
        "Если данных недостаточно, честно скажи об этом и предложи, как уточнить запрос."
    )

    logger.info("Synthesis prompt length: %s", len(prompt))
    start = time.time()
    answer = llm_func(prompt, settings.LLM_TEMPERATURE)
    logger.info("Synthesis completed in %.2f sec", time.time() - start)
    return {"final_answer": answer}
