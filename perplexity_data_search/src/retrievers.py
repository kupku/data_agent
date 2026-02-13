"""Гибридные ретриверы: BM25 + vector + RRF."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from src.config import settings
from src.logger import setup_logger
from src.utils import tokenize

logger = setup_logger(__name__)


def bm25_retrieve(query: str, bm25: Any, documents: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Возвращает top-k документов из BM25."""
    scores = bm25.get_scores(tokenize(query))
    top_idx = np.argsort(scores)[::-1][:k]
    results: List[Dict[str, Any]] = []
    for idx in top_idx:
        doc = documents[int(idx)]
        results.append(
            {
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "product_id": doc["metadata"]["product_id"],
                "product_name": doc["metadata"].get("product_name", ""),
                "metadata": doc["metadata"],
                "score": float(scores[int(idx)]),
            }
        )
    return results


def vector_retrieve(
    query: str,
    collection: Any,
    k: int,
    where_filter: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Возвращает top-k документов из Chroma vector search."""
    output = collection.query(
        query_texts=[query],
        n_results=k,
        where=where_filter,
    )

    ids = output.get("ids", [[]])[0]
    docs = output.get("documents", [[]])[0]
    metas = output.get("metadatas", [[]])[0]
    distances = output.get("distances", [[]])[0]

    results: List[Dict[str, Any]] = []
    for doc_id, text, meta, dist in zip(ids, docs, metas, distances):
        results.append(
            {
                "doc_id": doc_id,
                "text": text,
                "product_id": meta["product_id"],
                "product_name": meta.get("product_name", ""),
                "metadata": meta,
                "score": float(1 - dist),
            }
        )
    return results


def hybrid_search(
    sub_query: str,
    bm25: Any,
    documents: List[Dict[str, Any]],
    collection: Any,
    k: int,
    weights: Tuple[float, float],
) -> List[Dict[str, Any]]:
    """Выполняет гибридный поиск с Reciprocal Rank Fusion."""
    bm25_results = bm25_retrieve(sub_query, bm25, documents, k)
    vector_results = vector_retrieve(sub_query, collection, k)

    product_scores: Dict[str, float] = defaultdict(float)
    product_docs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    product_names: Dict[str, str] = {}

    for rank, result in enumerate(bm25_results, start=1):
        pid = result["product_id"]
        product_scores[pid] += weights[0] * (1 / (rank + settings.RRF_K))
        product_docs[pid].append(result)
        product_names[pid] = result.get("product_name", "")

    for rank, result in enumerate(vector_results, start=1):
        pid = result["product_id"]
        product_scores[pid] += weights[1] * (1 / (rank + settings.RRF_K))
        product_docs[pid].append(result)
        product_names[pid] = result.get("product_name", "")

    candidates: List[Dict[str, Any]] = []
    for pid, score in sorted(product_scores.items(), key=lambda item: item[1], reverse=True):
        docs = sorted(product_docs[pid], key=lambda d: d["score"], reverse=True)[:3]
        candidates.append(
            {
                "product_id": pid,
                "product_name": product_names.get(pid, ""),
                "rrf_score": float(score),
                "docs": docs,
            }
        )

    if candidates:
        logger.info(
            "Hybrid search for sub_query='%s': products=%s top1=%s",
            sub_query,
            len(candidates),
            candidates[0]["product_id"],
        )
    else:
        logger.info("Hybrid search for sub_query='%s': no products found", sub_query)

    return candidates
