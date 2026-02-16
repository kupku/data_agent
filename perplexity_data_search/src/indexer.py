"""Построение BM25 и Chroma индексов для каталога дата-продуктов."""

import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from src.config import settings
from src.logger import setup_logger
from src.utils import chunk_text, load_catalog, tokenize

logger = setup_logger(__name__)


def create_documents_from_catalog(catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Создаёт индексируемые документы из каталога."""
    documents: List[Dict[str, Any]] = []
    for product in catalog:
        product_id = str(product.get("id", "unknown"))
        product_name = product.get("name", "")
        description = product.get("description", "")
        tags = product.get("tags", []) or []

        documents.append(
            {
                "doc_id": f"{product_id}_meta",
                "text": f"{product_name} {description} {' '.join(tags)}",
                "metadata": {
                    "type": "metadata",
                    "product_id": product_id,
                    "product_name": product_name,
                },
            }
        )

        for idx, chunk in enumerate(chunk_text(product.get("eda", ""))):
            documents.append(
                {
                    "doc_id": f"{product_id}_chunk_{idx}",
                    "text": chunk,
                    "metadata": {
                        "type": "eda_chunk",
                        "product_id": product_id,
                        "product_name": product_name,
                        "chunk_index": idx,
                    },
                }
            )

        for q_idx, question in enumerate(product.get("generated_questions", []) or []):
            documents.append(
                {
                    "doc_id": f"{product_id}_q_{q_idx}",
                    "text": question,
                    "metadata": {
                        "type": "generated_question",
                        "product_id": product_id,
                        "product_name": product_name,
                    },
                }
            )

    logger.info("Created %s documents from %s products", len(documents), len(catalog))
    return documents


def build_bm25_index(documents: List[Dict[str, Any]]) -> None:
    """Строит и сохраняет BM25 индекс вместе с документами."""
    corpus = [doc["text"] for doc in documents]
    tokenized_corpus = [tokenize(text) for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_path = settings.resolve_path(settings.BM25_PATH)
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with bm25_path.open("wb") as file:
        pickle.dump({"bm25": bm25, "documents": documents}, file)

    logger.info("BM25 index saved to %s", bm25_path)


def build_chroma_index(documents: List[Dict[str, Any]]) -> None:
    """Строит и сохраняет Chroma векторный индекс."""
    chroma_path = settings.resolve_path(settings.CHROMA_PATH)
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=str(settings.resolve_path(settings.EMBEDDER_PATH))
    )

    existing = {col.name for col in client.list_collections()}
    if "data_products" in existing:
        client.delete_collection("data_products")

    collection = client.create_collection(name="data_products", embedding_function=ef)

    batch_size = settings.CHROMA_BATCH_SIZE
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        collection.add(
            ids=[doc["doc_id"] for doc in batch],
            documents=[doc["text"] for doc in batch],
            metadatas=[doc["metadata"] for doc in batch],
        )
    logger.info("Chroma index saved to %s. Added %s vectors", chroma_path, len(documents))


def build_all() -> None:
    """Полный пайплайн индексации каталога."""
    start = time.time()
    catalog_path = settings.resolve_path(settings.CATALOG_PATH)
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {catalog_path}. Create synthetic_catalog.json before indexing."
        )

    catalog = load_catalog()
    documents = create_documents_from_catalog(catalog)
    build_bm25_index(documents)
    build_chroma_index(documents)

    elapsed = time.time() - start
    logger.info("All indices built in %.2f seconds", elapsed)


if __name__ == "__main__":
    build_all()
