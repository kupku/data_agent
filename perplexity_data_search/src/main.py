"""Точка входа интерактивного поиска по каталогу дата-продуктов."""

import pickle
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src.agent_graph import build_app
from src.config import settings
from src.indexer import build_all
from src.logger import setup_logger
from src.utils import load_catalog

logger = setup_logger(__name__)


def _load_indices() -> tuple:
    """Загружает BM25 и Chroma индексы с диска."""
    bm25_path = settings.resolve_path(settings.BM25_PATH)
    chroma_path = settings.resolve_path(settings.CHROMA_PATH)

    with bm25_path.open("rb") as file:
        payload = pickle.load(file)

    client = chromadb.PersistentClient(path=str(chroma_path))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=str(settings.resolve_path(settings.EMBEDDER_PATH))
    )
    collection = client.get_collection(name="data_products", embedding_function=ef)
    return payload["bm25"], payload["documents"], collection


def main() -> None:
    """Запускает интерактивный цикл обработки аналитических запросов."""
    try:
        catalog_path = settings.resolve_path(settings.CATALOG_PATH)
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Каталог не найден: {catalog_path}. Сначала создайте synthetic_catalog.json."
            )

        bm25_path = settings.resolve_path(settings.BM25_PATH)
        chroma_path = settings.resolve_path(settings.CHROMA_PATH)
        if not bm25_path.exists() or not chroma_path.exists():
            logger.info("Indices not found. Starting full indexing...")
            build_all()

        catalog = load_catalog()
        bm25, documents, collection = _load_indices()
        app = build_app(bm25=bm25, documents=documents, collection=collection, catalog=catalog)

        print("Интеллектуальный поиск дата-продуктов. Ctrl+C для выхода.")
        while True:
            query = input("\n>>> ").strip()
            if not query:
                continue

            logger.info("Received query: %s", query)
            result = app.invoke(
                {
                    "query": query,
                    "sub_queries": [],
                    "filters": {},
                    "candidates": [],
                    "final_answer": "",
                }
            )
            print("\n" + result.get("final_answer", "Ответ не сформирован."))
            logger.info("Request processed successfully")

    except KeyboardInterrupt:
        print("\nВыход.")
        logger.info("Interrupted by user")
    except Exception as exc:
        logger.exception("Application failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
