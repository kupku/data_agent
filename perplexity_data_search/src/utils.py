"""Общие утилиты проекта: LLM, модели, токенизация, загрузка данных."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from gigachat import GigaChat
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings
from src.logger import setup_logger

logger = setup_logger(__name__)

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_gigachat_client: GigaChat | None = None


def _build_gigachat_client() -> GigaChat:
    """Создаёт экземпляр клиента GigaChat на основе настроек."""
    kwargs: Dict[str, Any] = {
        "credentials": settings.GIGACHAT_CREDENTIALS,
        "scope": settings.GIGACHAT_SCOPE,
        "verify_ssl_certs": settings.GIGACHAT_VERIFY_SSL,
    }
    if settings.GIGACHAT_CA_CERT_PATH:
        kwargs["ca_bundle_file"] = settings.GIGACHAT_CA_CERT_PATH
    if settings.GIGACHAT_CERT_FILE:
        kwargs["cert_file"] = settings.GIGACHAT_CERT_FILE
    if settings.GIGACHAT_KEY_FILE:
        kwargs["key_file"] = settings.GIGACHAT_KEY_FILE

    return GigaChat(**kwargs)


def get_gigachat_client() -> GigaChat:
    """Возвращает singleton-клиент GigaChat."""
    global _gigachat_client
    if _gigachat_client is None:
        logger.info("Initializing GigaChat client")
        _gigachat_client = _build_gigachat_client()
    return _gigachat_client


@retry(
    stop=stop_after_attempt(settings.GIGACHAT_RETRY_ATTEMPTS),
    wait=wait_exponential(min=settings.GIGACHAT_RETRY_MIN_WAIT, max=settings.GIGACHAT_RETRY_MAX_WAIT),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def gigachat_complete(prompt: str, temperature: float = 0.7) -> str:
    """Выполняет запрос к GigaChat с автоматическими повторами.

    Args:
        prompt: Промпт для модели.
        temperature: Температура генерации.

    Returns:
        Текст ответа модели.

    Raises:
        RuntimeError: Если не удалось получить ответ после всех попыток.
    """
    logger.info("Calling GigaChat (prompt_len=%s)", len(prompt))
    try:
        client = get_gigachat_client()
        response = client.chat(
            prompt,
            model=settings.GIGACHAT_MODEL,
            temperature=temperature,
        )
        content = getattr(response, "choices", [{}])[0].get("message", {}).get("content")
        if not content:
            content = getattr(response, "text", None)
        if not content and isinstance(response, str):
            content = response
        if not content:
            raise ValueError("Empty response from GigaChat")
        return str(content)
    except Exception as exc:
        logger.exception("GigaChat request failed: %s", exc)
        raise RuntimeError(f"GigaChat failed: {exc}") from exc


def tokenize(text: str) -> List[str]:
    """Токенизирует текст для BM25 без внешних NLTK-корпусов."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def get_embedder() -> SentenceTransformer:
    """Возвращает singleton эмбеддинг-модель."""
    global _embedder
    if _embedder is None:
        model_path = settings.resolve_path(settings.EMBEDDER_PATH)
        logger.info("Loading embedder from %s", model_path)
        _embedder = SentenceTransformer(str(model_path))
    return _embedder


def get_reranker() -> CrossEncoder:
    """Возвращает singleton cross-encoder модель."""
    global _reranker
    if _reranker is None:
        model_path = settings.resolve_path(settings.RERANKER_PATH)
        logger.info("Loading reranker from %s", model_path)
        _reranker = CrossEncoder(str(model_path))
    return _reranker


def load_catalog() -> List[Dict[str, Any]]:
    """Загружает JSON-каталог дата-продуктов."""
    catalog_path = settings.resolve_path(settings.CATALOG_PATH)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    try:
        with catalog_path.open("r", encoding="utf-8") as file:
            catalog = json.load(file)
        if not isinstance(catalog, list):
            raise ValueError("Catalog must be a JSON array")
        logger.info("Loaded catalog with %s products", len(catalog))
        return catalog
    except json.JSONDecodeError as exc:
        logger.exception("Invalid catalog JSON: %s", exc)
        raise ValueError(f"Invalid catalog JSON in {catalog_path}") from exc


def chunk_text(text: str) -> List[str]:
    """Разбивает длинный текст на перекрывающиеся чанки."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    return splitter.split_text(text or "")
