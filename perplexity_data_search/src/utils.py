"""Общие утилиты проекта: LLM, модели, токенизация, загрузка данных."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage
from langchain_gigachat.chat_models import GigaChat
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings
from src.logger import setup_logger

logger = setup_logger(__name__)

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_gigachat_llm: GigaChat | None = None


def _build_gigachat_llm(temperature: float | None = None) -> GigaChat:
    """Создаёт экземпляр LangChain GigaChat на основе настроек."""
    kwargs: Dict[str, Any] = {
        "credentials": settings.GIGACHAT_CREDENTIALS,
        "scope": settings.GIGACHAT_SCOPE,
        "model": settings.GIGACHAT_MODEL,
        "verify_ssl_certs": settings.GIGACHAT_VERIFY_SSL,
        "max_tokens": settings.GIGACHAT_MAX_TOKENS,
        "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
        "top_p": settings.GIGACHAT_TOP_P,
        "timeout": settings.GIGACHAT_TIMEOUT,
        "verbose": settings.GIGACHAT_VERBOSE,
        "profanity_check": settings.GIGACHAT_PROFANITY_CHECK,
        "streaming": settings.GIGACHAT_STREAMING,
    }

    if settings.GIGACHAT_BASE_URL:
        kwargs["base_url"] = settings.GIGACHAT_BASE_URL
    if settings.GIGACHAT_CERT_FILE:
        kwargs["cert_file"] = settings.GIGACHAT_CERT_FILE
    if settings.GIGACHAT_KEY_FILE:
        kwargs["key_file"] = settings.GIGACHAT_KEY_FILE

    return GigaChat(**kwargs)


def get_gigachat_llm() -> GigaChat:
    """Возвращает singleton-экземпляр GigaChat LangChain."""
    global _gigachat_llm
    if _gigachat_llm is None:
        logger.info("Initializing LangChain GigaChat client")
        _gigachat_llm = _build_gigachat_llm()
    return _gigachat_llm


def get_supervisor_agent(tools: List[Any]) -> Any:
    """Создаёт tool-bound агент в стиле llm.bind_tools(tools=...)."""
    llm = get_gigachat_llm()
    return llm.bind_tools(tools=tools)


@retry(
    stop=stop_after_attempt(settings.GIGACHAT_RETRY_ATTEMPTS),
    wait=wait_exponential(min=settings.GIGACHAT_RETRY_MIN_WAIT, max=settings.GIGACHAT_RETRY_MAX_WAIT),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def gigachat_complete(prompt: str, temperature: float = 0.7) -> str:
    """Выполняет запрос к GigaChat через LangChain invoke с автоматическими повторами."""
    logger.info("Calling GigaChat (prompt_len=%s)", len(prompt))
    try:
        llm = get_gigachat_llm()
        if abs(temperature - settings.LLM_TEMPERATURE) > 1e-9:
            llm = _build_gigachat_llm(temperature=temperature)

        response = llm.invoke(prompt)

        if isinstance(response, BaseMessage):
            content = response.content
        else:
            content = response

        if isinstance(content, list):
            content = " ".join(str(item) for item in content)
        content = str(content).strip()

        if not content:
            raise ValueError("Empty response from GigaChat")
        return content
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
