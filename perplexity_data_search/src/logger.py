"""Настройка структурированного логирования."""

import logging
from pathlib import Path

from src.config import settings


def setup_logger(name: str) -> logging.Logger:
    """Создаёт и настраивает логгер с выводом в консоль и файл.

    Args:
        name: Имя логгера (обычно __name__).

    Returns:
        Настроенный экземпляр logging.Logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    log_dir = settings.resolve_path(Path("logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "perplexity_search.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
