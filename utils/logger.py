"""Logging helpers shared across the application."""
import logging
import os
from typing import List


def get_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger; creates handlers only on first call."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class KeywordSuppressFilter(logging.Filter):
    """Drop log records whose message contains any of the given keywords."""

    def __init__(self, keywords: List[str]):
        super().__init__()
        self.keywords = keywords

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(kw in msg for kw in self.keywords)
