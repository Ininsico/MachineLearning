"""Console and file logging helpers that make pipeline stages easy to follow."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

LOGGER_NAME = "hmr"
_WIDTH = 78

_LEVEL_COLOURS = {
    "DEBUG": "\033[90m",
    "INFO": "\033[0m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[41m",
}
_RESET = "\033[0m"


class _ColourFormatter(logging.Formatter):
    def __init__(self, use_colour: bool) -> None:
        super().__init__("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
        self.use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if self.use_colour:
            colour = _LEVEL_COLOURS.get(record.levelname, "")
            if colour:
                return f"{colour}{text}{_RESET}"
        return text


def setup_logging(log_file: str | Path | None = None, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream.setFormatter(_ColourFormatter(use_colour=sys.stdout.isatty() or True))
    logger.addHandler(stream)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def banner(title: str, logger: logging.Logger | None = None) -> None:
    logger = logger or get_logger()
    logger.info("")
    logger.info("=" * _WIDTH)
    logger.info(title.upper().center(_WIDTH))
    logger.info("=" * _WIDTH)


def section(title: str, logger: logging.Logger | None = None) -> None:
    logger = logger or get_logger()
    logger.info("")
    logger.info(f"--- {title} " + "-" * max(0, _WIDTH - len(title) - 5))


def kv_table(rows: Iterable[tuple[str, object]], indent: int = 2, logger: logging.Logger | None = None) -> None:
    logger = logger or get_logger()
    rows = [(str(k), str(v)) for k, v in rows]
    if not rows:
        return
    pad = max(len(k) for k, _ in rows) + indent
    prefix = " " * indent
    for key, value in rows:
        logger.info(f"{prefix}{key.ljust(pad - indent)} : {value}")


def format_metric_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    float_fmt: str = "{:.4f}",
) -> str:
    """Render an aligned plain-text table (used for console + markdown export)."""
    str_rows: list[list[str]] = []
    for row in rows:
        str_rows.append([float_fmt.format(v) if isinstance(v, float) else str(v) for v in row])
    widths = [len(h) for h in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "  ".join("-" * widths[i] for i in range(len(headers)))
    lines = [header_line, sep_line]
    for row in str_rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)
