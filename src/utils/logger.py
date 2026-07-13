"""
src/utils/logger.py
-------------------
Standard logger factory with rotating file handler.

On read-only filesystems (e.g., Render free tier), the log file write will
fail gracefully — the console handler is always added first so logs always
appear in stdout regardless of filesystem access.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(
    name: str,
    log_file: str = "logs/app.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a logger with console + optional rotating file handler.

    Parameters
    ----------
    name     : Logger name (module-level, e.g. 'api_main').
    log_file : Path for rotating file output. Skipped silently if the
               directory cannot be created (read-only FS on Render).
    level    : Logging level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers when module is imported multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler (always added) ───────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler (best-effort — skipped if FS is read-only) ─────────
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,   # 5 MB per file
            backupCount=5,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (OSError, PermissionError):
        # Running on a read-only filesystem (e.g., Render free tier)
        # Console handler above is sufficient.
        pass

    return logger
