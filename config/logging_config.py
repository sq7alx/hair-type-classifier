import logging
import logging.handlers
import sys
from pathlib import Path

def setup_logger(
    name: str = "hair_type_classifier",
    level: int = logging.INFO,
    log_file: str | None = None,
    console: bool = True,
    file: bool = True,
    force: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers and not force:
        return logger

    if force:
        logger.handlers.clear()

    logger.setLevel(level)

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file and file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            filename=str(path),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger

def get_logger(name: str = __name__) -> logging.Logger:
    return logging.getLogger(name)
