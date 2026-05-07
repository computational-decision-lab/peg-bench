import logging
from pathlib import Path


def setup_logging(log_dir: str | Path, name: str = "kronos") -> logging.Logger:
    """Configure a simple rotating logger that writes to console and file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Reuse handlers if already configured
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s (%(filename)s:%(lineno)d) - %(message)s"
        )
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
