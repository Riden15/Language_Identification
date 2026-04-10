import logging
import os
from logging.handlers import RotatingFileHandler

from config import LOG_BACKUP_COUNT, LOG_FILE, LOG_MAX_BYTES, LOGS_DIR

# ---------------------------------------------------------------------------
# Configurazione del Logging
# ---------------------------------------------------------------------------
# Il logger scrive sia su file (con rotazione automatica) che sulla console.

log_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("MuseumLangApi")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Handler su console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # Handler su file con rotazione automatica
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
