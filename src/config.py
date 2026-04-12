import os

# ---------------------------------------------------------------------------
# Configurazione Logging
# ---------------------------------------------------------------------------
LOGS_DIR = os.path.join(os.getcwd(), "../logs")
LOG_FILE = os.path.join(LOGS_DIR, "api.log")
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file di log
LOG_BACKUP_COUNT = 3  # mantieni al massimo 3 file di backup

# ---------------------------------------------------------------------------
# Configurazione Modelli
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.getcwd(), "../models")

AVAILABLE_MODELS = {
    "naive_bayes": os.path.join(MODELS_DIR, "naive_bayes_model.pkl"),
    "svm": os.path.join(MODELS_DIR, "svm_model.pkl"),
    "mlp": os.path.join(MODELS_DIR, "mlp_model.pkl"),
}

VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

DEFAULT_MODEL = "naive_bayes"
