import os
from backend.config import settings

def ensure_directories():
    """Ensure all required directories exist."""
    dirs = [
        settings.RAW_DATA_DIR,
        settings.STAGED_DATA_DIR,
        settings.CURATED_DATA_DIR,
        settings.DATA_STORE_DIR,
        settings.POWERBI_EXPORT_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Ensured directory: {d}")

if __name__ == "__main__":
    ensure_directories()
