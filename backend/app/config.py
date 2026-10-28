import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
MODEL_PATH = Path(os.getenv("FRAME_MODEL_PATH", ROOT / "pre_release_movie_classifier.joblib"))
DATA_PATH = Path(os.getenv("FRAME_DATA_PATH", ROOT / "dataset" / "movie_metadata.csv"))
ARTIFACTS = BACKEND / "artifacts"
SCHEMA_PATH = Path(os.getenv("FRAME_SCHEMA_PATH", ARTIFACTS / "prerelease_schema.json"))
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "FRAME_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
CLASSES = ["Flop", "Average", "Hit"]
