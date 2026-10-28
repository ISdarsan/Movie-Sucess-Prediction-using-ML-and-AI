"""Loads the saved PRE-RELEASE pipeline once and turns validated requests into model input.

The input contract comes from backend/artifacts/prerelease_schema.json, written by backend/ml/run_prerelease.py in the same run
that saved the model. On load the service checks that the pipeline really expects exactly those columns, so the form, the API
and the fitted preprocessing cannot silently disagree.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import CLASSES, MODEL_PATH, SCHEMA_PATH

log = logging.getLogger("frame.model")


class ModelUnavailable(RuntimeError):
    """Raised when the saved pipeline could not be loaded."""


class ModelService:
    def __init__(self) -> None:
        self.pipeline = None
        self.error: str | None = None
        self.schema: dict[str, Any] = {}
        self.columns: list[str] = []
        self.numeric: list[str] = []
        self.categorical: dict[str, list[str]] = {}
        self.genres: list[str] = []
        self.medians: dict[str, float] = {}
        self.use_people = False

    def load(self, model_path=None, schema_path=None) -> None:
        """Load the saved pipeline and its schema. Paths default to the live artifact; tests pass a candidate's paths."""
        try:
            self.schema = json.loads(Path(schema_path or SCHEMA_PATH).read_text(encoding="utf8"))
            pipe = joblib.load(model_path or MODEL_PATH)
            clf = pipe.named_steps["clf"]
            if sorted(map(str, clf.classes_)) != sorted(CLASSES):
                raise ValueError("unexpected class labels")
            expected = list(pipe.named_steps["prep"].feature_names_in_)
            if expected != self.schema["inputs"]:
                raise ValueError("pipeline input columns do not match the saved schema")
            self.pipeline = pipe
            self.columns = expected
            self.use_people = bool(self.schema["use_people"])
            self.numeric = list(self.schema["numeric"])
            self.categorical = {c: list(v) for c, v in self.schema["categorical"].items()}
            self.genres = list(self.schema["genres"])
            self.medians = {c: float(v["median"]) for c, v in self.schema["numeric"].items()}
            self.error = None
        except Exception as exc:  # noqa: BLE001 - reported through /api/health, never leaked to clients
            log.exception("model load failed")
            self.pipeline = None
            self.error = f"{type(exc).__name__}"

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def describe(self) -> dict[str, Any]:
        if not self.ready:
            return {"loaded": False, "error_type": self.error}
        clf = self.pipeline.named_steps["clf"]
        core = getattr(clf, "base_", clf)                      # unwrap WeightedClassifier to report the real algorithm
        return {"loaded": True, "estimator": type(core).__name__, "classes": list(map(str, clf.classes_)),
                "n_input_features": len(self.columns), "stage": "pre-release", "uses_people": self.use_people,
                "n_estimators": int(getattr(core, "n_estimators", getattr(core, "n_estimators_", 0)) or 0)}

    def frame(self, values: dict[str, Any]) -> pd.DataFrame:
        row: dict[str, Any] = {}
        for col in self.columns:
            v = values.get(col)
            if col in self.numeric:
                row[col] = np.nan if v is None or v == "" else float(v)
            else:
                row[col] = np.nan if v in (None, "") else str(v)
        df = pd.DataFrame([row], columns=self.columns)
        for col in self.numeric:
            df[col] = df[col].astype("float64")
        return df

    def predict(self, values: dict[str, Any]) -> dict[str, Any]:
        if not self.ready:
            raise ModelUnavailable
        df = self.frame(values)
        clf = self.pipeline.named_steps["clf"]
        proba = self.pipeline.predict_proba(df)[0]
        by_class = {str(c): float(p) for c, p in zip(clf.classes_, proba)}
        label = str(self.pipeline.predict(df)[0])
        return {"label": label, "probabilities": {c: round(by_class[c], 4) for c in CLASSES}}


service = ModelService()
