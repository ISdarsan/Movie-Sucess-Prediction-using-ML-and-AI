import re
from typing import Optional

from pydantic import Field, field_validator

from ..app.schemas import PredictRequest

_TITLE_OK = re.compile(r"^[\w\s.,:;'’&!?()\-–—/]+$", re.UNICODE)


class AgentRunRequest(PredictRequest):
    """Exactly the POST /api/predict body (extra fields still forbidden) plus ONE optional research-only field.
    `research_title` is used only to look up pre-release background; it is removed before prediction and never reaches the model."""
    research_title: Optional[str] = Field(default=None, max_length=120, description="Film title for contextual research only")

    @field_validator("research_title")
    @classmethod
    def _title_clean(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = re.sub(r"\s+", " ", v).strip()
        if not v:
            return None
        if len(v) < 2 or not _TITLE_OK.match(v):
            raise ValueError("Use letters, numbers and ordinary punctuation only.")
        return v
