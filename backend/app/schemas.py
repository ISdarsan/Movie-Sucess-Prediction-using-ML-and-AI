from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    """Pre-release inputs only: everything here is known before a film is released. Field names are the model's own columns
    (genres is sent as a list). Votes, reviews, gross and social-media likes are deliberately NOT accepted."""
    model_config = ConfigDict(extra="forbid")

    # required
    duration: float = Field(ge=1, le=600, description="Final runtime in minutes")
    title_year: int = Field(ge=1888, le=2030, description="Planned release year")
    genres: list[str] = Field(min_length=1, max_length=8)
    country: str = Field(min_length=1, max_length=60)
    content_rating: str = Field(min_length=1, max_length=20)

    # optional: when omitted the saved pipeline imputes the training median (numeric) or uses an 'Unknown' category
    budget: Optional[float] = Field(default=None, ge=0, le=10_000_000_000, description="Production budget, USD for US films (local currency elsewhere in the training data)")
    language: Optional[str] = Field(default=None, max_length=40)
    color: Optional[str] = Field(default=None, max_length=30)
    aspect_ratio: Optional[float] = Field(default=None, ge=0.5, le=10)

    @field_validator("genres")
    @classmethod
    def _genres_clean(cls, v: list[str]) -> list[str]:
        out = sorted({g.strip() for g in v if g and g.strip()})
        if not out:
            raise ValueError("Select at least one genre.")
        return out


class SensitivityRequest(BaseModel):
    """The exact inputs of a prediction plus the one input to sweep."""
    model_config = ConfigDict(extra="forbid")
    inputs: PredictRequest
    feature: Literal["budget", "duration", "title_year", "aspect_ratio"]
