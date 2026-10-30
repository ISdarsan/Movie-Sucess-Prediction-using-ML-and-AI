"""Whitelisted, validated, call-bounded tools.

The orchestrator (and any future model-requested call) goes through ToolRegistry.call(), which
  * rejects tools that are not registered,
  * validates and coerces arguments with Pydantic (llama3.1:8b returns lists/ints as strings, e.g. "['Drama']" and "2005"),
  * enforces a hard cap on calls per run.
Tools are read-only. The dataset has post-release columns (gross, votes, reviews, likes, imdb_score); comparables expose only
PUBLIC_COLUMNS, and the target film's own pre-release values are the only thing used to search.
"""
from __future__ import annotations

import ast
from typing import Annotated, Any, Callable, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from ..app import data, insight
from ..app.config import CLASSES
from . import config, research

# Columns a comparable film may show. Its historical class label is a descriptive outcome of ANOTHER film, never an input.
PUBLIC_COLUMNS = ["id", "title", "year", "genres", "country", "content_rating", "duration", "Classify"]


class ToolError(RuntimeError):
    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind


def _as_list(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                v = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                pass
        else:
            v = [x.strip() for x in s.replace("|", ",").split(",") if x.strip()]
    return v


GenreList = Annotated[list[str], BeforeValidator(_as_list), Field(min_length=1, max_length=8)]


class CohortArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    genres: GenreList
    country: Optional[str] = Field(default=None, max_length=60)
    content_rating: Optional[str] = Field(default=None, max_length=20)
    year: Optional[int] = Field(default=None, ge=1800, le=2100)
    duration: Optional[float] = Field(default=None, ge=1, le=600)
    title: Optional[str] = Field(default=None, max_length=120)      # the queried film: with `year` it identifies the dataset row to leave out


class ComparablesArgs(CohortArgs):
    limit: int = Field(default=8, ge=1, le=8)


class SensitivityArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: dict
    feature: str = Field(pattern="^(budget|duration|title_year|aspect_ratio)$")


class ResearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Optional[str] = Field(default=None, max_length=120)      # optional film title, used only for the contextual lookup
    genres: GenreList
    year: int = Field(ge=1888, le=2030)                             # planned release year: fixes the pre-release cutoff


# ---------------------------------------------------------------- implementations
def cohort_tool(a: CohortArgs) -> dict:
    return insight.cohort(sorted(set(a.genres)), a.country, a.content_rating, a.year, a.duration, a.title)


def comparables_tool(a: ComparablesArgs) -> list[dict]:
    """Most similar EARLIER films (release year <= the target's), ranked by genre overlap, same country/rating, runtime and era distance."""
    df = data.movies()
    me = insight.self_match(df, a.title, a.year)        # the queried film is never its own comparable
    if me.any():
        df = df[~me]
    if a.year is not None:
        df = df[df["year"] <= a.year]
    want = set(a.genres)
    overlap = df["genre_list"].map(lambda g: len(want & set(g)) / max(1, len(want | set(g))))
    score = overlap * 4.0
    if a.country:
        score = score + (df["country"] == a.country) * 1.0
    if a.content_rating:
        score = score + (df["content_rating"] == a.content_rating) * 0.5
    if a.duration is not None:
        score = score - ((df["duration"] - a.duration).abs().fillna(60) / 60).clip(upper=1.0) * 1.0
    if a.year is not None:
        score = score - ((a.year - df["year"]).abs() / 40).clip(upper=1.0) * 0.5
    top = df.assign(_s=score)[overlap > 0].sort_values(["_s", "id"], ascending=[False, True]).head(a.limit)
    rows = []
    for rec in top[PUBLIC_COLUMNS].to_dict("records"):
        rows.append({k: data.clean(v) for k, v in rec.items()})
    return rows


def sensitivity_tool(a: SensitivityArgs) -> dict:
    """Compact view of the model's response curve: which class scores highest across the swept range."""
    full = insight.sensitivity(a.inputs, a.feature)
    prob = full["probabilities"]
    labels = [max(CLASSES, key=lambda c: prob[c][i]) for i in range(len(full["values"]))]
    segments: list[dict] = []
    for v, lab in zip(full["values"], labels):
        if segments and segments[-1]["label"] == lab:
            segments[-1]["to"] = v
        else:
            segments.append({"from": v, "to": v, "label": lab})
    cur = full["current"]
    return {"feature": full["feature"], "label": full["label"], "current_value": cur["value"], "current_scores": cur["probabilities"],
            "top_class_by_range": segments, "note": full["note"]}


def research_tool(a: ResearchArgs) -> dict:
    return research.run(a.title, sorted(set(a.genres)), a.year)     # never raises; failures come back as a status


class Tool:
    def __init__(self, name: str, args: type[BaseModel], fn: Callable[[Any], Any], description: str) -> None:
        self.name, self.args, self.fn, self.description = name, args, fn, description


class ToolRegistry:
    """One registry per run; counts calls so a run cannot loop."""

    TOOLS = {t.name: t for t in (
        Tool("get_cohort", CohortArgs, cohort_tool, "Descriptive class mix of films narrowing by the chosen pre-release traits."),
        Tool("get_comparables", ComparablesArgs, comparables_tool, "Up to 8 similar earlier films (public columns only)."),
        Tool("get_sensitivity", SensitivityArgs, sensitivity_tool, "Model response curve when one input is swept."),
        Tool("research_web", ResearchArgs, research_tool, "Allow-listed, as-of-cutoff Wikipedia background (display-only context, never a model input)."),
    )}

    def __init__(self, max_calls: int | None = None) -> None:
        self.max_calls = config.MAX_TOOL_CALLS if max_calls is None else max_calls
        self.calls: list[dict] = []

    def call(self, name: str, raw_args: dict | None) -> Any:
        tool = self.TOOLS.get(name)
        if tool is None:
            raise ToolError("unknown_tool", name[:40])
        if len(self.calls) >= self.max_calls:
            raise ToolError("tool_budget_exceeded", f"max {self.max_calls}")
        try:
            args = tool.args.model_validate(raw_args or {})
        except ValidationError as exc:
            raise ToolError("invalid_arguments", "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())[:200]) from exc
        self.calls.append({"tool": name})
        return tool.fn(args)
