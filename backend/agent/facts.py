"""Evidence -> numbered facts.

The orchestrator (code) decides what is true and writes each fact as one plain sentence with its exact numbers. The LLM may only
explain these facts and cite them by id; the verifier checks its text against the cited facts. Nothing here reads post-release
columns: comparables arrive already reduced to public columns by tools.get_comparables.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..app.config import CLASSES
from . import risks

NUM = re.compile(r"\d+(?:\.\d+)?")


def pct_forms(n: str) -> set[str]:
    """A number as written, plus the percent reading of a 0-1 fraction (0.31 -> 31) and a trailing '.0' trimmed."""
    out = {n}
    if n.endswith(".0"):
        out.add(n[:-2])
    try:
        f = float(n)
    except ValueError:
        return out
    if "." in n and 0 < f <= 1:
        out |= {str(round(f * 100)), f"{f * 100:.1f}".rstrip("0").rstrip(".")}
    return out


def numbers_in(text: str) -> set[str]:
    """Numbers as written in a fact. A shown percentage also allows its fraction reading (64% -> 0.64), nothing finer."""
    out: set[str] = set()
    for n, pct in re.findall(r"(\d+(?:\.\d+)?)(%?)", text):
        out |= pct_forms(n)
        if pct:
            out.add(f"{float(n) / 100:g}")
    return out


@dataclass(frozen=True)
class Fact:
    id: str
    ref: str                     # prediction | cohort | sensitivity | comparables
    text: str
    numbers: frozenset = frozenset()
    by_class: dict = field(default_factory=dict)         # class -> numbers that legitimately go with that class
    changes_top_class: bool | None = None                # sensitivity facts only


def _pc(x: float) -> str:
    return f"{x * 100:.0f}%"


def _g(v: float) -> str:
    return f"{v:g}"


def build_facts(prediction: dict, evidence: dict) -> list[Fact]:
    raw: list[dict] = []
    probs = prediction["probabilities"]
    ranked = sorted(probs, key=probs.get, reverse=True)
    top = prediction["label"]
    raw.append({
        "ref": "prediction",
        "text": f"The saved Random Forest ranks {top} highest at {_pc(probs[top])}, then {ranked[1]} at {_pc(probs[ranked[1]])} and "
                f"{ranked[2]} at {_pc(probs[ranked[2]])}. These are uncalibrated class scores for an IMDb-score band.",
        "by_class": {c: {str(round(probs[c] * 100)), f"{round(probs[c] * 100) / 100:g}"} for c in CLASSES},
    })

    cohort = evidence.get("cohort") or {}
    final = cohort.get("final") or {}
    if final.get("n"):
        c = final["counts"]
        small = " This is a small sample." if risks.cohort_is_thin(cohort) else ""
        raw.append({
            "ref": "cohort",
            "text": f"{final['n']} films in the dataset match these traits (same genres, country, rating, within 10 years and 20 minutes): "
                    f"{c['Hit']} Hit, {c['Average']} Average, {c['Flop']} Flop, mean IMDb score {final['mean_score']}.{small}",
            "by_class": {k: {str(c[k])} for k in CLASSES},
        })
    elif risks.cohort_n(cohort) == 0:
        raw.append({"ref": "cohort", "text": "No films in the dataset match all of these traits."})

    sens = evidence.get("sensitivity")
    if sens and sens.get("top_class_by_range"):
        seg = sens["top_class_by_range"]
        lo, hi = seg[0]["from"], seg[-1]["to"]
        if len(seg) == 1:
            text = (f"Sweeping {sens['label']} across its observed range ({_g(lo)} to {_g(hi)}) with every other input unchanged, the top class "
                    f"stays {seg[0]['label']}. This input alone does not change the predicted class.")
        else:
            parts = "; ".join(f"{s['label']} from {_g(s['from'])} to {_g(s['to'])}" for s in seg)
            text = (f"Sweeping {sens['label']} across its observed range ({_g(lo)} to {_g(hi)}) with every other input unchanged, "
                    f"the top class changes along the range: {parts}.")
        raw.append({"ref": "sensitivity", "text": text, "changes_top_class": len(seg) > 1})

    comps = evidence.get("comparables") or []
    if comps:
        n = {k: sum(1 for x in comps if x["Classify"] == k) for k in CLASSES}
        raw.append({"ref": "comparables", "by_class": {k: {str(n[k])} for k in CLASSES},
                    "text": f"{len(comps)} similar earlier films were found: {n['Hit']} Hit, {n['Average']} Average, {n['Flop']} Flop."})
        for x in comps[:5]:
            raw.append({"ref": "comparables", "text": f"{x['title']} ({int(x['year'])}) was classed {x['Classify']}."})

    return [Fact(id=f"F{i}", ref=r["ref"], text=r["text"], numbers=frozenset(numbers_in(r["text"])), by_class=r.get("by_class", {}),
                 changes_top_class=r.get("changes_top_class")) for i, r in enumerate(raw, 1)]
