"""Deterministic, evidence-backed risk items. No LLM is involved anywhere in this module.

Each item names the rule that fired and the evidence values it used, so it can be traced and re-derived. If the evidence a rule needs is
missing or malformed, that rule simply does not fire; nothing is guessed. General model caveats (IMDb-band meaning, uncalibrated scores,
rare Flops) come from the model's own metrics via prediction["caveats"] and stay separate from these evidence-specific items.

Rules (id: condition -> item):
  cohort_empty      cohort evidence present and n == 0
  cohort_thin       0 < n < COHORT_THIN_N
  comparables_none  comparables list present, valid count == 0 (and list is empty)
  comparables_few   0 < valid count < MIN_COMPARABLES
  sensitivity_no_change   one top-class segment across the swept range
  sensitivity_changes     more than one top-class segment
"""
from __future__ import annotations

from numbers import Real

from ..app.config import CLASSES
from . import config


def _num(v) -> bool:
    return isinstance(v, Real) and not isinstance(v, bool) and v == v


def _g(v: float) -> str:
    return f"{v:g}"


def cohort_n(cohort) -> int | None:
    """The matched-film count if the cohort evidence is well formed, else None."""
    final = cohort.get("final") if isinstance(cohort, dict) else None
    n = final.get("n") if isinstance(final, dict) else None
    return n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else None


def cohort_is_thin(cohort) -> bool:
    n = cohort_n(cohort)
    return n is not None and 0 < n < config.COHORT_THIN_N


def valid_comparables(comps) -> list[dict] | None:
    """Comparable films that carry the fields the report relies on. None when the evidence is absent (e.g. the tool failed)."""
    if not isinstance(comps, list):
        return None
    return [c for c in comps if isinstance(c, dict) and isinstance(c.get("title"), str) and c.get("title")
            and _num(c.get("year")) and c.get("Classify") in CLASSES]


def _item(rid: str, category: str, severity: str, text: str, rule: str, evidence: dict) -> dict:
    return {"id": rid, "category": category, "severity": severity, "text": text, "rule": rule, "evidence": evidence}


def generate(prediction: dict, evidence: dict) -> list[dict]:
    """Structured risk items for one run. Pure function of its inputs: same inputs, same output, same order (cohort, comparables, sensitivity)."""
    out: list[dict] = []
    evidence = evidence if isinstance(evidence, dict) else {}

    n = cohort_n(evidence.get("cohort"))
    if n == 0:
        out.append(_item("cohort_empty", "cohort", "caution",
                         "No films in the dataset match all of these traits, so there is no historical class mix to compare against.",
                         "cohort.final.n == 0", {"cohort.final.n": 0}))
    elif n is not None and n < config.COHORT_THIN_N:
        out.append(_item("cohort_thin", "cohort", "caution",
                         f"Only {n} dataset films match these traits (fewer than {config.COHORT_THIN_N}), so their class mix is a weak guide.",
                         f"0 < cohort.final.n < {config.COHORT_THIN_N}", {"cohort.final.n": n}))

    raw = evidence.get("comparables")
    comps = valid_comparables(raw)
    if comps is not None and not (raw and not comps):          # a non-empty list with no valid film is malformed evidence: say nothing
        k = len(comps)
        if k == 0:
            out.append(_item("comparables_none", "comparables", "caution",
                             "No similar earlier films were found, so there are no comparable examples to show.",
                             "valid comparables == 0", {"comparables.count": 0}))
        elif k < config.MIN_COMPARABLES:
            out.append(_item("comparables_few", "comparables", "caution",
                             f"Only {k} similar earlier films were found (fewer than {config.MIN_COMPARABLES}), so the examples are too few to generalise from.",
                             f"0 < valid comparables < {config.MIN_COMPARABLES}", {"comparables.count": k}))

    sens = evidence.get("sensitivity")
    seg = sens.get("top_class_by_range") if isinstance(sens, dict) else None
    label = sens.get("label") if isinstance(sens, dict) else None
    if (isinstance(seg, list) and seg and isinstance(label, str) and label
            and all(isinstance(s, dict) and _num(s.get("from")) and _num(s.get("to")) and s.get("label") in CLASSES for s in seg)):
        lo, hi = seg[0]["from"], seg[-1]["to"]
        if len(seg) == 1:
            out.append(_item("sensitivity_no_change", "sensitivity", "info",
                             f"Sweeping {label} from {_g(lo)} to {_g(hi)} did not change the top class. That holds for this tested range with every other "
                             f"input unchanged; it does not show that {label} is unimportant.",
                             "one top-class segment across the sweep", {"sensitivity.segments": 1, "sensitivity.top_class": seg[0]["label"]}))
        else:
            out.append(_item("sensitivity_changes", "sensitivity", "info",
                             f"The top class changes within the observed {label} range ({_g(lo)} to {_g(hi)}), so this result depends on that input. "
                             f"The sweep describes how the model responds, not why a film is rated as it is.",
                             "more than one top-class segment across the sweep", {"sensitivity.segments": len(seg)}))
    return out
