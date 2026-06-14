"""Deterministic faithfulness check for generated explanations.

Faithfulness here means source-grounding: every clinical entity a summary
states must be supported by the structured ``DetectorOutput`` it was generated
from (Maynez et al. 2020; Dhingra et al. 2019). The check exploits the closed,
small vocabulary of the schema. It judges only whether the text stays within the
record, not whether the record itself is clinically correct.

The check is intentionally conservative (high precision): it flags entities that
cannot be supported by construction (radiographic findings outside the schema),
plus region types, locations, image labels, and probabilities that contradict or
exceed the record. A flagged case has at least one unsupported entity. The
reported metric is the fraction of such cases.

Usage as a library::

    from tb_explain import check_faithfulness
    result = check_faithfulness(record, summary)
    result.is_faithful          # bool
    result.violations()         # list[str]

Usage as a script (one JSON object per line, keys ``record`` and ``summary``)::

    python -m tb_explain.faithfulness pairs.jsonl
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tb_explain.schema import DetectorOutput

# Radiographic findings and anatomical structures the detector never reports.
# The schema has no field for any of these, so any mention is unsupported by
# construction. This is the highest-precision signal in the check.
OUT_OF_VOCAB_FINDINGS: frozenset[str] = frozenset(
    {
        "opacity", "opacities", "consolidation", "consolidations", "cavitation",
        "cavitations", "cavity", "cavities", "nodule", "nodules", "mass", "masses",
        "infiltrate", "infiltrates", "effusion", "effusions", "fibrosis", "fibrotic",
        "scarring", "scar", "scars", "edema", "oedema", "pneumonia", "pneumothorax",
        "atelectasis", "lymphadenopathy", "adenopathy", "calcification",
        "calcifications", "cardiomegaly", "cardiac", "heart", "diaphragm",
        "costophrenic", "pleural", "pleura", "mediastinum", "mediastinal", "hilar",
        "hilum", "bony", "bone", "bones", "rib", "ribs", "vascular", "interstitial",
        "reticular", "miliary", "emphysema", "fissure", "fissures",
    }
)

_ROW_ALIASES = {"upper": "upper", "middle": "middle", "mid": "middle", "lower": "lower"}
_COL_ALIASES = {
    "left": "left",
    "right": "right",
    "center": "center",
    "central": "center",
    "centre": "center",
}

_NEGATORS = {"no", "not", "without", "negative", "free", "absence", "absent", "rule", "ruled"}

# A summary may name excluded structures inside an explicit scope disclaimer
# ("...does not assess cardiac, pleural, or other structures"). Such a sentence
# states what is NOT covered, so its structure words are not asserted findings.
# Sentences containing any of these markers are excluded from the finding scan.
_SCOPE_MARKERS = (
    "not assess",
    "does not assess",
    "not assessed",
    "not evaluat",
    "covers only",
    "reflects only",
    "only the model",
    "only the tb",
    "no other find",
    "not cover",
    "other structures",
    "other thoracic",
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _nonscope_text(summary: str) -> str:
    """Drop sentences that are scope disclaimers, keeping the asserted body."""
    sentences = re.split(r"(?<=[.;])\s+|\n+", summary)
    kept = [s for s in sentences if not any(m in s.lower() for m in _SCOPE_MARKERS)]
    return " ".join(kept)


def _is_negated(tokens: list[str], idx: int, window: int = 3) -> bool:
    start = max(0, idx - window)
    return any(t in _NEGATORS for t in tokens[start:idx])


def _record_zones(record: DetectorOutput) -> set[tuple[str, str]]:
    zones: set[tuple[str, str]] = set()
    for region in record.regions:
        row, col = region.location.split("_")
        zones.add((row, col))
    return zones


def _mentioned_zones(tokens: list[str]) -> set[tuple[str, str]]:
    """Find (row, col) pairs named in the text, in either word order."""
    zones: set[tuple[str, str]] = set()
    for i, tok in enumerate(tokens):
        if tok in _ROW_ALIASES:
            for j in (i + 1, i - 1):
                if 0 <= j < len(tokens) and tokens[j] in _COL_ALIASES:
                    zones.add((_ROW_ALIASES[tok], _COL_ALIASES[tokens[j]]))
    return zones


@dataclass
class FaithfulnessResult:
    """Per-summary faithfulness verdict and the entities that violated it."""

    is_faithful: bool
    out_of_vocab_findings: list[str] = field(default_factory=list)
    unsupported_region_types: list[str] = field(default_factory=list)
    unsupported_locations: list[str] = field(default_factory=list)
    contradicted_label: str | None = None
    invented_probabilities: list[str] = field(default_factory=list)

    def violations(self) -> list[str]:
        out: list[str] = []
        out += [f"out-of-vocabulary finding: {t}" for t in self.out_of_vocab_findings]
        out += [f"unsupported region type: {t}" for t in self.unsupported_region_types]
        out += [f"unsupported location: {r} {c}" for r, c in self._loc_pairs()]
        if self.contradicted_label:
            out.append(f"contradicted image label: {self.contradicted_label}")
        out += [f"invented probability: {p}" for p in self.invented_probabilities]
        return out

    def _loc_pairs(self) -> list[tuple[str, str]]:
        return [tuple(loc.split(" ", 1)) for loc in self.unsupported_locations]  # type: ignore[misc]


def check_faithfulness(record: DetectorOutput, summary: str) -> FaithfulnessResult:
    """Return a :class:`FaithfulnessResult` for one (record, summary) pair."""
    text = summary.lower()
    tokens = _tokens(summary)
    token_set = set(tokens)

    # Findings and probabilities are scanned on the body only, so a structure
    # named inside a scope disclaimer is not counted as an asserted finding.
    body = _nonscope_text(summary)
    body_tokens = set(_tokens(body))
    findings = sorted(t for t in OUT_OF_VOCAB_FINDINGS if t in body_tokens)

    region_types = {r.type for r in record.regions}
    bad_types: list[str] = []
    if "latent" in token_set and "latent_tb" not in region_types:
        bad_types.append("latent")
    if "active" in token_set and "active_tb" not in region_types:
        bad_types.append("active")

    record_zones = _record_zones(record)
    bad_locations = sorted(
        f"{row} {col}" for row, col in _mentioned_zones(tokens) if (row, col) not in record_zones
    )

    contradicted = _check_label(record, text)

    invented = _check_probabilities(record, body.lower())

    is_faithful = not (findings or bad_types or bad_locations or contradicted or invented)
    return FaithfulnessResult(
        is_faithful=is_faithful,
        out_of_vocab_findings=findings,
        unsupported_region_types=bad_types,
        unsupported_locations=bad_locations,
        contradicted_label=contradicted,
        invented_probabilities=invented,
    )


def _normalize_label(phrase: str) -> str | None:
    """Map a free-text classification phrase to a schema label, or None."""
    p = phrase.lower()
    if "non" in p and "tb" in p:  # "non-TB", "sick non-TB"
        return "sick_non_tb"
    if "sick" in p:
        return "sick_non_tb"
    if "tuberculosis" in p or re.search(r"\btb\b", p):
        return "tb"
    if "healthy" in p or "normal" in p:
        return "healthy"
    return None


def _check_label(record: DetectorOutput, text: str) -> str | None:
    """Flag an overall classification the text asserts that contradicts the record.

    The asserted label is read from an explicit classification phrase ("classified
    as ...", "shows ...", "consistent with ..."), so the word TB used adjectivally
    ("TB regions", "TB screening model", "non-TB") is never mistaken for a TB call.
    """
    predicted = record.image_classification.predicted_label

    m = re.search(
        r"classif\w+\s+as\s+(?:a\s+|an\s+)?([a-z][a-z\- ]*?)\s*[.,;]", text
    ) or re.search(
        r"(?:image|case|finding)s?\s+(?:is|are|appears?)\s+([a-z][a-z\- ]*?)\s*[.,;]", text
    )
    if m:
        asserted = _normalize_label(m.group(1))
        if asserted and asserted != predicted:
            return f"states {asserted} but record predicts {predicted}"

    # Explicit, non-adjectival contradictions as a fallback.
    if predicted == "tb" and re.search(r"\b(no|negative for|no evidence of)\b[^.;]*\btb\b", text):
        return "states no TB but record predicts TB"
    if predicted != "tb" and re.search(
        r"\b(positive for|shows?|consistent with)\b[^.;]*\btuberculosis\b", text
    ):
        return f"asserts tuberculosis but record predicts {predicted}"
    return None


def _check_probabilities(record: DetectorOutput, text: str) -> list[str]:
    """Flag percentages in the text that match no record probability."""
    allowed = {round(p * 100) for p in record.image_classification.probabilities.values()}
    invented: list[str] = []
    for match in re.findall(r"(\d+(?:\.\d+)?)\s*%", text):
        value = round(float(match))
        if all(abs(value - a) > 1 for a in allowed):
            invented.append(f"{match}%")
    return invented


def evaluate(pairs: list[tuple[DetectorOutput, str]]) -> dict:
    """Score a list of (record, summary) pairs.

    Returns a dict with the violation rate (fraction of cases with at least one
    unsupported entity), the complementary faithful rate, the case count, and the
    per-case :class:`FaithfulnessResult` objects.
    """
    results = [check_faithfulness(rec, summ) for rec, summ in pairs]
    n = len(results)
    violations = sum(1 for r in results if not r.is_faithful)
    return {
        "n": n,
        "violations": violations,
        "violation_rate": violations / n if n else 0.0,
        "faithful_rate": (n - violations) / n if n else 0.0,
        "results": results,
    }


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pairs",
        help="JSONL file; each line an object with keys 'record' (DetectorOutput JSON) and 'summary'.",
    )
    parser.add_argument("--verbose", action="store_true", help="print every violating case")
    args = parser.parse_args(argv)

    pairs: list[tuple[DetectorOutput, str]] = []
    with open(args.pairs) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append((DetectorOutput.model_validate(obj["record"]), obj["summary"]))

    report = evaluate(pairs)
    print(
        f"cases: {report['n']}  "
        f"faithful: {report['n'] - report['violations']}  "
        f"violations: {report['violations']}  "
        f"violation_rate: {report['violation_rate']:.3f}"
    )
    if args.verbose:
        for i, (result, (_, summary)) in enumerate(zip(report["results"], pairs)):
            if not result.is_faithful:
                print(f"\n[case {i}] {summary}")
                for v in result.violations():
                    print(f"  - {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
