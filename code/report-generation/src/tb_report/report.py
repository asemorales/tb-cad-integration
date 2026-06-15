"""Template-based report generation: DetectorOutput -> radiology-style report.

This is the paper's report generation path. It fills a fixed template from the
record's closed vocabulary, so the output is faithful by construction: it can name
nothing the record does not contain, invent no probability, and contradict no
label. Every case passes :func:`tb_report.check_faithfulness` because there is no
free text to go out of vocabulary.

The generative path in :mod:`tb_report.inference` targets the same three-section
format from a prompt, and is explored as future work; this module is the
reference output that prompt was written to reproduce.
"""

from __future__ import annotations

from tb_report.schema import DetectorOutput, PredictedLabel, Region, RegionType

# Render order for the FINDINGS list: upper to lower. Regions in the same row keep
# their record order (stable sort).
_ROW_ORDER = {"upper": 0, "middle": 1, "lower": 2}
_COL_WORD = {"left": "left", "right": "right", "center": "central"}
_ACTIVITY = {"active_tb": "active", "latent_tb": "inactive"}

_IMPRESSION: dict[PredictedLabel, str] = {
    "tb": "Appearances are consistent with pulmonary tuberculosis.",
    "sick_non_tb": "The image is abnormal, with non-TB disease the likely consideration.",
    "healthy": "The screen is negative.",
}

_RECOMMENDATION: dict[PredictedLabel, str] = {
    "tb": "Refer for bacteriological confirmation.",
    "sick_non_tb": "Recommend clinical correlation; TB-specific workup is not indicated by this screen.",
    "healthy": "No further TB workup is indicated by this screen.",
}

_NO_FINDING = "No TB-suggestive abnormality was localized."


def _zone(location: str) -> str:
    """Render a grid location as a capitalized zone phrase, patient-side column first."""
    row, col = location.split("_")
    return f"{_COL_WORD[col]} {row} zone".capitalize()


def _finding_line(region: Region) -> str:
    return (
        f"{_zone(region.location)} abnormality suggestive of "
        f"{_ACTIVITY[region.type]} TB ({region.confidence_band} confidence)."
    )


def _findings(regions: list[Region]) -> list[str]:
    if not regions:
        return [_NO_FINDING]
    ordered = sorted(regions, key=lambda r: _ROW_ORDER[r.location.split("_")[0]])
    return [_finding_line(r) for r in ordered]


def generate_report(output: DetectorOutput) -> str:
    """Return a deterministic three-section report for a ``DetectorOutput``.

    The output has exactly the FINDINGS, IMPRESSION, and RECOMMENDATION sections,
    in that order, with no other text. Given the same record it always returns the
    same string.
    """
    label = output.image_classification.predicted_label
    findings = _findings(output.regions)
    numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(findings, start=1))
    return (
        "FINDINGS:\n"
        f"{numbered}\n"
        f"IMPRESSION: {_IMPRESSION[label]}\n"
        f"RECOMMENDATION: {_RECOMMENDATION[label]}"
    )
