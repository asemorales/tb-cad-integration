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
    "sick_non_tb": "The image is abnormal, with non-TB disease the likely consideration.",
    # A normal radiograph does not exclude TB (it can present with a clear film),
    # so the impression is scoped to the radiographic read, not the whole screen.
    "healthy": "No radiographic evidence of tuberculosis.",
}

# The TB impression reflects lesion activity, since activity drives management:
# active disease is potentially infectious and warrants confirmation, while old
# inactive disease does not. The wording avoids the word "latent" by construction.
_TB_IMPRESSION = "Appearances are consistent with pulmonary tuberculosis."
_TB_IMPRESSION_ACTIVE = "Appearances are consistent with active pulmonary tuberculosis."
_TB_IMPRESSION_INACTIVE = "Appearances are consistent with inactive pulmonary tuberculosis."
_TB_IMPRESSION_BOTH = (
    "Appearances are consistent with active pulmonary tuberculosis, "
    "with additional features of old, inactive disease."
)

_RECOMMENDATION: dict[PredictedLabel, str] = {
    "sick_non_tb": "Recommend clinical correlation. TB-specific workup is not indicated by this screen.",
    "healthy": "No further TB workup is indicated by this screen.",
}

# The TB recommendation tracks activity: active disease is potentially infectious
# and warrants bacteriological confirmation, while old inactive disease warrants
# clinical correlation rather than bacteriology.
_TB_RECOMMENDATION_ACTIVE = "Refer for bacteriological confirmation."
_TB_RECOMMENDATION_INACTIVE = "Recommend clinical correlation to confirm inactive disease."

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


def _impression(label: PredictedLabel, regions: list[Region]) -> str:
    """Return the IMPRESSION line, reflecting lesion activity for the TB class."""
    if label != "tb":
        return _IMPRESSION[label]
    has_active = any(r.type == "active_tb" for r in regions)
    has_inactive = any(r.type == "latent_tb" for r in regions)
    if has_active and has_inactive:
        return _TB_IMPRESSION_BOTH
    if has_active:
        return _TB_IMPRESSION_ACTIVE
    if has_inactive:
        return _TB_IMPRESSION_INACTIVE
    return _TB_IMPRESSION


def _recommendation(label: PredictedLabel, regions: list[Region]) -> str:
    """Return the RECOMMENDATION line, reflecting lesion activity for the TB class."""
    if label != "tb":
        return _RECOMMENDATION[label]
    has_active = any(r.type == "active_tb" for r in regions)
    # A positive TB call with no localized region still warrants confirmation.
    if has_active or not regions:
        return _TB_RECOMMENDATION_ACTIVE
    return _TB_RECOMMENDATION_INACTIVE


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
        f"IMPRESSION: {_impression(label, output.regions)}\n"
        f"RECOMMENDATION: {_recommendation(label, output.regions)}"
    )
