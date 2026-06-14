"""Raw detector output -> DetectorOutput (the LLM input contract)."""

from tb_explain.anatomy import BBox, bbox_to_location
from tb_explain.schema import (
    ConfidenceBand,
    DetectorOutput,
    ImageClassification,
    PredictedLabel,
    Region,
    RegionType,
)

DEFAULT_BAND_THRESHOLDS: tuple[float, float] = (0.5, 0.8)


def confidence_to_band(
    confidence: float,
    thresholds: tuple[float, float] = DEFAULT_BAND_THRESHOLDS,
) -> ConfidenceBand:
    low_cut, high_cut = thresholds
    if confidence >= high_cut:
        return "high"
    if confidence >= low_cut:
        return "medium"
    return "low"


def adapt(
    probabilities: dict[PredictedLabel, float],
    detections: list[tuple[BBox, RegionType, float]],
    image_size: tuple[int, int],
    band_thresholds: tuple[float, float] = DEFAULT_BAND_THRESHOLDS,
) -> DetectorOutput:
    """Build a validated `DetectorOutput` from raw detector outputs."""
    predicted_label = max(probabilities, key=probabilities.__getitem__)

    regions = [
        Region(
            type=region_type,
            confidence_band=confidence_to_band(confidence, band_thresholds),
            location=bbox_to_location(bbox, image_size),
        )
        for bbox, region_type, confidence in detections
    ]

    return DetectorOutput(
        image_classification=ImageClassification(
            predicted_label=predicted_label,
            probabilities=probabilities,
        ),
        regions=regions,
    )
