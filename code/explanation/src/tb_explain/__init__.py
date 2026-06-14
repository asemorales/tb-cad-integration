"""TB Explanation Module: structured detector output -> natural-language explanation."""

from tb_explain.adapter import adapt, confidence_to_band
from tb_explain.anatomy import BBox, bbox_to_location
from tb_explain.faithfulness import FaithfulnessResult, check_faithfulness, evaluate
from tb_explain.inference import explain
from tb_explain.schema import (
    ConfidenceBand,
    DetectorOutput,
    ImageClassification,
    Location,
    PredictedLabel,
    Region,
    RegionType,
)

__all__ = [
    "BBox",
    "ConfidenceBand",
    "DetectorOutput",
    "FaithfulnessResult",
    "ImageClassification",
    "Location",
    "PredictedLabel",
    "Region",
    "RegionType",
    "adapt",
    "bbox_to_location",
    "check_faithfulness",
    "confidence_to_band",
    "evaluate",
    "explain",
]
