"""TB report generation: structured detector output -> radiology-style report."""

from tb_report.adapter import adapt, confidence_to_band
from tb_report.anatomy import BBox, bbox_to_location
from tb_report.faithfulness import FaithfulnessResult, check_faithfulness, evaluate
from tb_report.inference import generate_report_llm
from tb_report.report import generate_report
from tb_report.schema import (
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
    "generate_report_llm",
    "generate_report",
]
