from typing import Literal

from pydantic import BaseModel

PredictedLabel = Literal["healthy", "sick_non_tb", "tb"]
RegionType = Literal["active_tb", "latent_tb"]
ConfidenceBand = Literal["low", "medium", "high"]
# Column is the PATIENT's side (radiographic convention): "upper_right" is the
# patient's right upper zone, i.e. the viewer's left. The detector must emit this
# in patient-side coordinates; the verbalizer renders it verbatim without flipping.
Location = Literal[
    "upper_left",
    "upper_center",
    "upper_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "lower_left",
    "lower_center",
    "lower_right",
]


class ImageClassification(BaseModel):
    predicted_label: PredictedLabel
    probabilities: dict[PredictedLabel, float]


class Region(BaseModel):
    type: RegionType
    confidence_band: ConfidenceBand
    location: Location  # patient-side coordinates; see Location above


class DetectorOutput(BaseModel):
    """Top-level JSON contract: detector output, LLM input."""

    image_classification: ImageClassification
    regions: list[Region] = []
