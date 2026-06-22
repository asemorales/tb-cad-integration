"""End-to-end smoke tests for adapt(), generate_report(), and generate_report_llm()."""

from unittest.mock import MagicMock, patch

import pytest

from tb_report import adapt, check_faithfulness, generate_report_llm, generate_report
from tb_report.schema import DetectorOutput, ImageClassification, Region


PROBS = {"healthy": 0.05, "sick_non_tb": 0.12, "tb": 0.83}
DETECTIONS = [((100, 50, 300, 200), "active_tb", 0.91)]
IMAGE_SIZE = (512, 512)


# --- adapt -------------------------------------------------------------------


def test_adapt_predicted_label():
    out = adapt(PROBS, DETECTIONS, IMAGE_SIZE)
    assert out.image_classification.predicted_label == "tb"


def test_adapt_confidence_band():
    out = adapt(PROBS, DETECTIONS, IMAGE_SIZE)
    assert out.regions[0].confidence_band == "high"


def test_adapt_location():
    # cx=(100+300)/2=200, cy=(50+200)/2=125; 512/3≈170.67 → center column, upper row
    out = adapt(PROBS, DETECTIONS, IMAGE_SIZE)
    assert out.regions[0].location == "upper_center"


def test_adapt_no_detections():
    out = adapt(PROBS, [], IMAGE_SIZE)
    assert out.regions == []


def test_adapt_returns_detector_output():
    out = adapt(PROBS, DETECTIONS, IMAGE_SIZE)
    assert isinstance(out, DetectorOutput)


# --- generate_report (template-based report generation) ----------------------


def _rec(label, probs, regions):
    return DetectorOutput(
        image_classification=ImageClassification(predicted_label=label, probabilities=probs),
        regions=[Region(**r) for r in regions],
    )


def test_generate_report_matches_spec_example():
    rec = _rec(
        "tb",
        {"healthy": 0.04, "sick_non_tb": 0.09, "tb": 0.87},
        [
            {"type": "active_tb", "confidence_band": "high", "location": "upper_right"},
            {"type": "latent_tb", "confidence_band": "medium", "location": "upper_left"},
        ],
    )
    assert generate_report(rec) == (
        "FINDINGS:\n"
        "1. Right upper zone abnormality suggestive of active TB (high confidence).\n"
        "2. Left upper zone abnormality suggestive of inactive TB (medium confidence).\n"
        "IMPRESSION: Appearances are consistent with active pulmonary tuberculosis, "
        "with additional features of old, inactive disease.\n"
        "RECOMMENDATION: Refer for bacteriological confirmation."
    )


def test_generate_report_no_regions():
    rec = _rec("healthy", {"healthy": 0.92, "sick_non_tb": 0.06, "tb": 0.02}, [])
    out = generate_report(rec)
    assert "1. No TB-suggestive abnormality was localized." in out
    assert "IMPRESSION: No radiographic evidence of tuberculosis." in out


def test_generate_report_orders_upper_to_lower():
    rec = _rec(
        "tb",
        {"healthy": 0.03, "sick_non_tb": 0.07, "tb": 0.90},
        [
            {"type": "active_tb", "confidence_band": "low", "location": "lower_left"},
            {"type": "latent_tb", "confidence_band": "high", "location": "upper_right"},
        ],
    )
    lines = generate_report(rec).splitlines()
    assert lines[1].startswith("1. Right upper zone")
    assert lines[2].startswith("2. Left lower zone")


def test_generate_report_is_deterministic():
    rec = _rec("tb", {"healthy": 0.03, "sick_non_tb": 0.07, "tb": 0.90},
               [{"type": "active_tb", "confidence_band": "medium", "location": "middle_center"}])
    assert generate_report(rec) == generate_report(rec)


def test_generate_report_is_faithful_by_construction():
    rec = _rec(
        "tb",
        {"healthy": 0.03, "sick_non_tb": 0.07, "tb": 0.90},
        [
            {"type": "active_tb", "confidence_band": "high", "location": "upper_right"},
            {"type": "active_tb", "confidence_band": "medium", "location": "middle_center"},
            {"type": "latent_tb", "confidence_band": "low", "location": "lower_left"},
        ],
    )
    assert check_faithfulness(rec, generate_report(rec)).is_faithful


# --- generate_report_llm (generative report generation) ----------------------


@pytest.fixture
def detector_output():
    return adapt(PROBS, DETECTIONS, IMAGE_SIZE)


def _mock_pipeline(response: str):
    """Return a mock that mimics the transformers pipeline chat output."""
    pipe = MagicMock()
    pipe.return_value = [{"generated_text": [{"role": "assistant", "content": response}]}]
    return pipe


def test_generate_report_llm_returns_string(detector_output):
    reply = "The model identified active TB in the upper-left region with high confidence."
    with patch("tb_report.inference._load_pipeline", return_value=_mock_pipeline(reply)):
        result = generate_report_llm(detector_output)
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_report_llm_passes_json_to_model(detector_output):
    reply = "TB findings noted."
    captured = {}

    def capturing_pipe(messages, **kwargs):
        captured["messages"] = messages
        return [{"generated_text": [{"role": "assistant", "content": reply}]}]

    with patch("tb_report.inference._load_pipeline", return_value=capturing_pipe):
        generate_report_llm(detector_output)

    user_content = captured["messages"][1]["content"]
    assert "tb" in user_content
    assert "active_tb" in user_content
