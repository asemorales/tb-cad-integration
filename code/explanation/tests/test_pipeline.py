"""End-to-end smoke tests for adapt() and explain()."""

from unittest.mock import MagicMock, patch

import pytest

from tb_explain import adapt, explain
from tb_explain.schema import DetectorOutput


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


# --- explain -----------------------------------------------------------------


@pytest.fixture
def detector_output():
    return adapt(PROBS, DETECTIONS, IMAGE_SIZE)


def _mock_pipeline(response: str):
    """Return a mock that mimics the transformers pipeline chat output."""
    pipe = MagicMock()
    pipe.return_value = [{"generated_text": [{"role": "assistant", "content": response}]}]
    return pipe


def test_explain_returns_string(detector_output):
    reply = "The model identified active TB in the upper-left region with high confidence."
    with patch("tb_explain.inference._load_pipeline", return_value=_mock_pipeline(reply)):
        result = explain(detector_output)
    assert isinstance(result, str)
    assert len(result) > 0


def test_explain_passes_json_to_model(detector_output):
    reply = "TB findings noted."
    captured = {}

    def capturing_pipe(messages, **kwargs):
        captured["messages"] = messages
        return [{"generated_text": [{"role": "assistant", "content": reply}]}]

    with patch("tb_explain.inference._load_pipeline", return_value=capturing_pipe):
        explain(detector_output)

    user_content = captured["messages"][1]["content"]
    assert "tb" in user_content
    assert "active_tb" in user_content
