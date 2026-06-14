"""Public entry point: detector output -> natural-language explanation."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from tb_explain.schema import DetectorOutput

if TYPE_CHECKING:
    from transformers import Pipeline

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

SYSTEM_PROMPT = (
    "You turn a tuberculosis (TB) screening model's JSON output into a short, report-style "
    "summary for a non-radiologist reader. You never see the radiograph, so state only what "
    "the JSON contains.\n\n"
    "Output exactly three labeled sections in this order and nothing else. Do not add notes, "
    "disclaimers, parenthetical asides, or any explanation of your reasoning.\n\n"
    "FINDINGS: a numbered list, one flagged region per line, ordered upper to lower. Write each "
    "line as 'ZONE abnormality suggestive of ACTIVITY TB (CONFIDENCE confidence).', where ZONE "
    "is the location with its first letter capitalized (for example 'Right upper zone'), "
    "ACTIVITY is 'active' for an active_tb region and 'inactive' for a latent_tb region, "
    "and CONFIDENCE is low, medium, or high. Do not use the word 'latent'. If there are no "
    "regions, write a single line: '1. No TB-suggestive abnormality was localized.'\n"
    "IMPRESSION: one sentence in clinical wording, with no zone or laterality detail. For the "
    "TB class, write exactly 'Appearances are consistent with pulmonary tuberculosis.'. For "
    "sick non-TB, write that the image is abnormal with non-TB disease the likely "
    "consideration. For healthy, write that the screen is negative.\n"
    "RECOMMENDATION: one line driven only by the class. TB: refer for bacteriological "
    "confirmation. Sick non-TB: recommend clinical correlation; TB-specific workup is not "
    "indicated by this screen. Healthy: no further TB workup is indicated by this screen.\n\n"
    "Location: the column is the patient's side. Render 'upper_right' as the right upper zone "
    "and 'lower_left' as the left lower zone; a 'center' column is the central zone.\n\n"
    "Never name a radiographic finding absent from the JSON (no opacities, consolidation, "
    "cavitation, effusion, fibrosis, heart, diaphragm, or ribs) and never invent probabilities "
    "or numbers.\n\n"
    "Example. class tb; regions: [active_tb, high, upper_right], [latent_tb, medium, "
    "upper_left]. Summary:\n"
    "FINDINGS:\n"
    "1. Right upper zone abnormality suggestive of active TB (high confidence).\n"
    "2. Left upper zone abnormality suggestive of inactive TB (medium confidence).\n"
    "IMPRESSION: Appearances are consistent with pulmonary tuberculosis.\n"
    "RECOMMENDATION: Refer for bacteriological confirmation."
)


def _format_user_message(output: DetectorOutput) -> str:
    return (
        "TB screening model output:\n"
        + output.model_dump_json(indent=2)
        + "\n\nWrite the summary now. Use only the values in this JSON."
    )


@lru_cache(maxsize=1)
def _load_pipeline() -> "Pipeline":
    import torch
    from transformers import pipeline

    return pipeline(
        "text-generation",
        model=MODEL_ID,
        torch_dtype=torch.bfloat16,
    )


def explain(output: DetectorOutput, max_new_tokens: int = 256, **gen_kwargs) -> str:
    """Return a natural-language explanation for a DetectorOutput.

    Lazy-loads the model on first call; subsequent calls reuse the cached pipeline.
    Extra generation arguments (for example ``do_sample=False`` for reproducible
    greedy decoding) are passed through to the pipeline.
    """
    pipe = _load_pipeline()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _format_user_message(output)},
    ]
    result = pipe(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
    return result[0]["generated_text"][-1]["content"].strip()
