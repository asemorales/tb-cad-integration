# TB Report Generation

Turns the output of a TB classification + detection model into a radiology-style report (the FINDINGS, IMPRESSION, and RECOMMENDATION sections of a chest X-ray read).

The public API has three calls: `adapt()` (raw detector output to a validated JSON record), and two report generators that turn that record into text.

- `generate_report()` is **template-based report generation**. It fills a fixed clinical template from the record's closed vocabulary, so its output is faithful by construction: it can name nothing the record does not contain, invent no probability, and contradict no label. It needs no model and is deterministic. This is the path the paper evaluates and the one `scripts/sample_figure.py` uses for the figure.
- `generate_report_llm()` is **generative report generation**. It prompts a small instruction-tuned model (Phi-3.5-mini) to produce the same three-section format, then leans on `check_faithfulness()` to catch any drift. It is explored as future work and requires the optional `inference` extra.

Both paths emit the same three sections (FINDINGS, IMPRESSION, RECOMMENDATION); the prompt that drives `generate_report_llm()` was written to reproduce `generate_report()`'s output.

## Install

```bash
# Core (adapt + template-based generate_report, no model needed)
uv pip install -e .

# With the LLM backend (required for generate_report_llm)
uv pip install -e ".[inference]"
```

## Pipeline

```
CXR image
  → TB detector (classifier + region detector)
  → structured record         (adapt)
  → radiology-style report     (generate_report: template-based, or generate_report_llm: generative)
```

## Schema (draft)

Report generator input:

```json
{
  "image_classification": {
    "predicted_label": "tb",
    "probabilities": {
      "healthy": 0.05,
      "sick_non_tb": 0.12,
      "tb": 0.83
    }
  },
  "regions": [
    {
      "type": "active_tb",
      "confidence_band": "high",
      "location": "upper_right"
    }
  ]
}
```

- `predicted_label` ∈ {`healthy`, `sick_non_tb`, `tb`}
- `type` ∈ {`active_tb`, `latent_tb`}
- `confidence_band` ∈ {`low`, `medium`, `high`}
- `location` is a string from a fixed anatomical vocabulary (currently a 3×3 grid over the lung field)
- `regions` is `[]` when no detections
