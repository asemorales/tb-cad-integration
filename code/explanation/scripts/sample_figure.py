"""Generate one explanation for the figure's two-box record (greedy, reproducible)."""

from tb_explain import check_faithfulness, explain
from tb_explain.schema import DetectorOutput, ImageClassification, Region

record = DetectorOutput(
    image_classification=ImageClassification(
        predicted_label="tb",
        probabilities={"healthy": 0.04, "sick_non_tb": 0.09, "tb": 0.87},
    ),
    regions=[
        Region(type="active_tb", confidence_band="high", location="upper_right"),
        Region(type="latent_tb", confidence_band="medium", location="upper_left"),
    ],
)

summary = explain(record, do_sample=False)
res = check_faithfulness(record, summary)

print("RECORD :", record.model_dump_json())
print("SUMMARY:", summary)
print("FAITHFUL:", res.is_faithful, res.violations())
