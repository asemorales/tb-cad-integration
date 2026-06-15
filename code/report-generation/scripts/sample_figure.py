"""Generate one report for the figure's two-box record (template-based, deterministic)."""

from tb_report import check_faithfulness, generate_report
from tb_report.schema import DetectorOutput, ImageClassification, Region

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

report = generate_report(record)
res = check_faithfulness(record, report)

print("RECORD :", record.model_dump_json())
print("REPORT :", report)
print("FAITHFUL:", res.is_faithful, res.violations())
