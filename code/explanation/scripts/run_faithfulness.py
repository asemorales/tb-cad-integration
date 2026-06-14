"""Generate explanations for representative detector outputs and score faithfulness.

This runs the real Phi-3.5 prompt over a case mix mirroring the radiologist demo
set (single active, single latent, bilateral active, mixed, central, two-region,
sick-non-TB, healthy), then reports the faithfulness violation rate. Faithfulness
is image-independent (the model only sees the record), so a record set matching
the real case distribution measures the same property as the full notebook set.
"""

from tb_explain import check_faithfulness, evaluate, explain
from tb_explain.schema import DetectorOutput, ImageClassification, Region

TB = {"healthy": 0.03, "sick_non_tb": 0.07, "tb": 0.90}
HEALTHY = {"healthy": 0.92, "sick_non_tb": 0.06, "tb": 0.02}
SICK = {"healthy": 0.10, "sick_non_tb": 0.85, "tb": 0.05}


def rec(label, probs, regions):
    return DetectorOutput(
        image_classification=ImageClassification(predicted_label=label, probabilities=probs),
        regions=[Region(**r) for r in regions],
    )


def R(t, b, loc):
    return {"type": t, "confidence_band": b, "location": loc}


CASES = [
    ("single active, patient-left upper", rec("tb", TB, [R("active_tb", "high", "upper_left")])),
    ("single latent, patient-right upper", rec("tb", TB, [R("latent_tb", "medium", "upper_right")])),
    ("bilateral active, mid zones", rec("tb", TB, [R("active_tb", "high", "middle_left"), R("active_tb", "high", "middle_right")])),
    ("mixed: two latent, one active", rec("tb", TB, [R("latent_tb", "low", "upper_left"), R("latent_tb", "medium", "lower_right"), R("active_tb", "high", "upper_right")])),
    ("two active, upper zones", rec("tb", TB, [R("active_tb", "high", "upper_left"), R("active_tb", "medium", "upper_right")])),
    ("single active, central", rec("tb", TB, [R("active_tb", "medium", "middle_center")])),
    ("sick non-TB, no regions", rec("sick_non_tb", SICK, [])),
    ("healthy, no regions", rec("healthy", HEALTHY, [])),
]

pairs = []
for name, record in CASES:
    summary = explain(record)
    pairs.append((record, summary))
    result = check_faithfulness(record, summary)
    flag = "OK  " if result.is_faithful else "FLAG"
    print(f"\n[{flag}] {name}")
    print(f"  record : {record.model_dump_json()}")
    print(f"  summary: {summary}")
    if not result.is_faithful:
        for v in result.violations():
            print(f"  violation: {v}")

report = evaluate(pairs)
print("\n" + "=" * 60)
print(f"N={report['n']}  faithful={report['n'] - report['violations']}  "
      f"violations={report['violations']}  violation_rate={report['violation_rate']:.3f}")
