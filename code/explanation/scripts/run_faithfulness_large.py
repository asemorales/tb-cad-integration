"""Sound faithfulness run: sample many diverse detector records, generate with
greedy decoding (reproducible), score, and report the rate with a confidence
interval and a per-category breakdown.

Records are constructed to span the schema (since end-to-end integration with the
real classifier and localizer does not exist yet). Generation is image-blind, so
the faithfulness rate depends on the record distribution, not on real images.

Usage: python scripts/run_faithfulness_large.py [N]
"""

import json
import math
import random
import sys
from collections import Counter

from tb_explain import check_faithfulness, evaluate, explain
from tb_explain.schema import DetectorOutput, ImageClassification, Region

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
random.seed(20260612)

LOCS = [
    "upper_left", "upper_center", "upper_right",
    "middle_left", "middle_center", "middle_right",
    "lower_left", "lower_center", "lower_right",
]
TYPES = ["active_tb", "latent_tb"]
BANDS = ["low", "medium", "high"]
KEYS = ["healthy", "sick_non_tb", "tb"]


def make_probs(label):
    vals = [random.uniform(0.01, 0.20) for _ in KEYS]
    vals[KEYS.index(label)] = random.uniform(0.55, 0.97)
    s = sum(vals)
    return {k: round(v / s, 3) for k, v in zip(KEYS, vals)}


def make_record():
    r = random.random()
    label = "tb" if r < 0.6 else ("sick_non_tb" if r < 0.8 else "healthy")
    regions = []
    if label == "tb":
        for _ in range(random.randint(1, 4)):
            regions.append(Region(
                type=random.choice(TYPES),
                confidence_band=random.choice(BANDS),
                location=random.choice(LOCS),
            ))
    return DetectorOutput(
        image_classification=ImageClassification(predicted_label=label, probabilities=make_probs(label)),
        regions=regions,
    )


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return min(1.0, center + half)


pairs = []
with open("/tmp/faith_pairs.jsonl", "w") as fout:
    for i in range(N):
        rec = make_record()
        summary = explain(rec, do_sample=False)
        pairs.append((rec, summary))
        res = check_faithfulness(rec, summary)
        fout.write(json.dumps({"record": json.loads(rec.model_dump_json()), "summary": summary}) + "\n")
        fout.flush()
        print(f"{i + 1}/{N} {'OK' if res.is_faithful else 'FLAG ' + str(res.violations())}", flush=True)

rep = evaluate(pairs)
cat = Counter()
tb_n = tb_v = 0
for res, (rec, _) in zip(rep["results"], pairs):
    if rec.image_classification.predicted_label == "tb":
        tb_n += 1
        tb_v += 0 if res.is_faithful else 1
    if res.out_of_vocab_findings:
        cat["out_of_vocab_finding"] += 1
    if res.unsupported_region_types:
        cat["unsupported_region_type"] += 1
    if res.unsupported_locations:
        cat["unsupported_location"] += 1
    if res.contradicted_label:
        cat["contradicted_label"] += 1
    if res.invented_probabilities:
        cat["invented_probability"] += 1

print("\n==== RESULTS ====", flush=True)
print(f"N={rep['n']} violations={rep['violations']} rate={rep['violation_rate']:.4f} "
      f"wilson95_upper={wilson_upper(rep['violations'], rep['n']):.4f}", flush=True)
print(f"TB-subset: n={tb_n} violations={tb_v} rate={(tb_v / tb_n if tb_n else 0):.4f}", flush=True)
print(f"per-category cases flagged: {dict(cat)}", flush=True)
