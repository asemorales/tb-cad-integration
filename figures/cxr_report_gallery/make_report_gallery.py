"""Rebuild the report-gallery figure reproducibly from TBX11K annotations.

Each panel pairs a chest radiograph (with ground-truth TB boxes and a ground-truth
class label) with the report the pipeline actually emits. The report text is the
real output of ``tb_report.generate_report`` on a record built from the boxes, so
the figure cannot drift from the code. Zones use the package's radiological
laterality (image-left is the patient's right). Cases are selected deterministically
from the detection labels, so re-running reproduces the same figure. Font is Inter.
"""

import sys
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(ROOT / "figures"))
sys.path.insert(0, str(ROOT / "code/report-generation/src"))
from paper_style import apply_style, BBOX_COLOR  # noqa: E402
from tb_report import generate_report, bbox_to_location  # noqa: E402
from tb_report.schema import DetectorOutput, ImageClassification, Region  # noqa: E402

apply_style()

DET = ROOT / "dataset/object-detection"
CLS = ROOT / "dataset/classification"

# Single ground-truth box color; activity is shown by line style instead of hue
# (active = solid, latent = dashed).
BOX_LS = {"active": "-", "latent": "--"}
# Map a detection box kind to the schema region type.
REGION_TYPE = {"active": "active_tb", "latent": "latent_tb"}
# Probabilities are not shown; a representative one-hot keeps the record valid.
_PROBS = {"healthy": {"healthy": 0.95, "sick_non_tb": 0.03, "tb": 0.02},
          "sick_non_tb": {"healthy": 0.05, "sick_non_tb": 0.90, "tb": 0.05},
          "tb": {"healthy": 0.03, "sick_non_tb": 0.07, "tb": 0.90}}


def load_boxes(stem):
    """Return (image_path, (W,H), [(kind, (x0,y0,x1,y1)), ...]) for a detection image."""
    for split in ("train", "val"):
        lp = DET / "labels" / split / f"{stem}.txt"
        ip = DET / "images" / split / f"{stem}.png"
        if lp.exists():
            W, H = Image.open(ip).size
            boxes = []
            for line in lp.read_text().strip().splitlines():
                c, xc, yc, w, h = (float(v) for v in line.split())
                x0, y0 = (xc - w / 2) * W, (yc - h / 2) * H
                x1, y1 = (xc + w / 2) * W, (yc + h / 2) * H
                boxes.append(("active" if int(c) == 0 else "latent", (x0, y0, x1, y1)))
            return ip, (W, H), boxes
    raise FileNotFoundError(stem)


def scan():
    """Categorize every TB image by its active/latent box composition."""
    cats = {}
    for split in ("train", "val"):
        for lp in sorted((DET / "labels" / split).glob("tb*.txt")):
            _, _, boxes = load_boxes(lp.stem)
            na = sum(b[0] == "active" for b in boxes)
            nl = sum(b[0] == "latent" for b in boxes)
            cats[lp.stem] = (na, nl)
    return cats


def pick(cats, predicate):
    for stem in sorted(cats):
        if predicate(*cats[stem]):
            return stem
    raise LookupError


cats = scan()
active_single = pick(cats, lambda a, l: a == 1 and l == 0)
latent_single = pick(cats, lambda a, l: l == 1 and a == 0)
both2 = [s for s in sorted(cats) if cats[s][0] >= 1 and cats[s][1] >= 1 and sum(cats[s]) == 2]
both3 = pick(cats, lambda a, l: a >= 1 and l >= 1 and a + l == 3)
active_multi = pick(cats, lambda a, l: a >= 3 and l == 0)

cls_img = lambda sub: sorted((CLS / "test" / sub).glob("*.png"))[0]

# Panel order: healthy, sick, active-only, latent-only, both, both, mixed-3, active-multi
CASES = [
    ("healthy", cls_img("healthy"), "No TB"),
    ("sick_non_tb", cls_img("sick-non-tb"), "Sick non-TB"),
    (active_single, None, "TB: active lesions only"),
    (latent_single, None, "TB: latent lesions only"),
    (both2[0], None, "TB: active and latent lesions"),
    (both2[1], None, "TB: active and latent lesions"),
    (both3, None, "TB: active and latent lesions"),
    (active_multi, None, "TB: active lesions only"),
]


def build_record(label, regions):
    """Build a DetectorOutput from a class label and (kind, zone) ground-truth boxes."""
    return DetectorOutput(
        image_classification=ImageClassification(predicted_label=label, probabilities=_PROBS[label]),
        regions=[Region(type=REGION_TYPE[k], confidence_band="high", location=z) for k, z in regions],
    )


def report_sections(label, regions):
    """Return (findings_lines, impression, recommendation) from the real pipeline output."""
    text = generate_report(build_record(label, regions))
    findings, impression, rec = [], "", ""
    in_findings = False
    for line in text.splitlines():
        if line == "FINDINGS:":
            in_findings = True
        elif line.startswith("IMPRESSION:"):
            in_findings = False
            impression = line.split(":", 1)[1].strip()
        elif line.startswith("RECOMMENDATION:"):
            rec = line.split(":", 1)[1].strip()
        elif in_findings:
            findings.append(line)
    return findings, impression, rec


def draw_text(ax, findings, impression, rec):
    ax.axis("off")
    y, dy, wrap = 0.98, 0.052, 34
    def block(title, body_lines):
        nonlocal y
        ax.text(0.0, y, title, transform=ax.transAxes, va="top", ha="left",
                fontsize=12, fontweight="bold")
        y -= dy
        for bl in body_lines:
            for wl in textwrap.wrap(bl, wrap) or [""]:
                ax.text(0.0, y, wl, transform=ax.transAxes, va="top", ha="left", fontsize=11)
                y -= dy
        y -= dy * 0.5
    block("FINDINGS", findings)
    block("IMPRESSION", [impression])
    block("RECOMMENDATION", [rec])


fig = plt.figure(figsize=(14, 16))
gs = fig.add_gridspec(4, 4, width_ratios=[1, 1.15, 1, 1.15], hspace=0.12, wspace=0.05)

for idx, (stem, override_path, gt_label) in enumerate(CASES):
    r, pc = divmod(idx, 2)
    ax_img = fig.add_subplot(gs[r, pc * 2])
    ax_txt = fig.add_subplot(gs[r, pc * 2 + 1])

    if override_path is not None:        # healthy / sick: no boxes
        img = Image.open(override_path).convert("L")
        regions = []
        label = stem
    else:
        ip, (W, H), boxes = load_boxes(stem)
        img = Image.open(ip).convert("L")
        regions = [(k, bbox_to_location(b, (W, H))) for k, b in boxes]
        label = "tb"

    ax_img.imshow(img, cmap="gray")
    ax_img.set_xticks([]); ax_img.set_yticks([])
    if override_path is None:
        for k, b in load_boxes(stem)[2]:
            x0, y0, x1, y1 = b
            ax_img.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                               edgecolor=BBOX_COLOR, ls=BOX_LS[k], lw=2))
    ax_img.text(0.03, 0.97, gt_label, transform=ax_img.transAxes, va="top", ha="left",
                fontsize=11, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75, edgecolor="none"))

    findings, impression, rec = report_sections(label, regions)
    draw_text(ax_txt, findings, impression, rec)

# Capitalized legend: one color, line style distinguishes activity
handles = [plt.Line2D([0], [0], color=BBOX_COLOR, lw=2.5, ls="-", label="Active TB"),
           plt.Line2D([0], [0], color=BBOX_COLOR, lw=2.5, ls="--", label="Latent TB")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           fontsize=12, bbox_to_anchor=(0.5, 0.005))

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.04)
out = ROOT / "figures/cxr_report_gallery/cxr_report_gallery.png"
fig.savefig(out, dpi=150)
print("wrote", out)
print("cases:", [c[0] for c in CASES])
