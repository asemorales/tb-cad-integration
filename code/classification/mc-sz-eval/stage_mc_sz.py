"""Unzip the NLM Montgomery + Shenzhen CXR sets and stage them as binary test dirs.

Both sets are public (openi.nlm.nih.gov) and encode the TB label in the filename
suffix before the extension: ``*_0.png`` = normal, ``*_1.png`` = TB. This stages
each into the standard mlx classification test layout, using only the two classes
present in these binary sets:

    <out>/mc/test/healthy/*.png   (Montgomery normals, _0)
    <out>/mc/test/tb/*.png        (Montgomery TB,      _1)
    <out>/sz/test/healthy/*.png   (Shenzhen  normals,  _0)
    <out>/sz/test/tb/*.png        (Shenzhen  TB,       _1)

healthy maps to class index 0 and tb to 2 under the fixed cohort class order
(0=healthy, 1=sick-non-tb, 2=tb); the absent sick-non-tb folder is simply skipped
by the dataset loader, so predict_mlx.py / predict_flipr.py run unchanged.

Usage:
    python stage_mc_sz.py <external_dir>
        # expects <external_dir>/NLM-MontgomeryCXRSet.zip and ChinaSet_AllFiles.zip
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

SETS = {
    "mc": "NLM-MontgomeryCXRSet.zip",
    "sz": "ChinaSet_AllFiles.zip",
}

# Expected byte sizes (openi.nlm.nih.gov), used to stage only fully-downloaded zips.
EXPECTED_BYTES = {
    "NLM-MontgomeryCXRSet.zip": 616853875,
    "ChinaSet_AllFiles.zip": 3770205534,
}


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def stage_one(external_dir: Path, key: str, zip_name: str) -> tuple[int, int]:
    zip_path = external_dir / zip_name
    want = EXPECTED_BYTES.get(zip_name)
    have = zip_path.stat().st_size if zip_path.exists() else 0
    if want is not None and have != want:
        print(f"{key}: SKIP, zip not complete ({have}/{want} bytes): {zip_path}")
        return -1, -1

    extract_dir = external_dir / "_extracted" / key
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    out_root = external_dir / key / "test"
    healthy_dir = out_root / "healthy"
    tb_dir = out_root / "tb"
    healthy_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    n_healthy = n_tb = 0
    for png in extract_dir.rglob("*.png"):
        # Only CXR images live under a CXR_png folder; skip masks/other assets.
        if "CXR_png" not in png.parts:
            continue
        stem = png.stem  # e.g. MCUCXR_0001_0
        suffix = stem.rsplit("_", 1)[-1]
        if suffix == "0":
            _link_or_copy(png, healthy_dir / png.name)
            n_healthy += 1
        elif suffix == "1":
            _link_or_copy(png, tb_dir / png.name)
            n_tb += 1
    return n_healthy, n_tb


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python stage_mc_sz.py <external_dir>")
    external_dir = Path(sys.argv[1])
    for key, zip_name in SETS.items():
        n_healthy, n_tb = stage_one(external_dir, key, zip_name)
        if n_healthy >= 0:
            print(f"{key}: staged healthy(normal)={n_healthy}  tb={n_tb}  -> {external_dir / key / 'test'}")


if __name__ == "__main__":
    main()
