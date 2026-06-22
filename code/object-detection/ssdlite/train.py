"""Train SSDLite320-MobileNetV3-Large on the TB detection set (from scratch).

A non-YOLO lightweight baseline (~2.2M params), trained under the same spirit as
the YOLO detectors: from scratch (no COCO pretrain), 100 epochs, fixed seed.
SSDLite is its own family (SSD multi-scale anchor head on a MobileNetV3 backbone
with depthwise-separable convs), so this is a representative cross-family point,
NOT a controlled head-to-head with the YOLO cohort. Its settings are disclosed.

NOTE ON RESOLUTION: SSDLite320's anchors are designed for 320x320 and the model's
internal transform resizes to that regardless of source size. We therefore train
and report it at its native 320 and disclose this, rather than forcing 512 (which
would fight the anchor design). The complexity/benchmark scripts reflect 320.

Outputs (mirrors the Ultralytics folders):
    results/results.csv          epoch, train_loss, val_map50, val_map
    results/weights/best.pt      state_dict with best val mAP@0.5
    results/weights/last.pt      most recent state_dict (for resume)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from data import YoloDetectionDataset, collate_fn
from transforms import build_transforms

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATASET = REPO / "dataset" / "object-detection"
NUM_CLASSES = 3  # background + ActiveTuberculosis + ObsoletePulmonaryTuberculosis


def build_model() -> torch.nn.Module:
    # from scratch to match the YOLO protocol: no COCO weights, no pretrained backbone
    return ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=NUM_CLASSES
    )


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds = model(images)
        preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
        gts = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
        metric.update(preds, gts)
    res = metric.compute()
    return float(res["map_50"]), float(res["map"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=str(HERE / "results"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output)
    (out / "weights").mkdir(parents=True, exist_ok=True)

    train_ds = YoloDetectionDataset(DATASET, "train", build_transforms(train=True))
    val_ds = YoloDetectionDataset(DATASET, "val", build_transforms(train=False))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True,
    )

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    csv_path = out / "results.csv"
    # resume if last.pt exists
    start_epoch, best_map50 = 0, -1.0
    last_pt = out / "weights" / "last.pt"
    if last_pt.exists():
        ckpt = torch.load(last_pt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_map50 = ckpt.get("best_map50", -1.0)
        print(f"resumed from epoch {start_epoch}")
    else:
        with open(csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(["epoch", "train_loss", "val_map50", "val_map"])

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        scheduler.step()
        train_loss = running / max(1, len(train_loader))

        map50, map5095 = evaluate(model, val_loader, device)
        with open(csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow([epoch + 1, f"{train_loss:.5f}", f"{map50:.5f}", f"{map5095:.5f}"])
        print(f"epoch {epoch+1}/{args.epochs}  loss={train_loss:.4f}  mAP50={map50:.4f}  mAP={map5095:.4f}")

        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "best_map50": best_map50,
        }
        torch.save(state, last_pt)
        if map50 > best_map50:
            best_map50 = map50
            torch.save(model.state_dict(), out / "weights" / "best.pt")
            print(f"  saved best (mAP50={map50:.4f})")

    print(f"done. best val mAP50={best_map50:.4f}")


if __name__ == "__main__":
    main()
