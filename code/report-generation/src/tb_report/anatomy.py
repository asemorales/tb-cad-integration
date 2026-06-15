"""Bounding box -> anatomical region string (3x3 grid).

Laterality follows radiological convention. On a correctly oriented frontal
(PA/AP) chest radiograph the patient is viewed face-to-face, so the patient's
right side appears on the viewer's left. The column label therefore reports the
patient's side, not the image side: image-left maps to "right" and image-right
maps to "left". This assumes a properly oriented frontal projection.
"""

from tb_report.schema import Location

BBox = tuple[float, float, float, float]


def bbox_to_location(bbox: BBox, image_size: tuple[int, int]) -> Location:
    """Map a pixel-space bbox to one of the 9 grid cells by its center point.

    Columns use radiological laterality (patient side), not image side.
    """
    x0, y0, x1, y1 = bbox
    width, height = image_size
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    # Frontal CXR: image-left is the patient's right, and image-right is the
    # patient's left. Center is unchanged.
    if cx < width / 3:
        col = "right"
    elif cx < 2 * width / 3:
        col = "center"
    else:
        col = "left"

    if cy < height / 3:
        row = "upper"
    elif cy < 2 * height / 3:
        row = "middle"
    else:
        row = "lower"

    return f"{row}_{col}"  # type: ignore[return-value]
