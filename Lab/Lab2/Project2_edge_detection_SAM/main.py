import numpy as np
import cv2
import csv
from pathlib import Path

EXTS = ("*.jpg",)
CANNY_THRESHOLD1 = 100
CANNY_THRESHOLD2 = 175
AREA_THRESHOLD = 256
STABILITY_THRESHOLD = 0.85
KERNEL_SIZE = (3, 3)
MASK_EXTS = ("*.png",)


def cvrt2uint8(img):
    if img.dtype == np.uint8:
        return img
    return cv2.convertScaleAbs(
        cv2.normalize(img, np.zeros_like(img), 0, 255, cv2.NORM_MINMAX)
    )


def edge_detect_canny(path, threshold1, threshold2):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        img = cvrt2uint8(img)
        if img.ndim != 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        canny_edges = cv2.Canny(gray, threshold1, threshold2)
        return canny_edges
    return None


def edge_detect_sam(
    path,
    mask_folder,
    mask_exts,
    area_threshold,
    stability_threshold,
    kernal_size,
    meta_data=None,
):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.ndim != 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        sam_edges = np.zeros_like(gray)
        mask_files = []
        if meta_data is None:
            for m_ext in mask_exts:
                mask_files.extend(sorted(mask_folder.glob(m_ext)))
        else:
            with open(meta_data, "r") as meta_file:
                reader = csv.DictReader(meta_file)
                for row in reader:
                    stability_score = float(row.get("stability_score", 0))
                    area = float(row.get("area", 0))
                    if (
                        area >= area_threshold
                        and stability_score >= stability_threshold
                    ):
                        mask_id = row["id"]
                        mask_path = mask_folder / f"{mask_id}.png"
                        if mask_path.exists():
                            mask_files.append(mask_path)
                        else:
                            print(f"Warning: Mask file {mask_path} not found.")
        if mask_files:
            for mask_file in mask_files:
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernal_size)
                    mask_edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
                    sam_edges = cv2.bitwise_or(sam_edges, mask_edge)
                else:
                    print(f"Warning: Failed to read mask {mask_file}")
            return sam_edges
    return None


if __name__ == "__main__":
    inp = Path(".")
    out = Path("output")
    out.mkdir(exist_ok=True)
    files = []
    for e in EXTS:
        files.extend(inp.glob(e))
    if files:
        for f in files:
            name, ext = f.stem, f.suffix
            edges = edge_detect_canny(f, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
            if edges is None:
                print(f"Canny failed to process {f}")
            else:
                cv2.imwrite(str(out / f"{name}_canny{ext}"), edges)
            folder = f.parent / f"{name}_masks"
            meta = folder / "metadata.csv"
            edges = edge_detect_sam(
                f,
                folder,
                MASK_EXTS,
                AREA_THRESHOLD,
                STABILITY_THRESHOLD,
                KERNEL_SIZE,
                meta if meta.exists() else None,
            )
            if edges is None:
                print(f"SAM edge detection failed to process {f}")
            else:
                cv2.imwrite(str(out / f"{name}_sam_edges{ext}"), edges)
    else:
        print(f"No image files found in {inp.resolve()}")
