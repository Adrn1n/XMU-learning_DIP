import cv2
import numpy as np
from pathlib import Path

EXTS = ("*.jpg",)
ALPHA = 1
KERNEL_SIZES = 3


def cvrt2uint8(img):
    if img.dtype == np.uint8:
        return img
    return cv2.convertScaleAbs(
        cv2.normalize(img, np.zeros_like(img), 0, 255, cv2.NORM_MINMAX)
    )


def process(path, alpha, ksize):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        img = cvrt2uint8(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        chf = gray.astype(np.float32)
        return np.clip(
            chf - alpha * cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize), 0, 255
        ).astype(np.uint8)
    return None


if __name__ == "__main__":
    inp = Path(".")
    out = Path("./output")
    out.mkdir(exist_ok=True)
    files = []
    for e in EXTS:
        files.extend(sorted(inp.glob(e)))
    if files:
        for f in files:
            res = process(f, ALPHA, KERNEL_SIZES)
            if res is None:
                print("Failed to process", f)
            else:
                name, ext = f.stem, f.suffix
                cv2.imwrite(str(out / f"{name}_lapSharp{ext}"), res)
    else:
        print("No image files found in", inp)
