import cv2
import numpy as np
from pathlib import Path

EXTS = ("*.jpg",)
KERNEL_SIZES = 3


def cvrt2uint8(img):
    if img.dtype == np.uint8:
        return img
    return cv2.convertScaleAbs(
        cv2.normalize(img, np.zeros_like(img), 0, 255, cv2.NORM_MINMAX)
    )


def process(path, ksize):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        img = cvrt2uint8(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        med = cv2.medianBlur(gray, ksize)
        mean = cv2.blur(gray, (ksize, ksize))
        return med, mean
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
            res = process(f, KERNEL_SIZES)
            if res:
                name, ext = f.stem, f.suffix
                cv2.imwrite(str(out / f"{name}_median{ext}"), res[0])
                cv2.imwrite(str(out / f"{name}_mean{ext}"), res[1])
            else:
                print("Failed to process", f)
    else:
        print("No image files found in", inp)
