import numpy as np
import cv2
from pathlib import Path

EXTS = ("*.jpg", "*.tif")
KERNEL_SIZES = 3
MASK_RANGE = (180, 255)


def cvrt2uint8(img):
    if img.dtype == np.uint8:
        return img
    return cv2.convertScaleAbs(
        cv2.normalize(img, np.zeros_like(img), 0, 255, cv2.NORM_MINMAX)
    )


def smooth(path, ksize):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        img = cvrt2uint8(img)
        if img.ndim != 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        return cv2.medianBlur(gray, ksize)
    return None


def remove_watermark(path, ksize, mask_range):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        img = cvrt2uint8(img)
        if img.ndim != 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        return cv2.inpaint(
            img,
            cv2.dilate(
                cv2.inRange(gray, mask_range[0], mask_range[1]),
                np.ones((ksize, ksize), np.uint8),
                iterations=1,
            ),
            ksize,
            cv2.INPAINT_TELEA,
        )
    return None


def contrast_enhance(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        img = cvrt2uint8(img)
        if img.ndim != 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        return cv2.equalizeHist(gray)
    return None


if __name__ == "__main__":
    inp = Path("Image")
    out = Path("./output")
    out.mkdir(exist_ok=True)
    files = []
    for e in EXTS:
        files.extend(sorted(inp.glob(e)))
    if files:
        for f in files:
            name, ext = f.stem, f.suffix
            if name == "house":
                res = smooth(f, KERNEL_SIZES)
            elif name == "night":
                res = remove_watermark(f, KERNEL_SIZES, MASK_RANGE)
            elif name == "old":
                res = contrast_enhance(f)
            else:
                res = None
            if res is None:
                print(f"Failed to process {f}")
            else:
                cv2.imwrite(str(out / f"{name}{ext}"), res)
    else:
        print(f"No image files found in {inp.resolve()}")
