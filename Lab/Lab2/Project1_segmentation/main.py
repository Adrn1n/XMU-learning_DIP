import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import cv2
from pathlib import Path

EXTS = ("*.jpg", "*.png")
BLUR_KERNEL_SIZES = (15, 15)
CLIP_LIMIT = 1.25
TILE_GRID_SIZE = (8, 8)
MORPH_KERNEL_SIZES = (3, 3)
MORPH_ITERATIONS = 3


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")


PREDICTOR = SAM2ImagePredictor(
    build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "../SAM/sam2.1_hiera_tiny.pt",
        device=get_device(),
    )
)


def cvrt2uint8(img):
    if img.dtype == np.uint8:
        return img
    return cv2.convertScaleAbs(
        cv2.normalize(img, np.zeros_like(img), 0, 255, cv2.NORM_MINMAX)
    )


def seg_otsu(
    path, blur_ksize, clip_limit, tile_grid_size, morph_ksize, morph_iterations
):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        img = cvrt2uint8(img)
        blur = cv2.GaussianBlur(cvrt2uint8(img), blur_ksize, 0)
        if blur.ndim == 2:
            gray = blur
        else:
            l, a, b = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2LAB))
            l = cv2.createCLAHE(
                clipLimit=clip_limit, tileGridSize=tile_grid_size
            ).apply(l)
            gray = cv2.cvtColor(
                cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR),
                cv2.COLOR_BGR2GRAY,
            )
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones(morph_ksize, np.uint8)
        otsu_mask = cv2.morphologyEx(
            otsu_mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations
        )
        otsu_mask = cv2.dilate(otsu_mask, kernel, iterations=morph_iterations)
        seg_result = cv2.bitwise_and(img, img, mask=otsu_mask)
        return seg_result, otsu_mask
    return None, None


def seg_sam(path, predictor):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img)
        masks, _, _ = predictor.predict()
        if masks is not None and len(masks) > 0:
            masks = np.array(masks, dtype=np.uint8) * 255
            seg_results = [cv2.bitwise_and(img, img, mask=mask) for mask in masks]
            return seg_results, masks
    return None, None


if __name__ == "__main__":
    inp = Path(".")
    out = Path("output")
    out.mkdir(exist_ok=True)
    files = []
    for e in EXTS:
        files.extend(sorted(inp.glob(e)))
    if files:
        for f in files:
            name, ext = f.stem, f.suffix
            res, mask = seg_otsu(
                f,
                BLUR_KERNEL_SIZES,
                CLIP_LIMIT,
                TILE_GRID_SIZE,
                MORPH_KERNEL_SIZES,
                MORPH_ITERATIONS,
            )
            if res is None:
                print(f"Otsu failed to process {f}")
            else:
                cv2.imwrite(str(out / f"{name}_otsu_segmented{ext}"), res)
                cv2.imwrite(str(out / f"{name}_otsu_mask{ext}"), mask)
            res, mask = seg_sam(f, PREDICTOR)
            if res is None:
                print(f"SAM failed to process {f}")
            else:
                Path(out / "sam").mkdir(exist_ok=True)
                for i, (r, m) in enumerate(zip(res, mask)):
                    cv2.imwrite(str(out / "sam" / f"{name}_segmented_{i}{ext}"), r)
                    cv2.imwrite(str(out / "sam" / f"{name}_mask_{i}{ext}"), m)
    else:
        print(f"No image files found in {inp.resolve()}")
