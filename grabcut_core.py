import cv2
import numpy as np

def estimate_bg_color_from_edges(bgr, border=10):
    h, w = bgr.shape[:2]
    strips = np.vstack([
        bgr[:border, :, :].reshape(-1, 3),
        bgr[h-border:, :, :].reshape(-1, 3),
        bgr[:, :border, :].reshape(-1, 3),
        bgr[:, w-border:, :].reshape(-1, 3),
    ])
    return np.median(strips, axis=0)  # robust to outliers

def heuristic_foreground_mask(bgr, color_threshold=35, illum_normalize=False):
    img = bgr.copy()
    if illum_normalize:
        # illumination normalization via CLAHE on L channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    bg = estimate_bg_color_from_edges(img)
    diff = np.linalg.norm(img.astype(np.float32) - bg.astype(np.float32), axis=2)
    mask = (diff > color_threshold).astype(np.uint8) * 255

    mask = cv2.medianBlur(mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def bbox_from_mask(mask, expand_px=10):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h, w = mask.shape[:2]
    x1 = max(0, x1 - expand_px); y1 = max(0, y1 - expand_px)
    x2 = min(w-1, x2 + expand_px); y2 = min(h-1, y2 + expand_px)
    return (x1, y1, x2 - x1, y2 - y1)

def apply_grabcut(bgr, rect, iter_count=5):
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgModel, fgModel, iter_count, cv2.GC_INIT_WITH_RECT)  # [web:1][web:17]
    out_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
    return out_mask

def cutout_from_mask(bgr, mask):
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba

def auto_grabcut(bgr, color_threshold=35, expand_px=10, iter_count=5, illum_normalize=False):
    coarse = heuristic_foreground_mask(bgr, color_threshold=color_threshold, illum_normalize=illum_normalize)
    rect = bbox_from_mask(coarse, expand_px=expand_px)
    if rect is None:
        # fallback: central rect
        h, w = bgr.shape[:2]
        rect = (w//8, h//8, int(w*3/4), int(h*3/4))
    gc_mask = apply_grabcut(bgr, rect, iter_count=iter_count)
    cutout = cutout_from_mask(bgr, gc_mask)
    return gc_mask, cutout
