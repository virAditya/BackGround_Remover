import cv2
import numpy as np

# ---------- Heuristic and helpers ----------

def estimate_bg_color_from_edges(bgr, border=10, use_corners=True):
    h, w = bgr.shape[:2]
    if use_corners:
        c = border
        strips = np.vstack([
            bgr[:c, :c, :].reshape(-1, 3),
            bgr[:c, w-c:, :].reshape(-1, 3),
            bgr[h-c:, :c, :].reshape(-1, 3),
            bgr[h-c:, w-c:, :].reshape(-1, 3),
        ])
    else:
        strips = np.vstack([
            bgr[:border, :, :].reshape(-1, 3),
            bgr[h-border:, :, :].reshape(-1, 3),
            bgr[:, :border, :].reshape(-1, 3),
            bgr[:, w-border:, :].reshape(-1, 3),
        ])
    # Median is robust to outliers, handles light edge clutter
    return np.median(strips, axis=0).astype(np.float32)

def illumination_normalize(bgr, clip_limit=2.0, tile=(8, 8)):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def heuristic_foreground_mask(bgr, color_threshold=35, illum_normalize=False):
    img = bgr
    if illum_normalize:
        img = illumination_normalize(img, clip_limit=2.0, tile=(8, 8))
    bg = estimate_bg_color_from_edges(img, border=12, use_corners=True)
    diff = np.linalg.norm(img.astype(np.float32) - bg[None, None, :], axis=2)
    mask = (diff > float(color_threshold)).astype(np.uint8) * 255

    # Clean up small noise and fill holes
    mask = cv2.medianBlur(mask, 5)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    return mask

def kmeans_foreground_proposal(bgr, k=3):
    # Downscale for speed/stability
    h, w = bgr.shape[:2]
    scale = 256.0 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        small = bgr.copy()
    Z = small.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    ret, labels, centers = cv2.kmeans(Z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(small.shape[:2])

    # Choose cluster most dissimilar to border/corner colors
    corner_bg = estimate_bg_color_from_edges(small, border=8, use_corners=True)
    dists = np.linalg.norm(centers - corner_bg[None, :], axis=1)
    fg_label = int(np.argmax(dists))
    small_mask = (labels == fg_label).astype(np.uint8) * 255

    # Upscale back to original
    if small.shape[:2] != (h, w):
        mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask = small_mask
    # Smooth a bit
    mask = cv2.medianBlur(mask, 5)
    return mask

def maybe_invert(mask, majority_threshold=0.90):
    fg = int((mask > 0).sum())
    total = mask.size
    # If >90% of image is labeled foreground, we likely grabbed the background
    if fg > majority_threshold * total:
        return 255 - mask
    return mask

def bbox_from_mask(mask, expand_px=10):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h, w = mask.shape[:2]
    x1 = max(0, x1 - expand_px)
    y1 = max(0, y1 - expand_px)
    x2 = min(w - 1, x2 + expand_px)
    y2 = min(h - 1, y2 + expand_px)
    return (int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1))

# ---------- GrabCut seeding and execution ----------

def apply_grabcut_with_mask(bgr, coarse_mask, rect=None, iter_count=5):
    h, w = bgr.shape[:2]
    gc_mask = np.empty((h, w), np.uint8)
    gc_mask[:] = cv2.GC_PR_BGD
    gc_mask[coarse_mask > 0] = cv2.GC_PR_FGD

    # Strengthen foreground inside the rect if available
    if rect is not None:
        x, y, ww, hh = rect
        x2, y2 = x + ww, y + hh
        roi = gc_mask[y:y2, x:x2]
        roi_fg = (coarse_mask[y:y2, x:x2] > 0)
        roi[roi_fg] = cv2.GC_FGD
        gc_mask[y:y2, x:x2] = roi

    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc_mask, None, bgModel, fgModel, iter_count, cv2.GC_INIT_WITH_MASK)
    out_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return out_mask

def cutout_from_mask(bgr, mask):
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba

# ---------- Full pipeline ----------

def auto_grabcut(
    bgr,
    color_threshold=35,
    expand_px=12,
    iter_count=5,
    illum_normalize=False,
    use_kmeans_fallback=True
):
    # Step 1: heuristic mask
    coarse = heuristic_foreground_mask(
        bgr,
        color_threshold=color_threshold,
        illum_normalize=illum_normalize
    )

    # Step 2: maybe invert if it likely selected background
    coarse = maybe_invert(coarse, majority_threshold=0.90)

    # Step 3: if empty or too small, try k-means fallback
    if (coarse > 0).sum() < 200 and use_kmeans_fallback:
        km = kmeans_foreground_proposal(bgr, k=3)
        coarse = km

    # Step 4: bbox from coarse with dilation for coverage
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    coarse_dil = cv2.dilate(coarse, k, iterations=1)
    rect = bbox_from_mask(coarse_dil, expand_px=expand_px)

    # Step 5: final grabcut using mask seeding
    if rect is None:
        h, w = bgr.shape[:2]
        rect = (w // 8, h // 8, int(w * 3 / 4), int(h * 3 / 4))
    gc_mask = apply_grabcut_with_mask(bgr, coarse_dil, rect=rect, iter_count=iter_count)

    # Step 6: cutout
    cutout = cutout_from_mask(bgr, gc_mask)
    return gc_mask, cutout
