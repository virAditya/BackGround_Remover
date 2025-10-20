import cv2
import numpy as np

# ========== Saliency and Edge Priors ==========

def compute_saliency_map(bgr):
    """
    Compute spectral residual saliency map using OpenCV's built-in algorithm.
    Returns a normalized grayscale map [0, 255] where bright = salient.
    """
    try:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()  # [web:164][web:161]
        success, sal_map = saliency.computeSaliency(bgr)
        if not success:
            raise RuntimeError("Saliency computation failed")
        sal_map = (sal_map * 255).astype(np.uint8)  # Scale [0,1] to [0,255] [web:164]
        return sal_map
    except Exception:
        # Fallback: use simple gradient magnitude as pseudo-saliency
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(grad_x**2 + grad_y**2)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return mag

def compute_edge_map(bgr, low=50, high=150):
    """
    Compute Canny edges and dilate for boundary closure.
    Returns a binary edge map [0, 255].
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)  # [web:167][web:173]
    edges = cv2.Canny(gray, low, high)  # [web:173][web:167]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)  # Thicken edges for closure
    return edges

def estimate_bg_color_from_border(bgr, border=15):
    """
    Robust background color estimation from image corners.
    """
    h, w = bgr.shape[:2]
    c = border
    strips = np.vstack([
        bgr[:c, :c, :].reshape(-1, 3),
        bgr[:c, w-c:, :].reshape(-1, 3),
        bgr[h-c:, :c, :].reshape(-1, 3),
        bgr[h-c:, w-c:, :].reshape(-1, 3),
    ])
    return np.median(strips, axis=0).astype(np.float32)

def color_contrast_mask(bgr, color_threshold=35, illum_normalize=False):
    """
    Generate a mask based on color distance from estimated background.
    """
    img = bgr.copy()
    if illum_normalize:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    bg = estimate_bg_color_from_border(img, border=15)
    diff = np.linalg.norm(img.astype(np.float32) - bg[None, None, :], axis=2)
    mask = (diff > float(color_threshold)).astype(np.uint8) * 255
    
    # Clean noise
    mask = cv2.medianBlur(mask, 5)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask

# ========== Multi-Prior Fusion and Seed Generation ==========

def fuse_priors(color_mask, saliency_map, edge_map):
    """
    Fuse color, saliency, and edge information to produce confident foreground/background seeds.
    Returns: sure_fg, sure_bg, probable_fg masks.
    """
    h, w = color_mask.shape
    
    # Adaptive thresholds for saliency
    sal_high = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # [web:164]
    sal_low = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)[1]
    
    # Sure foreground: high saliency AND strong color contrast
    sure_fg = cv2.bitwise_and(sal_high, color_mask)
    
    # Enhance with edge closure: if a region has edges around it, keep interior
    contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_closed = np.zeros_like(edge_map)
    cv2.drawContours(edge_closed, contours, -1, 255, thickness=cv2.FILLED)
    sure_fg = cv2.bitwise_and(sure_fg, edge_closed)
    
    # Clean sure_fg
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, k, iterations=2)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_CLOSE, k, iterations=2)
    
    # Probable foreground: broader saliency OR color contrast (union)
    probable_fg = cv2.bitwise_or(sal_low, color_mask)
    probable_fg = cv2.morphologyEx(probable_fg, cv2.MORPH_CLOSE, k, iterations=1)
    
    # Sure background: force border frame + low saliency + low color contrast
    border_width = 20
    sure_bg = np.zeros((h, w), np.uint8)
    sure_bg[:border_width, :] = 255
    sure_bg[h-border_width:, :] = 255
    sure_bg[:, :border_width] = 255
    sure_bg[:, w-border_width:] = 255
    
    # Add regions with very low saliency and color contrast
    low_sal = cv2.bitwise_not(sal_low)
    low_color = cv2.bitwise_not(color_mask)
    sure_bg_inner = cv2.bitwise_and(low_sal, low_color)
    sure_bg = cv2.bitwise_or(sure_bg, sure_bg_inner)
    
    return sure_fg, sure_bg, probable_fg

def create_gc_mask(sure_fg, sure_bg, probable_fg):
    """
    Create a GrabCut initialization mask with all four label classes.
    GC_BGD=0, GC_FGD=1, GC_PR_BGD=2, GC_PR_FGD=3
    """
    h, w = sure_fg.shape
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)  # Default: probable background [web:17][web:152]
    
    # Probable foreground regions
    gc_mask[probable_fg > 0] = cv2.GC_PR_FGD  # [web:17][web:149]
    
    # Sure background (highest priority in overlap)
    gc_mask[sure_bg > 0] = cv2.GC_BGD  # [web:17][web:149]
    
    # Sure foreground (highest priority)
    gc_mask[sure_fg > 0] = cv2.GC_FGD  # [web:17][web:149]
    
    return gc_mask

# ========== GrabCut Execution ==========

def apply_grabcut_with_seeds(bgr, gc_mask, iter_count=5, refine=True):
    """
    Run GrabCut with GC_INIT_WITH_MASK using the seed mask, then optionally refine with GC_EVAL.
    Returns final binary mask [0, 255].
    """
    h, w = bgr.shape[:2]
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    
    # Initial GC with mask seeds [web:17][web:1][web:149]
    cv2.grabCut(bgr, gc_mask, None, bgModel, fgModel, iter_count, cv2.GC_INIT_WITH_MASK)
    
    # Optional refinement with GC_EVAL (preserves user seeds, refines boundaries) [web:9][web:152]
    if refine:
        cv2.grabCut(bgr, gc_mask, None, bgModel, fgModel, 2, cv2.GC_EVAL)
    
    # Extract final foreground mask [web:17][web:1]
    out_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    
    # Post-process: remove small noise and fill small holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out_mask = cv2.morphologyEx(out_mask, cv2.MORPH_OPEN, k, iterations=1)
    out_mask = cv2.morphologyEx(out_mask, cv2.MORPH_CLOSE, k, iterations=2)
    
    return out_mask

def cutout_from_mask(bgr, mask):
    """
    Apply mask to create BGRA cutout with alpha channel.
    """
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba

# ========== Main Pipeline ==========

def auto_grabcut(
    bgr,
    color_threshold=35,
    expand_px=12,
    iter_count=5,
    illum_normalize=False,
    use_kmeans_fallback=False  # Kept for API compatibility but not needed with strong priors
):
    """
    Automatic GrabCut background removal with multi-prior initialization.
    
    Steps:
    1. Compute saliency map (spectral residual)
    2. Compute edge map (Canny + dilation)
    3. Compute color contrast mask
    4. Fuse priors to create confident foreground/background seeds
    5. Initialize GrabCut with GC_INIT_WITH_MASK
    6. Refine with GC_EVAL
    7. Return mask and BGRA cutout
    """
    h, w = bgr.shape[:2]
    
    # Step 1: Compute priors [web:164][web:173][web:167]
    saliency_map = compute_saliency_map(bgr)
    edge_map = compute_edge_map(bgr, low=50, high=150)
    color_mask = color_contrast_mask(bgr, color_threshold=color_threshold, illum_normalize=illum_normalize)
    
    # Step 2: Fuse priors into seeds
    sure_fg, sure_bg, probable_fg = fuse_priors(color_mask, saliency_map, edge_map)
    
    # Step 3: Create GC mask with all four label classes [web:17][web:149][web:152]
    gc_mask = create_gc_mask(sure_fg, sure_bg, probable_fg)
    
    # Step 4: Run GrabCut with mask initialization [web:17][web:1][web:149]
    final_mask = apply_grabcut_with_seeds(bgr, gc_mask, iter_count=iter_count, refine=True)
    
    # Step 5: Create cutout
    cutout = cutout_from_mask(bgr, final_mask)
    
    return final_mask, cutout

