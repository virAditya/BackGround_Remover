import os
import cv2

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(path, img)
