import io
import os
import numpy as np
import streamlit as st
from PIL import Image
import cv2
from grabcut_core import auto_grabcut
from utils import ensure_dirs, save_image

st.set_page_config(page_title="Automatic Background Remover", page_icon="🧼", layout="centered")
st.title("Automatic Background Remover (GrabCut + Heuristics)")

ensure_dirs(["data/inputs", "data/outputs"])

with st.sidebar:
    st.header("Parameters")
    color_thresh = st.slider("Color distance threshold", 10, 100, 35, 1)
    expand_px = st.slider("Expand bbox (px)", 0, 60, 10, 1)
    iter_count = st.slider("GrabCut iterations", 1, 10, 5, 1)
    refine = st.checkbox("Illumination normalization", value=False)
    keep_size = st.checkbox("Keep original size", value=True)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])  # [web:96][web:99][web:106]

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)  # [web:99]
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Failed to decode image.")
        st.stop()

    st.subheader("Input")
    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), channels="RGB")

    if st.button("Remove Background"):
        mask, cutout = auto_grabcut(
            bgr,
            color_threshold=color_thresh,
            expand_px=expand_px,
            iter_count=iter_count,
            illum_normalize=refine,
        )

        st.subheader("Mask")
        st.image(mask, clamp=True)

        st.subheader("Result")
        rgba = cv2.cvtColor(cutout, cv2.COLOR_BGRA2RGBA)
        st.image(rgba, channels="RGBA")

        base = os.path.splitext(uploaded.name)[0]
        out_mask = f"data/outputs/{base}_mask.png"
        out_cut = f"data/outputs/{base}_cutout.png"
        save_image(out_mask, mask)
        save_image(out_cut, cutout)

        st.success("Saved to data/outputs/")
        st.download_button("Download Cutout (PNG)", data=cv2.imencode(".png", cutout)[1].tobytes(),
                           file_name=f"{base}_cutout.png", mime="image/png")
