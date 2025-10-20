# Automatic Background Remover (GrabCut + Heuristics)

CPU-friendly background removal using OpenCV GrabCut with automatic bounding box initialization from simple color/position heuristics.

## Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Run (Web - Streamlit)
streamlit run app_streamlit.py

## Run (Desktop - PyQt)
python app_pyqt.py

## Outputs
Saved to data/outputs/ as PNG with alpha channel.

## Notes
- Illumination normalization (CLAHE) can help under uneven lighting.
- If heuristics fail, app falls back to a central rectangle before GrabCut.
