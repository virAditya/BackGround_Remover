import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QHBoxLayout, QSlider, QCheckBox, QSpinBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from grabcut_core import auto_grabcut
    # same API, now robust with mask-seeding and inversion
from utils import ensure_dirs, save_image

def to_qpixmap(bgr_or_rgba):
    if bgr_or_rgba is None:
        return QPixmap()
    if bgr_or_rgba.ndim == 3 and bgr_or_rgba.shape[2] == 4:
        h, w, _ = bgr_or_rgba.shape
        img = QImage(bgr_or_rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
    else:
        rgb = cv2.cvtColor(bgr_or_rgba, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Automatic Background Remover")
        ensure_dirs(["data/inputs", "data/outputs"])

        self.bgr = None
        self.mask = None
        self.cutout = None

        central = QWidget()
        layout = QVBoxLayout(central)

        self.input_label = QLabel("Input")
        self.input_label.setAlignment(Qt.AlignCenter)
        self.result_label = QLabel("Result")
        self.result_label.setAlignment(Qt.AlignCenter)

        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.proc_btn = QPushButton("Remove Background")
        self.save_btn = QPushButton("Save Result")
        self.save_btn.setEnabled(False)
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.proc_btn)
        btn_row.addWidget(self.save_btn)

        # Controls
        ctrl = QHBoxLayout()
        self.thresh = QSlider(Qt.Horizontal)
        self.thresh.setMinimum(10); self.thresh.setMaximum(100); self.thresh.setValue(35)
        self.expand = QSpinBox(); self.expand.setRange(0, 60); self.expand.setValue(12)
        self.iterc = QSpinBox(); self.iterc.setRange(1, 10); self.iterc.setValue(5)
        self.illum = QCheckBox("Illumination normalization")
        self.kmeans = QCheckBox("KMeans fallback"); self.kmeans.setChecked(True)

        ctrl.addWidget(QLabel("Threshold")); ctrl.addWidget(self.thresh)
        ctrl.addWidget(QLabel("Expand(px)")); ctrl.addWidget(self.expand)
        ctrl.addWidget(QLabel("Iters")); ctrl.addWidget(self.iterc)
        ctrl.addWidget(self.illum); ctrl.addWidget(self.kmeans)

        layout.addLayout(btn_row)
        layout.addLayout(ctrl)
        layout.addWidget(self.input_label)
        layout.addWidget(self.result_label)
        self.setCentralWidget(central)

        self.load_btn.clicked.connect(self.load_image)
        self.proc_btn.clicked.connect(self.process)
        self.save_btn.clicked.connect(self.save)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            self.input_label.setPixmap(to_qpixmap(self.bgr))

    def process(self):
        if self.bgr is None:
            return
        mask, cut = auto_grabcut(
            self.bgr,
            color_threshold=self.thresh.value(),
            expand_px=self.expand.value(),
            iter_count=self.iterc.value(),
            illum_normalize=self.illum.isChecked(),
            use_kmeans_fallback=self.kmeans.isChecked()
        )
        self.mask = mask
        self.cutout = cut
        self.result_label.setPixmap(to_qpixmap(cv2.cvtColor(cut, cv2.COLOR_BGRA2RGBA)))
        self.save_btn.setEnabled(True)

    def save(self):
        if self.cutout is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "data/outputs/cutout.png", "PNG (*.png)")
        if path:
            save_image(path, self.cutout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1100, 700)
    w.show()
    sys.exit(app.exec_())
