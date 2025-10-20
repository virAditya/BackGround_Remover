import sys
import os
import cv2
import numpy as np

from PyQt5 import QtCore, QtWidgets  # For HiDPI attributes and QApplication [web:137]
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QDockWidget, QFormLayout, QSlider, QCheckBox, QSpinBox,
    QAction, QScrollArea, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
from PyQt5.QtCore import Qt

# Dark theme via stylesheet for compatibility with older qdarktheme releases [web:141]
try:
    import qdarktheme  # pip install pyqtdarktheme
    HAS_QDARK = True
except Exception:
    HAS_QDARK = False

from grabcut_core import auto_grabcut  # Robust GrabCut with mask seeding and inversion [web:1][web:110]
from utils import ensure_dirs, save_image  # Basic IO helpers

def np_to_qpixmap(img: np.ndarray) -> QPixmap:
    if img is None:
        return QPixmap()  # Empty pixmap if nothing to show [web:102]
    if len(img.shape) == 3 and img.shape[2] == 4:
        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, 4 * w, QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimg)  # RGBA path for cutouts [web:102]
    elif len(img.shape) == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)  # RGB path for input previews [web:102]
    else:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg)  # Grayscale fallback [web:102]

class DropLabel(QLabel):
    def __init__(self, on_file):
        super().__init__("Drop image here or Ctrl+O")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.on_file = on_file
        self.setStyleSheet("QLabel { border: 2px dashed #666; padding: 16px; }")  # Simple drop target styling [web:129]

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()  # Accept file drops [web:129]
        else:
            e.ignore()  # Ignore other drags [web:129]

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if not urls:
            return  # No files dropped [web:129]
        path = urls[0].toLocalFile()
        if os.path.isfile(path):
            self.on_file(path)  # Callback to load image [web:117]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Background Remover — GrabCut")
        ensure_dirs(["data/inputs", "data/outputs"])  # Prepare folders [web:77]

        # State
        self.bgr = None
        self.mask = None
        self.cutout = None
        self.current_path = None

        # Central scrollable preview with drag-and-drop [web:117][web:129]
        self.preview_label = DropLabel(self.load_image_from_path)
        self.preview = QScrollArea()
        self.preview.setWidgetResizable(True)
        self.preview.setWidget(self.preview_label)
        self.setCentralWidget(self.preview)

        # Sidebar dock similar to Streamlit sidebar [web:112][web:113]
        self.sidebar = QDockWidget("Controls", self)
        self.sidebar.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)  # Movable/floatable [web:119]
        self.sidebar.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.sidebar.setMinimumWidth(280)

        side_widget = QWidget()
        form = QFormLayout(side_widget)

        self.s_thresh = QSlider(Qt.Horizontal)
        self.s_thresh.setRange(10, 100)
        self.s_thresh.setValue(35)
        form.addRow("Color distance", self.s_thresh)  # Threshold for heuristic mask [web:1]

        self.s_expand = QSpinBox()
        self.s_expand.setRange(0, 80)
        self.s_expand.setValue(12)
        form.addRow("Expand bbox (px)", self.s_expand)  # Expand rect for coverage [web:1]

        self.s_iters = QSpinBox()
        self.s_iters.setRange(1, 10)
        self.s_iters.setValue(5)
        form.addRow("Iterations", self.s_iters)  # GrabCut iteration count [web:110]

        self.cb_illum = QCheckBox("Illumination normalization")
        form.addRow(self.cb_illum)  # CLAHE toggle used in core [web:110]

        self.cb_kmeans = QCheckBox("KMeans fallback")
        self.cb_kmeans.setChecked(True)
        form.addRow(self.cb_kmeans)  # KMeans proposal fallback toggle

        self.btn_process = QPushButton("Process")
        self.btn_process.clicked.connect(self.process)
        form.addRow(self.btn_process)  # Run pipeline button

        self.btn_save = QPushButton("Save PNG with alpha")
        self.btn_save.clicked.connect(self.save)
        self.btn_save.setEnabled(False)
        form.addRow(self.btn_save)  # Save output with alpha

        side_widget.setLayout(form)
        self.sidebar.setWidget(side_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.sidebar)  # Attach sidebar left [web:112]

        # Menu and shortcuts similar to productivity apps [web:125]
        open_action = QAction("Open...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_dialog)

        save_action = QAction("Save Result...", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save)

        process_action = QAction("Process", self)
        process_action.setShortcut("Ctrl+Enter")
        process_action.triggered.connect(self.process)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        run_menu = menubar.addMenu("&Run")
        run_menu.addAction(process_action)

        self.statusBar().showMessage("Ready. Drop an image or press Ctrl+O")  # Initial hint [web:125]

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.load_image_from_path(path)  # Open via dialog [web:102]

    def load_image_from_path(self, path):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            QMessageBox.warning(self, "Error", "Failed to load image.")  # Basic error handling [web:102]
            return
        self.current_path = path
        self.bgr = bgr
        self.mask = None
        self.cutout = None
        self.btn_save.setEnabled(False)
        self.preview_label.setText("")
        self.preview_label.setPixmap(np_to_qpixmap(bgr))  # Show input [web:102]
        self.statusBar().showMessage(f"Loaded: {os.path.basename(path)}")  # Status update [web:125]

    def process(self):
        if self.bgr is None:
            QMessageBox.information(self, "No image", "Load an image first.")  # Guard [web:102]
            return
        self.statusBar().showMessage("Processing...")
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)  # Busy cursor [web:125]
        try:
            mask, cut = auto_grabcut(
                self.bgr,
                color_threshold=int(self.s_thresh.value()),
                expand_px=int(self.s_expand.value()),
                iter_count=int(self.s_iters.value()),
                illum_normalize=bool(self.cb_illum.isChecked()),
                use_kmeans_fallback=bool(self.cb_kmeans.isChecked()),
            )  # Run robust pipeline [web:110]
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()  # Restore cursor [web:125]
        self.mask = mask
        self.cutout = cut
        self.preview_label.setPixmap(np_to_qpixmap(cv2.cvtColor(cut, cv2.COLOR_BGRA2RGBA)))  # Show result [web:102]
        self.btn_save.setEnabled(True)
        self.statusBar().showMessage("Done. Press Ctrl+S to save.")  # Status update [web:125]

    def save(self):
        if self.cutout is None:
            QMessageBox.information(self, "Nothing to save", "Process an image first.")  # Guard [web:102]
            return
        suggested = "cutout.png"
        if self.current_path:
            base = os.path.splitext(os.path.basename(self.current_path))[0]
            suggested = f"{base}_cutout.png"  # Suggest name from input [web:125]
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", f"data/outputs/{suggested}", "PNG (*.png)")
        if path:
            save_image(path, self.cutout)  # Save RGBA PNG [web:102]
            self.statusBar().showMessage(f"Saved: {path}")  # Status update [web:125]

def main():
    # HiDPI attributes for crisp rendering on Windows/HiDPI screens [web:137]
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)

    # Apply dark theme via stylesheet for maximum compatibility with older qdarktheme [web:141]
    if HAS_QDARK:
        try:
            app.setStyleSheet(qdarktheme.load_stylesheet("dark"))  # Backward-compatible API [web:141]
        except Exception:
            pass  # If it fails, continue without theming [web:141]

    w = MainWindow()
    w.resize(1300, 800)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
