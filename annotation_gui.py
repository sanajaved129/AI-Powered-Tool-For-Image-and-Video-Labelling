

import sys
import os
import csv
import numpy as np
import torch
import subprocess
from PIL import Image
import torchvision.transforms as T
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout,
    QWidget, QLabel, QComboBox, QListWidget, QMessageBox, QCheckBox, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QMouseEvent
from PyQt5.QtCore import Qt, QPoint, QProcess 

from inference_eval import Predictor, Evaluator, make_overlay, load_pred_mask_from_outputs
from heatmap_tool import build_heatmap_overlay 

from tkinter import Tk, filedialog



class ImageLoaderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" IMAGE SEGMENTATION ")
        self.setGeometry(100, 100, 900, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Load from Folder"])
        self.mode_selector.currentIndexChanged.connect(self.handle_mode_change)
        self.layout.addWidget(self.mode_selector)
        
        # Button to select source folder
        self.select_button = QPushButton("Select Source")
        self.select_button.clicked.connect(self.select_source)
        self.layout.addWidget(self.select_button)
        
        # List widget to display image paths
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(150)
        self.image_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.image_list.setWordWrap(False)
        self.image_list.itemClicked.connect(self.display_image)
        self.layout.addWidget(self.image_list)
        
        # Tool selector for drawing mode
        self.tool_selector = QComboBox()
        self.tool_selector.addItems(["-- Choose Drawing Type --","Polygon", "Freehand"])
        self.tool_selector.currentTextChanged.connect(self.change_draw_mode)
        self.layout.addWidget(self.tool_selector)
        
        # --- Canvases row (left: RAW w/ annotation tools, right: PREVIEW overlay) ---
        canvases = QHBoxLayout()

        # Left: annotation canvas on raw image
        self.canvas = AnnotationCanvas(self)
        self.canvas.setFixedSize(512, 512)
        canvases.addWidget(self.canvas)

        # Right: preview overlay canvas (read-only)
        self.preview_canvas = QLabel()
        self.preview_canvas.setFixedSize(512, 512)
        self.preview_canvas.setAlignment(Qt.AlignCenter)
        self.preview_canvas.setStyleSheet("background:#fff; border:1px solid #ccc;")  # white
        self.preview_canvas.setScaledContents(True)
        canvases.addWidget(self.preview_canvas)

        self.layout.addLayout(canvases)

        # Evaluation label
        self.eval_label = QLabel("Dice: N/A | IoU: N/A | Precision: N/A | Accuracy: N/A")
        self.layout.addWidget(self.eval_label)
        

        # Device setup for model inference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  
        
        # Undo/Reset buttons
        undo_reset_layout = QHBoxLayout()

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.canvas.undo)
        undo_reset_layout.addWidget(self.undo_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.canvas.reset)
        undo_reset_layout.addWidget(self.reset_button)

        self.layout.addLayout(undo_reset_layout)
        # -----------------
        
        # Save button for annotations
        self.save_button = QPushButton("Save Annotation")
        self.save_button.clicked.connect(self.save_annotation)
        self.layout.addWidget(self.save_button)
        

        self.central_widget.setLayout(self.layout)
        
        self.path_to_label = {}  # maps full path to class label

        self.image_paths = []
        
        # Button to convert annotations to masks
        self.convert_button = QPushButton("Convert Annotations to Masks")
        self.convert_button.clicked.connect(self.convert_annotations)
        self.layout.addWidget(self.convert_button)
        
        # Button to generate masks from green boxes
        self.generate_btn = QPushButton("Generate Masks And Train Model")
        self.generate_btn.clicked.connect(self.generate_green_box_masks)
        self.layout.addWidget(self.generate_btn)

        # Temporal inference button
        self.btn_run_temporal = QPushButton("Temporal Inference")
        self.btn_run_temporal.clicked.connect(self.run_temporal_inference_from_annotations)
        self.layout.addWidget(self.btn_run_temporal)

        # Prediction toggle (master gate)
        self.prediction_toggle = QCheckBox("Show Model Prediction")
        self.layout.addWidget(self.prediction_toggle)
        self.prediction_toggle.setChecked(False)      # start OFF
        self.prediction_toggle.setEnabled(True)      # gate is usable without any model
        self.prediction_toggle.toggled.connect(self._on_prediction_gate_toggled)

        
        
        # ------ Model selector | GT Source -------------- 
        
        # create model selector once
        self.model_selector = QComboBox()
        self.refresh_model_selector()
        self.model_selector.setToolTip("Select a model checkpoint to load")

        # create GT source selector
        self.eval_source = QComboBox()
        self.eval_source.addItems(["NPY", "PNG", "JSON"])
        self.eval_source.currentIndexChanged.connect(self._recompute_scores_for_current)

        # hook model change AFTER creating the widget (only once)
        self._predictor = None
        self.model_selector.currentIndexChanged.connect(self._on_model_selected)

        # place both selectors side-by-side with equal stretch
        row   = QHBoxLayout()
        left  = QFormLayout();  left.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        right = QFormLayout(); right.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.model_selector_label = QLabel("Select Model Checkpoint")
        left.addRow(self.model_selector_label, self.model_selector)
        right.addRow("GT Source", self.eval_source)
        row.addLayout(left, 1)
        row.addLayout(right, 1)
        self.layout.addLayout(row)

        # --- End of controls row ---
        
        
        # Prediction button (create it first, keep disabled until model loads)
        self.predict_overlay_button = QPushButton("Predict and Overlay")
        self.predict_overlay_button.setEnabled(False)
        self.predict_overlay_button.clicked.connect(lambda: self.run_predict_and_overlay())
        self.layout.addWidget(self.predict_overlay_button)


        # Heatmap button 
        self.show_heatmap_button = QPushButton("Show Prediction Heatmap")
        self.show_heatmap_button.setEnabled(False)
        try:
            self.show_heatmap_button.clicked.disconnect()
        except Exception:
            pass
        self.show_heatmap_button.clicked.connect(self.show_heatmap_for_selected)
        self.layout.addWidget(self.show_heatmap_button)
        self._heatmap_predictor = None
        self._heatmap_ckpt_path = None
        self._heatmap_busy = False  # re-entrancy guard

        # Batch evaluation button
        self.eval_all_button = QPushButton("Run Batch Evaluation")
        self.eval_all_button.setEnabled(False)
        self.eval_all_button.clicked.connect(self.run_batch_evaluation)
        self.layout.addWidget(self.eval_all_button)
        
        

    def _has_valid_ckpt(self):
        try:
            p = self.model_selector.currentText().strip()
            return bool(p) and p != "(no checkpoints found)" and os.path.exists(p) and p.lower().endswith(".pth")
        except Exception:
            return False
    
                
    def select_source(self):
        choice = self.mode_selector.currentText()
        
        if choice == "Load from Folder":
            folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
            if folder_path:
                self.load_from_folder(folder_path)



    def load_from_folder(self, folder_path):
        # Accept only data/raw/<case>
        raw_folder = self._ensure_raw_folder(folder_path)
        if not raw_folder:
            return

        valid_exts = ('.jpg', '.jpeg', '.png')
        self.image_paths = []

        for root, _, files in os.walk(raw_folder):
            for f in files:
                fl = f.lower()
                if fl.endswith(valid_exts) and not fl.endswith("_preview.png"):
                    self.image_paths.append(os.path.join(root, f))

        if not self.image_paths:
            QMessageBox.warning(self, "No Images",
                                f"No raw images found in:\n{raw_folder}\n(accepted: .jpg/.jpeg/.png)")
            return

        self.image_paths.sort()
        self.populate_list()
        
        
    def _infer_label_case_from_path(self, image_path: str):
        """
        Returns (label, case) from a file path.
        case like 'Benign cases' ; label like 'benign'.
        Falls back gracefully if the structure is unusual.
        """
        p = os.path.normpath(image_path).replace("\\", "/").lower()
        # Try to grab the folder under data/raw/<case>/...
        case = None
        if "/data/raw/" in p:
            try:
                case = p.split("/data/raw/", 1)[1].split("/", 1)[0]  # e.g. 'benign cases'
            except Exception:
                case = None

        # If not found, search common case names anywhere in the path
        known_cases = ("benign cases", "malignant cases", "normal cases", "test cases")
        if case not in known_cases:
            for kc in known_cases:
                if f"/{kc}/" in p:
                    case = kc
                    break

        # Derive label
        label = case.replace(" cases", "") if case else "benign"
        return label, case


 
        
    def handle_mode_change(self, index):
        selected = self.mode_selector.currentText()
        print(f" Source mode changed to: {selected}")
        
    def change_draw_mode(self, mode):
      if mode in ["Polygon", "Freehand"]:
        self.canvas.set_mode(mode)

    def populate_list(self):
        self.image_list.clear()
        for path in self.image_paths:
            self.image_list.addItem(path)
            
            
    def is_model_ready(self):
        selected_model = self.model_selector.currentText()
        return bool(selected_model.strip())
    
    
    def _update_predict_controls(self):
        gate = self.prediction_toggle.isChecked()
        # Predict & batch (use temporal outputs): only need gate
        enable_pred_batch = bool(gate)  # Using temporal outputs
        if hasattr(self, "predict_overlay_button"):
            self.predict_overlay_button.setEnabled(enable_pred_batch)
        if hasattr(self, "eval_all_button"):
            self.eval_all_button.setEnabled(enable_pred_batch)

        # Heatmap needs gate + a valid checkpoint
        enable_heatmap = bool(gate and self._has_valid_ckpt())
        if hasattr(self, "show_heatmap_button"):
            self.show_heatmap_button.setEnabled(enable_heatmap)

            

    def _on_prediction_gate_toggled(self, checked: bool):
        """Master switch: when OFF, hide preview and disable all below-gate actions."""
        if not checked:
            # Hide preview only (keep cached pred if any)
            self.preview_canvas.clear()
            
        # ⬇️ auto-refresh the list when turns prediction ON
        self.refresh_model_selector()
        # Recompute enabled state for buttons
        self._update_predict_controls()

        # Optional: auto-predict when turning ON and an image is already selected
        if checked and (self._predictor is not None) and self.image_list.currentItem():
            self.run_predict_and_overlay(os.path.abspath(self.image_list.currentItem().text()))

    
                    
    def refresh_model_selector(self):
        """Populate the checkpoint combo from <project_root>/checkpoints (absolute, non‑recursive, .pth only)."""
        self.model_selector.clear()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_dir = os.path.join(base_dir, "checkpoints")

        paths = []
        if os.path.isdir(ckpt_dir):
            for f in os.listdir(ckpt_dir):  # non‑recursive on purpose
                fl = f.lower()
                if fl.endswith(".pth"):
                    paths.append(os.path.join(ckpt_dir, f))

        if paths:
            # sort newest first (optional; remove if you prefer alpha sort)
            paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for p in paths:
                self.model_selector.addItem(p)  # store full absolute path
        else:
            self.model_selector.addItem("(no checkpoints found)")




    def _on_model_selected(self):
        # switching ckpt invalidates cached predictor so we reload next time
        self._heatmap_predictor = None
        self._heatmap_ckpt_path = None
        if hasattr(self, "ckpt_status"):
            txt = self.model_selector.currentText().strip()
            self.ckpt_status.setText(txt if os.path.exists(txt) else "No valid checkpoint selected")
        self._update_predict_controls()
        
     
    
    def _maybe_auto_predict(self):
        if not self.prediction_toggle.isChecked():
            return
        item = self.image_list.currentItem()
        if not item:
            return
        self.run_predict_and_overlay(os.path.abspath(item.text()))

        
        
    def _recompute_scores_for_current(self):
        """Recompute Dice/IoU for the current image using the cached prediction, when GT source changes."""
        # Need a loaded model and a cached prediction
        if not hasattr(self, "_last_pred_mask"):
            self.eval_label.setText("Dice: N/A | IoU: N/A")
            return
        if self._last_pred_mask is None:
            self.eval_label.setText("Dice: N/A | IoU: N/A")
            return

        item = self.image_list.currentItem()
        if not item:
            self.eval_label.setText("Dice: N/A | IoU: N/A")
            return

        try:
            src = self.eval_source.currentText().upper()
            evaluator = Evaluator(src)
            dice, iou, prec, acc = evaluator.evaluate_one(self._last_pred_mask, item.text())
            if dice is None:
                self.eval_label.setText(f"{src}: GT not found")
            else:
                self.eval_label.setText(
                    f"{src} → Dice: {dice:.4f} | IoU: {iou:.4f} | Precision: {prec:.4f} | Accuracy: {acc:.4f}"
                )

        except Exception as e:
            print(f"[Eval] Error: {e}")
            self.eval_label.setText("Dice: N/A | IoU: N/A")


    
    def display_image(self, item):
        path = item.text()
        pixmap = QPixmap(path)
        print(" Loading image:", path)

        if pixmap.isNull():
            #QMessageBox.warning(self, "Warning", f"Could not load image: {path}")
            print(f" Failed to load image from: {path}")
            return

        # Show the actual image first
        self.canvas.setPixmap(pixmap)  # canvas computes scale/offset itself
        
        # Auto-run only if model is already loaded
        self._maybe_auto_predict()

        # Clear existing annotations FIRST
        self.canvas.points.clear()
        self.canvas.freehand_path.clear()

        # Auto-load annotation JSON if it exists
        img_path = item.text()
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label, case = self._infer_label_case_from_path(img_path)

        cands = []
        if label:
            cands.append(os.path.join("data", "annotations", label, f"{stem}.json"))
        if case:
            cands.append(os.path.join("data", "annotations", case, f"{stem}.json"))

        json_path = next((p for p in cands if os.path.exists(p)), None)
        # Clear drawings first
        self.canvas.points.clear()
        self.canvas.freehand_path.clear()

        if json_path:
            try:
                import json
                with open(json_path, "r") as f:
                    data = json.load(f)
                self.canvas.set_mode(data.get("mode", "Polygon"))
                pts256 = data.get("points", [])
                img_w, img_h = self.canvas.image_width, self.canvas.image_height
                qpoints = self._from_256_points(pts256, img_w, img_h)
                if self.canvas.mode == "Polygon":
                    self.canvas.points = qpoints
                else:
                    self.canvas.freehand_path = qpoints
                self.canvas.update()
            except Exception as e:
                print(f"⚠️ Failed to load annotation JSON: {e}")

                
    
    def _set_preview_rgba(self, pil_rgba):
        qimg = QImage(pil_rgba.tobytes("raw", "RGBA"),
                    pil_rgba.width, pil_rgba.height,
                    QImage.Format_RGBA8888)
        self.preview_canvas.setPixmap(QPixmap.fromImage(qimg))

       
    
    def _to_256_points(self, qpoints, img_w, img_h):
        out = []
        for pt in qpoints:
            x256 = int(round(pt.x() * 256.0 / max(1, img_w)))
            y256 = int(round(pt.y() * 256.0 / max(1, img_h)))
            # clamp defensive
            x256 = max(0, min(255, x256))
            y256 = max(0, min(255, y256))
            out.append([x256, y256])
        return out
    
    
    def _from_256_points(self, pts256, img_w, img_h):
        qpts = []
        for x, y in pts256:
            xi = int(round(float(x) * img_w / 256.0))
            yi = int(round(float(y) * img_h / 256.0))
            xi = max(0, min(img_w - 1, xi))
            yi = max(0, min(img_h - 1, yi))
            qpts.append(QPoint(xi, yi))
        return qpts


    
    def save_annotation(self):
        item = self.image_list.currentItem()
        if not item:
            QMessageBox.warning(self, "Warning", "No image selected.")
            return
        image_path = item.text()

        label, case = self._infer_label_case_from_path(image_path)

        # where you create the folder:
        save_path = os.path.join("data", "annotations", label)
        os.makedirs(save_path, exist_ok=True)
        
        # ---- name like <raw_stem>.json
        filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"

        # --- convert from image-space → 256×256 ---
        img_w = self.canvas.image_width
        img_h = self.canvas.image_height
        if self.canvas.mode == "Polygon":
            pts256 = self._to_256_points(self.canvas.points, img_w, img_h)
        else:
            pts256 = self._to_256_points(self.canvas.freehand_path, img_w, img_h)

        annotation = {
            "mode": self.canvas.mode,
            "points": pts256,          # <<< normalized to 256×256
            "label": label,
            "coord_space": 256,        # <<< tells downstream code how to interpret points
            "imageWidth": 256,         # (optional metadata)
            "imageHeight": 256
        }

        try:
            import json
            with open(os.path.join(save_path, filename), 'w') as f:
                json.dump(annotation, f, indent=2)
            QMessageBox.information(self, "SAVED", f"Annotation saved to {save_path}/{filename}")
        except Exception as e:
            QMessageBox.critical(self, "ERROR", f"Failed to save annotation: {e}")

      
    
    def convert_annotations(self):
        try:
            from convert_json_to_mask import convert_all
            folder = QFileDialog.getExistingDirectory(self, "Select Annotaion Folder for Conversion", "data")
            if not folder:
                return  

            item = self.image_list.currentItem()
            if not item:
                QMessageBox.warning(self, "No image", "Select an image first.")
                return
            label, _ = self._infer_label_case_from_path(item.text())
            convert_all(json_dir=os.path.join("data", "annotations", label),
                    png_output_dir="data/masks_png",
                    npy_output_dir="data/masks_npy",
                    raw_image_folder="data/raw"
            )

            QMessageBox.information(self, "Success", f"✅ Masks created for '{label}' class.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ Failed to convert annotations: {e}")
            
            
            
    

    def generate_green_box_masks(self):

        script_path = os.path.join(os.path.dirname(__file__), "generate_masks_from_embedded_boxes.py")
        if not os.path.exists(script_path):
            QMessageBox.critical(self, "Script Error", f"❌ File not found:\n{script_path}")
            return

        self._proc = QProcess(self)
        self._proc.setProgram(sys.executable)                # same interpreter as GUI
        self._proc.setWorkingDirectory(os.path.dirname(__file__))
        self._proc.setProcessEnvironment(self._proc.processEnvironment())
        self._proc.setArguments([script_path])

        # Optional: capture live output (to print or log)
        self._proc.readyReadStandardOutput.connect(
            lambda: print(str(self._proc.readAllStandardOutput(), "utf-8", errors="ignore"), end="")
        )
        self._proc.readyReadStandardError.connect(
            lambda: print(str(self._proc.readAllStandardError(), "utf-8", errors="ignore"), end="")
        )

        def on_finished(code, _status):
            if code == 0:
                QMessageBox.information(self, "Success", "✅ Masks generated successfully from green boxes!")
            else:
                QMessageBox.critical(self, "Error", f"❌ Script failed with exit code {code}.")

        self._proc.finished.connect(on_finished)
        self._proc.start()

            
            
    
    def run_temporal_inference_from_annotations(self):
        try:
            from PyQt5.QtWidgets import QFileDialog
            import subprocess, sys, os

            folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", "data/raw/Benign cases")
            if not folder:
                return

            # ensure UTF-8 so checkmarks etc. don't explode on Windows
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            # Stream output line-by-line to the same console you launched the GUI from
            proc = subprocess.Popen(
                [sys.executable, "temporal_inference.py", "--image_dir", folder],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                env=env,
                bufsize=1  # line-buffered
            )

            # Optional: also tee to a logfile
            os.makedirs("logs", exist_ok=True)
            log_path = os.path.join("logs", "temporal_inference.log")
            with open(log_path, "w", encoding="utf-8") as logf:
                for line in proc.stdout:
                    print(line, end="")        # forward to terminal live
                    logf.write(line)

            proc.wait()

            if proc.returncode == 0:
                print(f"[OK] Temporal inference finished. See logs/temporal_inference.log for details.")
            else:
                print(f"[WARN] Temporal inference exited with code {proc.returncode}. See logs/temporal_inference.log for details.")

        except Exception as e:
            print(f"[GUI ERROR] {e}")


           
    
    def run_predict_and_overlay(self, image_path=None):
        # master gate guard
        if not self.prediction_toggle.isChecked():
            QMessageBox.information(self, "Prediction Disabled",
                                    "Turn on 'Show Model Prediction' to display temporal results.")
            return

        # 1) Resolve current raw image
        if image_path is None:
            item = self.image_list.currentItem()
            if not item:
                QMessageBox.warning(self, "No image", "Select a raw image first.")
                return
            image_path = item.text()

        # RAW-only guard
        p = image_path.replace("\\", "/").lower()
        if any(seg in p for seg in ("/masks_png/", "/masks_npy/", "/annotations/")) or p.endswith("_preview.png"):
            QMessageBox.warning(self, "Raw Images Only", "Pick from data/raw/<Case>/")
            return

        # 2) Load the precomputed prediction mask from temporal_inference outputs
        try:
            from inference_eval import load_pred_mask_from_outputs
            pred_bin = load_pred_mask_from_outputs(image_path)
            if pred_bin is None or pred_bin.sum() == 0:
                QMessageBox.warning(self, "No Temporal Result",
                                    "No prediction found for this image. "
                                    "Run Temporal Inference first.")
                self.preview_canvas.clear()
                self.eval_label.setText("Dice: N/A | IoU: N/A")
                return
            self._last_pred_mask = pred_bin
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load temporal prediction:\n{e}")
            return

        # 3) Update the right preview canvas with overlay (512 for display)
        try:
            from inference_eval import make_overlay
            overlay = make_overlay(image_path, self._last_pred_mask, out_size=512)
            self._set_preview_rgba(overlay)
        except Exception as e:
            QMessageBox.critical(self, "Overlay Error", f"{e}")
            return

        # 4) Evaluate vs selected GT source (NPY / PNG / JSON)
        try:
            src = self.eval_source.currentText().upper() if hasattr(self, "eval_source") else "NPY"
            from inference_eval import Evaluator
            evaluator = Evaluator(src)
            dice, iou, prec, acc = evaluator.evaluate_one(self._last_pred_mask, image_path)
            if dice is None:
                self.eval_label.setText(f"{src}: GT not found")
            else:
                self.eval_label.setText(
                    f"{src} → Dice: {dice:.4f} | IoU: {iou:.4f} | Precision: {prec:.4f} | Accuracy: {acc:.4f}"
                )

        except Exception as e:
            self.eval_label.setText("Dice: N/A | IoU: N/A")
            print(f"[Eval] Error: {e}")



        
    def evaluate_prediction(self, image_path):
        # RAW-only guard
        p = image_path.replace("\\", "/")
        if "/masks_png/" in p or "/masks_npy/" in p or "/annotations/" in p or p.endswith("_preview.png"):
            QMessageBox.warning(self, "Raw Images Only",
                                "Please select an image from data/raw/<Case>/ and try again.")
            return

        # Need a prediction first
        if not hasattr(self, "_last_pred_mask"):
            self.eval_label.setText("Dice: N/A | IoU: N/A")
            return

        src = self.eval_source.currentText().upper()
        try:
            evaluator = Evaluator(src)
        except Exception as e:
            QMessageBox.critical(self, "Eval Source Error", str(e))
            return

        dice, iou, prec, acc = evaluator.evaluate_one(self._last_pred_mask, image_path)
        if dice is None:
            self.eval_label.setText(f"{src}: GT not found")
        else:
            self.eval_label.setText(
                f"{src} → Dice: {dice:.4f} | IoU: {iou:.4f} | Precision: {prec:.4f} | Accuracy: {acc:.4f}"
            )

        
    
    
    
    # --- Enforce RAW-only selection ---------------------------------------------
    def _ensure_raw_folder(self, folder_path: str):
        """
        Accept only folders under data/raw/<case>. If user picks masks_png / masks_npy /
        annotations, auto-map to the matching data/raw/<case> when possible.
        """
        norm = folder_path.replace("\\", "/")
        raw_root = os.path.join("data", "raw")

        # Already a raw folder? ok.
        if "/data/raw/" in norm or norm.endswith("/data/raw") or norm.endswith("\\data\\raw"):
            return folder_path

        # If they picked masks/annotations, try to map to raw/<case>
        for key in ("/data/masks_png/", "/data/masks_npy/", "/data/annotations/"):
            if key in norm:
                try:
                    case = norm.split(key, 1)[1].split("/", 1)[0]
                except Exception:
                    case = None
                if case:
                    candidate = os.path.join(raw_root, case)
                    if os.path.isdir(candidate):
                        QMessageBox.information(
                            self, "Switching to Raw Images",
                            f"Detected non-raw folder.\nUsing matching raw folder:\n{candidate}"
                        )
                        return candidate

        # Not a raw folder and no mapping → block
        QMessageBox.warning(
            self, "Raw Images Only",
            "Please select a folder under data/raw/<Case>/ containing the original CT images."
        )
        return None
    # ---------------------------------------------------------------------------

        
    
    
    def show_heatmap_for_selected(self):
        # master gate guard
        if not self.prediction_toggle.isChecked():
            QMessageBox.information(self, "Model Prediction Disabled",
                                    "Turn on 'Show Model Prediction' to view heatmaps.")
            return

        # avoid double-trigger (some platforms fire twice)
        if self._heatmap_busy:
            return
        self._heatmap_busy = True
        try:
            item = self.image_list.currentItem()
            if not item:
                QMessageBox.warning(self, "No image", "Select an image first.")
                return
            image_path = item.text()

            # require a valid checkpoint path
            if not self._has_valid_ckpt():
                QMessageBox.warning(self, "No checkpoint",
                                    "Select a valid checkpoint file to show heatmaps.")
                return
            ckpt_path = self.model_selector.currentText().strip()

            # cache predictor per-ckpt
            if (self._heatmap_predictor is None) or (self._heatmap_ckpt_path != ckpt_path):
                self._heatmap_predictor = Predictor(ckpt_path, device=self.device)
                self._heatmap_ckpt_path = ckpt_path

            # build overlay using cached predictor (won’t reload ckpt each time)
            overlay = build_heatmap_overlay(
                image_path,
                predictor=self._heatmap_predictor,
                colormap="jet",
                alpha=0.40,
                out_size=512
            )
            qimg = QImage(overlay.tobytes("raw", "RGBA"), 512, 512, QImage.Format_RGBA8888)
            self.preview_canvas.setPixmap(QPixmap.fromImage(qimg))
        except Exception as e:
            QMessageBox.critical(self, "Heatmap Error", str(e))
        finally:
            self._heatmap_busy = False

        
    
    
    def run_batch_evaluation(self):
        # master gate guard
        if not self.prediction_toggle.isChecked():
            QMessageBox.information(self, "Prediction Disabled",
                                    "Turn on 'Show Model Prediction' to run batch evaluation.")
            return

        src = self.eval_source.currentText().upper()
        evaluator = Evaluator(src)

        # Choose image folder
        Tk().withdraw()
        image_dir = filedialog.askdirectory(title="Select RAW folder to evaluate")
        if not image_dir:
            return

        valid = [f for f in os.listdir(image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.endswith("_preview.png")]
        if not valid:
            QMessageBox.warning(self, "Batch Evaluation", "No raw images in that folder.")
            return

        dice_scores, iou_scores, prec_scores, acc_scores, cnt = [], [], [], [], 0
        for fname in valid:
            img_path = os.path.join(image_dir, fname)
            pred = load_pred_mask_from_outputs(img_path)
            if pred is None:
                print(f"[Eval] Skip (no temporal prediction) → {fname}")
                continue
            d, j, p, a = evaluator.evaluate_one(pred, img_path)
            if d is None:
                print(f"[Eval] Skip (GT not found) → {fname}")
                continue
            dice_scores.append(d); iou_scores.append(j); prec_scores.append(p); acc_scores.append(a); cnt += 1

        if cnt > 0:
            avg_d, avg_j = float(np.mean(dice_scores)), float(np.mean(iou_scores))
            avg_p, avg_a = float(np.mean(prec_scores)), float(np.mean(acc_scores))
            
            QMessageBox.information(
                self, "Batch Evaluation",
                f"✅ Evaluated {cnt} images\n\nSource: {src}"
                f"\nAvg Dice: {avg_d:.4f}\nAvg IoU: {avg_j:.4f}\nAvg Precision: {avg_p:.4f}\nAvg Accuracy: {avg_a:.4f}"
            )
        else:
            QMessageBox.warning(self, "Batch Evaluation",
                f"⚠️ No pairs evaluated (source={src}); check that temporal predictions and GT exist.")

            
            
       

class AnnotationCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.mode = "Polygon"  # or "Freehand"
        self.points = []
        self.freehand_path = []
        self.setMouseTracking(True)
        self.image_offset = (0, 0)  # x, y offset to center image manually
        self.display_size = 512
        self.image_display_scale = 1.0
        # --- keep-aspect scaling state ---
        self.image_width = 256
        self.image_height = 256
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.image_offset = QPoint(0, 0)  # top-left corner of the scaled pixmap inside the label


    def set_mode(self, mode):
        self.mode = mode
        self.points.clear()
        self.freehand_path.clear()
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            return
        # position in the scaled image area
        raw = event.pos() - self.image_offset
        ix = int(raw.x() / self.scale_x)
        iy = int(raw.y() / self.scale_y)
        # clamp to image bounds
        ix = max(0, min(self.image_width - 1, ix))
        iy = max(0, min(self.image_height - 1, iy))
        p = QPoint(ix, iy)

        if self.mode == "Polygon":
            self.points.append(p)      # store in IMAGE coordinates
        elif self.mode == "Freehand":
            self.freehand_path = [p]
            self.drawing = True
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.mode != "Freehand" or not self.drawing:
            return
        raw = event.pos() - self.image_offset
        ix = int(raw.x() / self.scale_x)
        iy = int(raw.y() / self.scale_y)
        ix = max(0, min(self.image_width - 1, ix))
        iy = max(0, min(self.image_height - 1, iy))
        self.freehand_path.append(QPoint(ix, iy))
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.mode == "Freehand" and self.drawing:
            self.drawing = False

            
            
    def setPixmap(self, pixmap: QPixmap):
        """Display the image with aspect-ratio preserved and track scale/offset."""
        if pixmap is None:
            super().setPixmap(QPixmap())
            return

        # Original image size (used for storing points in image-space)
        self.image_width = pixmap.width()
        self.image_height = pixmap.height()

        # Fit inside the label size, keep aspect
        target_w = max(1, self.width())
        target_h = max(1, self.height())
        scaled = pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Compute scale and offset so we can map mouse↔image coordinates
        self.scale_x = scaled.width() / float(self.image_width)
        self.scale_y = scaled.height() / float(self.image_height)
        off_x = (target_w - scaled.width()) // 2
        off_y = (target_h - scaled.height()) // 2
        self.image_offset = QPoint(off_x, off_y)

        super().setPixmap(scaled)
        self.update()



    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # helper to map image → screen coords for drawing
        def to_screen(pt: QPoint) -> QPoint:
            return QPoint(int(pt.x() * self.scale_x) + self.image_offset.x(),
                        int(pt.y() * self.scale_y) + self.image_offset.y())

        pen = QPen(Qt.green, 2)
        painter.setPen(pen)

        if self.mode == "Polygon" and self.points:
            scr_pts = [to_screen(p) for p in self.points]
            for i in range(1, len(scr_pts)):
                painter.drawLine(scr_pts[i-1], scr_pts[i])
            # optionally close polygon visually
            if len(scr_pts) > 2:
                painter.drawLine(scr_pts[-1], scr_pts[0])

        elif self.mode == "Freehand" and self.freehand_path:
            scr_pts = [to_screen(p) for p in self.freehand_path]
            for i in range(1, len(scr_pts)):
                painter.drawLine(scr_pts[i-1], scr_pts[i])



    def undo(self):
        if self.mode == "Polygon" and self.points:
            self.points.pop()
        elif self.mode == "Freehand" and self.freehand_path:
            self.freehand_path.pop()
        self.update()

    def reset(self):
        self.points.clear()
        self.freehand_path.clear()
        self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageLoaderApp()
    window.show()
    sys.exit(app.exec_())

