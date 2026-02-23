import sys
import os
import cv2
import numpy as np
import zarr
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QSlider, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QScrollArea, QSplitter,
    QMessageBox, QProgressDialog
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
import pyqtgraph as pg

class VideoReaderThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    playback_finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.cap = None
        self.running = False
        self.playing = False
        self.fps = 30
        self.speed_multiplier = 1.0
        
    def load_video(self, path):
        if self.cap is not None:
            self.cap.release()
        self.video_path = path
        self.cap = cv2.VideoCapture(self.video_path)
        if self.cap.isOpened():
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            return True
        return False
        
    def run(self):
        self.running = True
        while self.running:
            if self.playing and self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.frame_ready.emit(frame)
                    # Simple delay for playback speed
                    delay = int(1000 / (self.fps * self.speed_multiplier))
                    self.msleep(delay)
                else:
                    self.playing = False
                    self.playback_finished.emit()
            else:
                self.msleep(50)
                
    def seek(self, frame_idx):
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
                
    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()

from engine import process_video

class ComputeThread(QThread):
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, video_path, patch_size, freq_min, freq_max, sal_thresh):
        super().__init__()
        self.video_path = video_path
        self.patch_size = patch_size
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.sal_thresh = sal_thresh
        
    def run(self):
        try:
            output_dir = "data/results"
            os.makedirs(output_dir, exist_ok=True)
            process_video(
                input_path=self.video_path,
                output_dir=output_dir,
                patch_size=self.patch_size,
                freq_min=self.freq_min,
                freq_max=self.freq_max,
                saliency_threshold=self.sal_thresh
            )
            self.finished_signal.emit(True, "Computation completed successfully.")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class VideoLabel(QLabel):
    roiAdded = pyqtSignal(tuple, tuple) # (template_rect, search_rect)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing_state = "IDLE" # IDLE, DRAW_TEMPLATE, DRAW_SEARCH
        self.start_point = None
        self.current_point = None
        self.template_rect = None
        self.search_rect = None
        self.setMouseTracking(True)
        
    def set_state(self, state):
        self.drawing_state = state
        self.start_point = None
        self.current_point = None
        if state == "IDLE":
            self.template_rect = None
            self.search_rect = None

    def mousePressEvent(self, event):
        if self.drawing_state in ["DRAW_TEMPLATE", "DRAW_SEARCH"] and event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.pos()
            self.current_point = event.pos()
            
    def mouseMoveEvent(self, event):
        if self.start_point is not None:
            self.current_point = event.pos()
            # We don't paint here directly, we rely on the main window to update frame or we can just update the pixmap.
            # But the video is playing/paused. We should trigger a repaint.
            self.update() # triggers paintEvent
            
    def mouseReleaseEvent(self, event):
        if self.start_point is not None and event.button() == Qt.MouseButton.LeftButton:
            self.current_point = event.pos()
            rect = (
                min(self.start_point.x(), self.current_point.x()),
                min(self.start_point.y(), self.current_point.y()),
                abs(self.start_point.x() - self.current_point.x()),
                abs(self.start_point.y() - self.current_point.y())
            )
            self.start_point = None
            self.current_point = None
            
            # Map widget coordinates to image coordinates
            # This requires knowing the pixmap scale. 
            # For simplicity, we'll map them later in the main window when passing rects.
            
            if self.drawing_state == "DRAW_TEMPLATE":
                self.template_rect = rect
                self.drawing_state = "DRAW_SEARCH"
                QMessageBox.information(self, "ROI", "Template selected. Now draw the search region.")
            elif self.drawing_state == "DRAW_SEARCH":
                self.search_rect = rect
                self.drawing_state = "IDLE"
                self.roiAdded.emit(self.template_rect, self.search_rect)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.drawing_state != "IDLE":
            from PyQt6.QtGui import QPainter, QPen, QColor
            painter = QPainter(self)
            
            if self.template_rect:
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.drawRect(*self.template_rect)
                
            if self.start_point and self.current_point:
                rect = (
                    min(self.start_point.x(), self.current_point.x()),
                    min(self.start_point.y(), self.current_point.y()),
                    abs(self.start_point.x() - self.current_point.x()),
                    abs(self.start_point.y() - self.current_point.y())
                )
                if self.drawing_state == "DRAW_TEMPLATE":
                    painter.setPen(QPen(QColor(255, 0, 0), 2))
                else:
                    painter.setPen(QPen(QColor(0, 255, 0), 2))
                painter.drawRect(*rect)


class RoiTrackingThread(QThread):
    finished_signal = pyqtSignal(dict) # Returns dict with tracking results
    
    def __init__(self, video_path, template_rect, search_rect):
        super().__init__()
        self.video_path = video_path
        self.template_rect = template_rect # (x, y, w, h)
        self.search_rect = search_rect     # (sx, sy, sw, sh)
        
    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("Cannot open video for ROI tracking")
                
            ret, first_frame = cap.read()
            if not ret:
                raise Exception("Cannot read first frame")
                
            gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            
            x, y, w, h = self.template_rect
            sx, sy, sw, sh = self.search_rect
            
            # Ensure boundaries
            H, W = gray.shape
            x = max(0, min(x, W-w))
            y = max(0, min(y, H-h))
            sx = max(0, min(sx, W-sw))
            sy = max(0, min(sy, H-sh))
            
            template = gray[y:y+h, x:x+w]
            
            displacements_x = []
            displacements_y = []
            
            # First frame displacement is 0 
            # We track the center of the template relative to its initial position
            init_center_x = x + w/2.0
            init_center_y = y + h/2.0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                search_region = gray_frame[sy:sy+sh, sx:sx+sw]
                
                res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)
                
                # max_loc is (x, y) in search_region
                matched_center_x = sx + max_loc[0] + w/2.0
                matched_center_y = sy + max_loc[1] + h/2.0
                
                dx = matched_center_x - init_center_x
                dy = matched_center_y - init_center_y
                
                displacements_x.append(dx)
                displacements_y.append(dy)
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0:
                fps = 30.0
            
            cap.release()
            
            dt = 1.0 / fps
            time_axis = np.arange(len(displacements_x)) * dt
            
            result = {
                'success': True,
                'time': time_axis,
                'x': np.array(displacements_x),
                'y': np.array(displacements_y),
                'template_rect': self.template_rect,
                'search_rect': self.search_rect
            }
            self.finished_signal.emit(result)
            
        except Exception as e:
            self.finished_signal.emit({'success': False, 'error': str(e)})


class RoiItemWidget(QWidget):
    def __init__(self, roi_id, data, parent=None):
        super().__init__(parent)
        self.roi_id = roi_id
        self.data = data
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_name = QLabel(f"ROI {roi_id}")
        layout.addWidget(self.lbl_name)
        
        self.btn_xy = QPushButton("XY Wave")
        self.btn_xy.setCheckable(True)
        self.btn_xy.toggled.connect(self.toggle_xy)
        layout.addWidget(self.btn_xy)
        
        self.btn_orbit = QPushButton("Orbit")
        self.btn_orbit.setCheckable(True)
        self.btn_orbit.toggled.connect(self.toggle_orbit)
        layout.addWidget(self.btn_orbit)
        
        self.btn_vector = QPushButton("Vector Layer")
        self.btn_vector.setCheckable(True)
        layout.addWidget(self.btn_vector)
        
        # Delete button
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setStyleSheet("background-color: #8b0000; color: white;")
        layout.addWidget(self.btn_delete)
        
        # Pyqtgraph windows
        self.xy_window = None
        self.orbit_window = None
        
    def toggle_xy(self, checked):
        if checked:
            if self.xy_window is None:
                self.xy_window = pg.GraphicsLayoutWidget(title=f"ROI {self.roi_id} Waveforms")
                self.xy_window.resize(600, 400)
                
                p1 = self.xy_window.addPlot(title="X Displacement")
                p1.plot(self.data['time'], self.data['x'], pen='r')
                p1.setLabel('left', 'Displacement (px)')
                p1.setLabel('bottom', 'Time (s)')
                
                self.xy_window.nextRow()
                
                p2 = self.xy_window.addPlot(title="Y Displacement")
                p2.plot(self.data['time'], self.data['y'], pen='g')
                p2.setLabel('left', 'Displacement (px)')
                p2.setLabel('bottom', 'Time (s)')
                
            self.xy_window.show()
        else:
            if self.xy_window:
                self.xy_window.hide()
                
    def toggle_orbit(self, checked):
        if checked:
            if self.orbit_window is None:
                self.orbit_window = pg.GraphicsLayoutWidget(title=f"ROI {self.roi_id} Orbit")
                self.orbit_window.resize(400, 400)
                p = self.orbit_window.addPlot(title="X vs Y Orbit")
                p.plot(self.data['x'], self.data['y'], pen='b')
                p.setLabel('left', 'Y - Displacement (px)')
                p.setLabel('bottom', 'X - Displacement (px)')
                # Aspect ratio 1:1
                p.setAspectLocked(True)
            self.orbit_window.show()
        else:
            if self.orbit_window:
                self.orbit_window.hide()

class AnalysisMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VibeField-Engine Analysis GUI")
        self.resize(1200, 800)
        
        self.video_thread = VideoReaderThread()
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.playback_finished.connect(self.on_playback_finished)
        self.video_thread.start()
        
        self.init_ui()
        self.populate_video_list()
        
    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Splitter to separate left/right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left Panel (Video and Controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Video List
        self.video_list = QListWidget()
        self.video_list.setMaximumHeight(100)
        self.video_list.itemClicked.connect(self.on_video_selected)
        left_layout.addWidget(QLabel("Videos in data/resources/:"))
        left_layout.addWidget(self.video_list)
        
        # Video Display
        self.video_label = VideoLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.roiAdded.connect(self.on_roi_drawn)
        left_layout.addWidget(self.video_label, 1) # Give it stretching weight
        
        # Playback Controls
        ctrl_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_playback)
        self.slider_progress = QSlider(Qt.Orientation.Horizontal)
        self.slider_progress.sliderMoved.connect(self.on_progress_moved)
        self.combo_speed = QComboBox()
        self.combo_speed.setEditable(True)
        from PyQt6.QtGui import QDoubleValidator
        validator = QDoubleValidator(0.1, 10.0, 1)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.combo_speed.setValidator(validator)
        self.combo_speed.addItems(["0.5", "1.0", "1.5", "2.0", "5.0"])
        self.combo_speed.setCurrentText("1.0")
        self.combo_speed.currentTextChanged.connect(self.on_speed_changed)
        
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.slider_progress)
        ctrl_layout.addWidget(QLabel("Speed:"))
        ctrl_layout.addWidget(self.combo_speed)
        left_layout.addLayout(ctrl_layout)
        
        splitter.addWidget(left_panel)
        
        # Right Panel (Tools)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(350)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # --- Area 1: Global Computing ---
        group_compute = QGroupBox("Global Computing")
        layout_compute = QVBoxLayout()
        
        # Patch Size
        h_patch = QHBoxLayout()
        h_patch.addWidget(QLabel("Patch Size:"))
        self.spin_patch = QSpinBox()
        self.spin_patch.setRange(1, 128)
        self.spin_patch.setValue(1)
        h_patch.addWidget(self.spin_patch)
        layout_compute.addLayout(h_patch)
        
        # Freq Range
        h_freq = QHBoxLayout()
        h_freq.addWidget(QLabel("Freq (Hz):"))
        self.spin_freq_min = QDoubleSpinBox()
        self.spin_freq_min.setRange(0.1, 1000.0)
        self.spin_freq_min.setValue(1.0)
        self.spin_freq_min.setDecimals(1)
        h_freq.addWidget(self.spin_freq_min)
        h_freq.addWidget(QLabel("-"))
        self.spin_freq_max = QDoubleSpinBox()
        self.spin_freq_max.setRange(0.1, 1000.0)
        self.spin_freq_max.setValue(50.0)
        self.spin_freq_max.setDecimals(1)
        h_freq.addWidget(self.spin_freq_max)
        layout_compute.addLayout(h_freq)
        
        # Saliency Threshold
        h_sal = QHBoxLayout()
        h_sal.addWidget(QLabel("Saliency Threshold:"))
        self.spin_sal = QDoubleSpinBox()
        self.spin_sal.setRange(0.0, 1.0)
        self.spin_sal.setSingleStep(1e-5)
        self.spin_sal.setDecimals(6)
        self.spin_sal.setValue(1e-5)
        h_sal.addWidget(self.spin_sal)
        layout_compute.addLayout(h_sal)
        
        # Compute Button
        self.btn_compute = QPushButton("Compute Global Vector Field")
        self.btn_compute.clicked.connect(self.on_compute_global)
        layout_compute.addWidget(self.btn_compute)
        
        group_compute.setLayout(layout_compute)
        right_layout.addWidget(group_compute)
        
        # --- Area 2: Overlay Rendering ---
        group_overlay = QGroupBox("Overlay Rendering")
        layout_overlay = QVBoxLayout()
        
        # Enable Overlay
        self.chk_overlay = QCheckBox("Enable Overlay Display")
        self.chk_overlay.toggled.connect(self.on_overlay_toggled)
        layout_overlay.addWidget(self.chk_overlay)
        
        # Overlay Params
        h_scale = QHBoxLayout()
        h_scale.addWidget(QLabel("Vector Scale (%):"))
        self.spin_scale = QSpinBox()
        self.spin_scale.setRange(1, 500)
        self.spin_scale.setValue(100)
        h_scale.addWidget(self.spin_scale)
        layout_overlay.addLayout(h_scale)
        
        h_minth = QHBoxLayout()
        h_minth.addWidget(QLabel("Min Thresh (%):"))
        self.spin_minth = QSpinBox()
        self.spin_minth.setRange(0, 100)
        self.spin_minth.setValue(2)
        h_minth.addWidget(self.spin_minth)
        layout_overlay.addLayout(h_minth)
        
        h_maxth = QHBoxLayout()
        h_maxth.addWidget(QLabel("Max Thresh (%):"))
        self.spin_maxth = QSpinBox()
        self.spin_maxth.setRange(1, 200)
        self.spin_maxth.setValue(100)
        h_maxth.addWidget(self.spin_maxth)
        layout_overlay.addLayout(h_maxth)
        
        group_overlay.setLayout(layout_overlay)
        right_layout.addWidget(group_overlay)
        
        # --- Area 3: ROI Tracking & Analysis ---
        group_roi = QGroupBox("ROI Tracking & Analysis")
        layout_roi = QVBoxLayout()
        
        self.btn_add_roi = QPushButton("Add ROI (+)")
        self.btn_add_roi.clicked.connect(self.on_add_roi)
        layout_roi.addWidget(self.btn_add_roi)
        
        self.roi_list = QListWidget()
        layout_roi.addWidget(self.roi_list)
        
        group_roi.setLayout(layout_roi)
        right_layout.addWidget(group_roi)
        
        right_scroll.setWidget(right_panel)
        splitter.addWidget(right_scroll)
        
        # Adjust splitter sizing
        splitter.setSizes([800, 400])

    def populate_video_list(self):
        self.video_list.clear()
        res_dir = "data/resources"
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
            return
            
        for f in os.listdir(res_dir):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.video_list.addItem(f)

    def on_video_selected(self, item):
        video_name = item.text()
        video_path = os.path.join("data/resources", video_name)
        if self.video_thread.load_video(video_path):
            self.btn_play.setText("Play")
            self.video_thread.playing = False
            total_frames = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider_progress.setRange(0, total_frames - 1)
            self.slider_progress.setValue(0)
            self.video_thread.seek(0)
            
    def toggle_playback(self):
        if self.video_thread.cap is None or not self.video_thread.cap.isOpened():
            return
            
        if self.video_thread.playing:
            self.video_thread.playing = False
            self.btn_play.setText("Play")
        else:
            self.video_thread.playing = True
            self.btn_play.setText("Pause")

    def on_progress_moved(self, value):
        self.video_thread.playing = False
        self.btn_play.setText("Play")
        self.video_thread.seek(value)
        
    def on_speed_changed(self, text):
        try:
            # Handle possible trailing 'x' if user types it, or just float
            clean_text = text.replace("x", "").replace("X", "").strip()
            if not clean_text: return
            speed = float(clean_text)
            if speed < 0.1: speed = 0.1
            if speed > 10.0: speed = 10.0
            self.video_thread.speed_multiplier = speed
        except ValueError:
            pass

    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Overlay rendering
        if hasattr(self, 'chk_overlay') and self.chk_overlay.isChecked() and getattr(self, 'overlay_data', None):
            od = self.overlay_data
            
            scale_perc = self.spin_scale.value() / 100.0
            min_th_perc = self.spin_minth.value() / 100.0
            max_th_perc = self.spin_maxth.value() / 100.0
            
            if scale_perc == 0: scale_perc = 0.01
            if max_th_perc < min_th_perc: max_th_perc = min_th_perc + 0.01
            
            current_arrow_scale = od['init_arrow_scale'] * scale_perc
            
            # Reconstruct time t based on frame index
            current_frame = int(self.video_thread.cap.get(cv2.CAP_PROP_POS_FRAMES))
            t = current_frame / self.video_thread.fps
            
            omega = 2 * np.pi * od['freqs'] * t
            inst_u = od['amp_u'] * np.cos(omega + od['phase_u'])
            inst_v = od['amp_v'] * np.cos(omega + od['phase_v'])
            
            draw_u = inst_u * current_arrow_scale
            draw_v = inst_v * current_arrow_scale
            inst_mags = np.sqrt(inst_u**2 + inst_v**2)
            
            draw_mask = (inst_mags >= (od['p98_amp'] * min_th_perc)) & (inst_mags <= (od['p98_amp'] * max_th_perc))
            
            d_cx = od['centers_x'][draw_mask]
            d_cy = od['centers_y'][draw_mask]
            d_du = draw_u[draw_mask]
            d_dv = draw_v[draw_mask]
            d_mags = inst_mags[draw_mask]
            
            # Extract unscaled actual displacements for anchoring
            d_inst_u = inst_u[draw_mask]
            d_inst_v = inst_v[draw_mask]
            
            intensities = np.clip(d_mags / od['p98_amp'], 0.0, 1.0)
            color_idxs = (intensities * 255).astype(int)
            
            # Since frame_rgb is RGB, color_lut should be RGB (which we made sure it is)
            for i in range(len(d_cx)):
                cx = d_cx[i]
                cy = d_cy[i]
                
                # We use the unscaled physiological sub-pixel displacement for anchor
                actual_u = d_inst_u[i]
                actual_v = d_inst_v[i]
                
                # Using round to be slightly more accurate than int
                cx_dyn = int(round(cx + actual_u))
                cy_dyn = int(round(cy + actual_v))
                
                du = d_du[i]
                dv = d_dv[i]
                
                end_x = int(round(cx_dyn + du * 1.5))
                end_y = int(round(cy_dyn + dv * 1.5))
                
                color = od['color_lut'][color_idxs[i]]
                
                # Draw directly on frame_rgb
                cv2.arrowedLine(frame_rgb, (cx_dyn, cy_dyn), (end_x, end_y), color, 1, tipLength=0.3)
                cv2.circle(frame_rgb, (cx_dyn, cy_dyn), 1, color, -1)
                
        # Draw ROI Tracking elements
        current_frame = int(self.video_thread.cap.get(cv2.CAP_PROP_POS_FRAMES))
        for row in range(self.roi_list.count()):
            item = self.roi_list.item(row)
            widget = self.roi_list.itemWidget(item)
            if widget and widget.btn_vector.isChecked():
                # ensure frame index is within bounds
                idx = min(current_frame, len(widget.data['x']) - 1)
                dx = widget.data['x'][idx]
                dy = widget.data['y'][idx]
                
                # Origin = center of template
                tx, ty, tw, th = widget.data['template_rect']
                cx = int(tx + tw/2)
                cy = int(ty + th/2)
                
                # Apply an arbitrary scaling factor for visibility, let's use global scale if available
                scale = self.spin_scale.value() / 20.0 if hasattr(self, 'spin_scale') else 5.0
                
                end_x = int(cx + dx * scale)
                end_y = int(cy + dy * scale)
                
                # Draw red vector
                cv2.circle(frame_rgb, (cx, cy), 3, (255, 0, 0), -1)
                cv2.arrowedLine(frame_rgb, (cx, cy), (end_x, end_y), (255, 0, 0), 2, tipLength=0.2)
                
                # Also draw the template rect
                cv2.rectangle(frame_rgb, (tx, ty), (tx+tw, ty+th), (255, 0, 0), 1)

        # Convert to QImage
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale pixmap to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update slider (if playing)
        if self.video_thread.playing and self.video_thread.cap:
            current_frame = int(self.video_thread.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider_progress.blockSignals(True)
            self.slider_progress.setValue(current_frame)
            self.slider_progress.blockSignals(False)

    def on_playback_finished(self):
        # Loop playback by default
        self.video_thread.seek(0)
        # If it was playing, keep playing
        if self.btn_play.text() == "Pause":
            self.video_thread.playing = True
        else:
            self.slider_progress.setValue(0)
        
    def on_compute_global(self):
        if not getattr(self.video_thread, 'video_path', None):
            QMessageBox.warning(self, "Error", "No video selected.")
            return
            
        patch_size = self.spin_patch.value()
        freq_min = self.spin_freq_min.value()
        freq_max = self.spin_freq_max.value()
        sal_thresh = self.spin_sal.value()
        
        self.progress_dialog = QProgressDialog("Computing global vector field...", "Cancel", 0, 0, self)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.setWindowTitle("Computing")
        self.progress_dialog.setModal(True)
        self.progress_dialog.show()
        
        self.compute_thread = ComputeThread(self.video_thread.video_path, patch_size, freq_min, freq_max, sal_thresh)
        self.compute_thread.finished_signal.connect(self.on_compute_finished)
        self.compute_thread.start()

    def on_compute_finished(self, success, message):
        self.progress_dialog.close()
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", f"Computation failed:\n{message}")
            self.chk_overlay.setChecked(False)
        
    def get_color_map(self):
        color_lut = []
        for i in range(256):
            h = (1.0 - (i / 255.0)) * 120
            hsv = np.uint8([[[h, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
            # Convert BGR to RGB for PyQt
            color_lut.append((int(bgr[2]), int(bgr[1]), int(bgr[0])))
        return color_lut

    def on_overlay_toggled(self, checked):
        if not checked:
            self.overlay_data = None
            return
            
        if not getattr(self.video_thread, 'video_path', None):
            QMessageBox.warning(self, "Error", "No video selected.")
            self.chk_overlay.setChecked(False)
            return
            
        base_name = os.path.basename(self.video_thread.video_path)
        base_name_no_ext, _ = os.path.splitext(base_name)
        zarr_path = os.path.join("data/results", f"{base_name_no_ext}.zarr")
        
        if not os.path.exists(zarr_path):
            QMessageBox.warning(self, "Error", f"No computation result found for this video.\nPlease compute the global vector field first.\nPath checked: {zarr_path}")
            self.chk_overlay.setChecked(False)
            return
            
        try:
            root = zarr.open(zarr_path, mode='r')
            data = root[:]  # Shape (1, H', W', 5)
            res_view = data[0]
            
            freqs = res_view[..., 0]      # (H', W')
            amp_u = res_view[..., 1]      # (H', W')
            amp_v = res_view[..., 2]      # (H', W')
            phase_u = res_view[..., 3]    # (H', W')
            phase_v = res_view[..., 4]    # (H', W')
            
            active_mask = freqs > 0
            y_idxs, x_idxs = np.where(active_mask)
            
            orig_W = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_H = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _, new_H, new_W, _ = data.shape
            
            if 'patch_size' in root.attrs:
                patch_size = root.attrs['patch_size']
            else:
                patch_size_h = int(np.ceil(orig_H / float(new_H)))
                patch_size_w = int(np.ceil(orig_W / float(new_W)))
                patch_size = max(patch_size_h, patch_size_w)
                if patch_size == 0: patch_size = 1
            
            centers_x = (x_idxs * patch_size + patch_size / 2.0).astype(float)
            centers_y = (y_idxs * patch_size + patch_size / 2.0).astype(float)
            
            act_freqs = freqs[active_mask]
            act_amp_u = amp_u[active_mask]
            act_amp_v = amp_v[active_mask]
            act_phase_u = phase_u[active_mask]
            act_phase_v = phase_v[active_mask]
            
            amps = np.sqrt(act_amp_u**2 + act_amp_v**2)
            p98_amp = np.percentile(amps, 98) if len(amps) > 0 else 1.0
            if p98_amp == 0: p98_amp = 1.0
            
            init_arrow_scale = (patch_size * 2.0) / p98_amp
            
            self.overlay_data = {
                'centers_x': centers_x,
                'centers_y': centers_y,
                'freqs': act_freqs,
                'amp_u': act_amp_u,
                'amp_v': act_amp_v,
                'phase_u': act_phase_u,
                'phase_v': act_phase_v,
                'p98_amp': p98_amp,
                'init_arrow_scale': init_arrow_scale,
                'color_lut': self.get_color_map()
            }
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Zarr data:\n{str(e)}")
            self.chk_overlay.setChecked(False)
            
    def on_add_roi(self):
        if not getattr(self.video_thread, 'video_path', None):
            QMessageBox.warning(self, "Error", "No video selected.")
            return
            
        self.video_thread.playing = False
        self.btn_play.setText("Play")
        
        self.video_label.set_state("DRAW_TEMPLATE")
        QMessageBox.information(self, "ROI", "Please draw the tracking template (small rectangle) on the video.")

    def on_roi_drawn(self, template_rect, search_rect):
        # We need to map from QLabel coordinates to actual video coordinates
        # Get actual video dimensions
        orig_W = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_H = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Get QLabel pixmap dimensions and offset
        pixmap = self.video_label.pixmap()
        if not pixmap:
            return
            
        pw = pixmap.width()
        ph = pixmap.height()
        
        lw = self.video_label.width()
        lh = self.video_label.height()
        
        offset_x = (lw - pw) // 2
        offset_y = (lh - ph) // 2
        
        scale_x = orig_W / pw
        scale_y = orig_H / ph
        
        def map_rect(r):
            x, y, w, h = r
            # map to pixmap
            px = x - offset_x
            py = y - offset_y
            
            # map to original video
            vx = int(px * scale_x)
            vy = int(py * scale_y)
            vw = int(w * scale_x)
            vh = int(h * scale_y)
            return (vx, vy, vw, vh)
            
        actual_template = map_rect(template_rect)
        actual_search = map_rect(search_rect)
        
        self.progress_dialog = QProgressDialog("Computing ROI tracking...", "Cancel", 0, 0, self)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.setWindowTitle("Tracking")
        self.progress_dialog.setModal(True)
        self.progress_dialog.show()
        
        self.roi_thread = RoiTrackingThread(self.video_thread.video_path, actual_template, actual_search)
        self.roi_thread.finished_signal.connect(self.on_roi_tracked)
        self.roi_thread.start()

    def on_roi_tracked(self, result):
        self.progress_dialog.close()
        if not result.get('success', False):
            QMessageBox.critical(self, "Error", f"ROI tracking failed: {result.get('error')}")
            return
            
        # Give it a unique ID based on a running counter
        if not hasattr(self, 'roi_counter'):
            self.roi_counter = 0
            
        self.roi_counter += 1
        roi_id = self.roi_counter
        
        from PyQt6.QtWidgets import QListWidgetItem
        item = QListWidgetItem(self.roi_list)
        widget = RoiItemWidget(roi_id, result)
        
        # Attach item reference to widget for easy removal
        widget.list_item = item
        widget.btn_delete.clicked.connect(lambda: self.on_delete_roi(widget))
        
        item.setSizeHint(widget.sizeHint())
        self.roi_list.addItem(item)
        self.roi_list.setItemWidget(item, widget)
        
    def on_delete_roi(self, widget):
        # Clean up windows
        if widget.xy_window:
            widget.xy_window.close()
        if widget.orbit_window:
            widget.orbit_window.close()
            
        # Remove from list
        row = self.roi_list.row(widget.list_item)
        self.roi_list.takeItem(row)
        
    def closeEvent(self, event):
        self.video_thread.stop()
        self.video_thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Simple dark palette
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)

    window = AnalysisMainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
