import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import StringVar, IntVar, DoubleVar, BooleanVar
from PIL import Image, ImageTk
import threading
import numpy as np
from .realsense_core import RealSenseAppCore

# --- GUI Scaffold for RealSense Camera Control ---

class RealSenseGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RealSense Camera GUI")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.geometry("1600x900")
        self.resizable(True, True)

        # --- Variables for Controls ---
        self.resolution = StringVar(value='640x360')
        self.fps = IntVar(value=30)
        self.min_depth = DoubleVar(value=0.1)
        self.max_depth = DoubleVar(value=5.0)
        self.enable_color = BooleanVar(value=False)
        self.no_auto_range = BooleanVar(value=False)
        self.no_auto_exposure = BooleanVar(value=False)
        self.no_hand_tracking = BooleanVar(value=False)
        self.enable_object_detection = BooleanVar(value=False)
        self.object_model = StringVar(value='mobilenet-ssd')
        self.object_confidence = DoubleVar(value=0.5)
        self.colormap = StringVar(value='RAINBOW')
        self.advanced_depth = BooleanVar(value=False)
        self.no_edge_enhancement = BooleanVar(value=False)
        self.no_highlight = BooleanVar(value=False)
        self.no_3d_viz = BooleanVar(value=False)
        self.no_hands_in_3d = BooleanVar(value=False)

        # --- Layout ---
        self.create_widgets()

        # --- Camera Core and State ---
        self.core = None
        self.camera_running = False
        self.stop_event = threading.Event()
        self.update_interval = 33  # ms, ~30fps
        self.depth_imgtk = None
        self.ir_imgtk = None
        self.viz3d_imgtk = None

        # --- Trace Additions ---
        self.resolution.trace_add("write", self.config_changed)
        self.fps.trace_add("write", self.config_changed)
        self.min_depth.trace_add("write", self.config_changed)
        self.max_depth.trace_add("write", self.config_changed)
        self.enable_color.trace_add("write", self.config_changed)
        self.no_auto_range.trace_add("write", self.config_changed)
        self.no_auto_exposure.trace_add("write", self.config_changed)
        self.no_hand_tracking.trace_add("write", self.config_changed)
        self.enable_object_detection.trace_add("write", self.config_changed)
        self.object_model.trace_add("write", self.config_changed)
        self.object_confidence.trace_add("write", self.config_changed)
        self.colormap.trace_add("write", self.config_changed)
        self.advanced_depth.trace_add("write", self.config_changed)
        self.no_edge_enhancement.trace_add("write", self.config_changed)
        self.no_highlight.trace_add("write", self.config_changed)
        self.no_3d_viz.trace_add("write", self.config_changed)
        self.no_hands_in_3d.trace_add("write", self.config_changed)

    def create_widgets(self):
        # Controls Frame
        controls_frame = ttk.Frame(self)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Camera Controls
        ttk.Label(controls_frame, text="Camera Controls", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0,5))
        ttk.Button(controls_frame, text="Start Camera", command=self.start_camera).pack(fill=tk.X)
        ttk.Button(controls_frame, text="Stop Camera", command=self.stop_camera).pack(fill=tk.X, pady=(0,10))

        ttk.Label(controls_frame, text="Resolution:").pack(anchor=tk.W)
        ttk.Combobox(controls_frame, textvariable=self.resolution, values=['1280x720', '848x480', '640x360', '480x270', '424x240'], state="readonly").pack(fill=tk.X)

        ttk.Label(controls_frame, text="FPS:").pack(anchor=tk.W)
        ttk.Combobox(controls_frame, textvariable=self.fps, values=[5, 15, 30, 60, 90], state="readonly").pack(fill=tk.X)

        ttk.Label(controls_frame, text="Min Depth (m):").pack(anchor=tk.W)
        tk.Scale(controls_frame, variable=self.min_depth, from_=0.1, to=4.0, orient=tk.HORIZONTAL, resolution=0.01).pack(fill=tk.X)
        ttk.Label(controls_frame, text="Max Depth (m):").pack(anchor=tk.W)
        tk.Scale(controls_frame, variable=self.max_depth, from_=0.5, to=10.0, orient=tk.HORIZONTAL, resolution=0.01).pack(fill=tk.X)

        ttk.Checkbutton(controls_frame, text="Enable Color Stream", variable=self.enable_color).pack(anchor=tk.W)
        ttk.Checkbutton(controls_frame, text="Disable Auto Depth Range", variable=self.no_auto_range).pack(anchor=tk.W)
        ttk.Checkbutton(controls_frame, text="Disable Auto Exposure", variable=self.no_auto_exposure).pack(anchor=tk.W)

        # Hand Tracking
        ttk.Label(controls_frame, text="Hand Tracking", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10,0))
        ttk.Checkbutton(controls_frame, text="Disable Hand Tracking", variable=self.no_hand_tracking).pack(anchor=tk.W)

        # Object Detection
        ttk.Label(controls_frame, text="Object Detection", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10,0))
        ttk.Checkbutton(controls_frame, text="Enable Object Detection", variable=self.enable_object_detection).pack(anchor=tk.W)
        ttk.Label(controls_frame, text="Model:").pack(anchor=tk.W)
        ttk.Combobox(controls_frame, textvariable=self.object_model, values=['mobilenet-ssd', 'yolo-tiny', 'mp-objects'], state="readonly").pack(fill=tk.X)
        ttk.Label(controls_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        tk.Scale(controls_frame, variable=self.object_confidence, from_=0.1, to=1.0, orient=tk.HORIZONTAL, resolution=0.01).pack(fill=tk.X)

        # Visualization
        ttk.Label(controls_frame, text="Visualization", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10,0))
        ttk.Label(controls_frame, text="Colormap:").pack(anchor=tk.W)
        ttk.Combobox(controls_frame, textvariable=self.colormap, values=['TURBO', 'JET', 'PLASMA', 'VIRIDIS', 'HOT', 'RAINBOW'], state="readonly").pack(fill=tk.X)
        ttk.Checkbutton(controls_frame, text="Use Advanced Depth Visualization", variable=self.advanced_depth, command=self.toggle_advanced_options).pack(anchor=tk.W)
        self.cb_no_edge = ttk.Checkbutton(controls_frame, text="Disable Edge Enhancement", variable=self.no_edge_enhancement)
        self.cb_no_highlight = ttk.Checkbutton(controls_frame, text="Disable Closest Point Highlighting", variable=self.no_highlight)
        self.cb_no_edge.pack(anchor=tk.W)
        self.cb_no_highlight.pack(anchor=tk.W)
        self.toggle_advanced_options()
        ttk.Checkbutton(controls_frame, text="Disable 3D Visualization", variable=self.no_3d_viz, command=self.toggle_3d_options).pack(anchor=tk.W)
        self.cb_no_hands_3d = ttk.Checkbutton(controls_frame, text="Disable Hands in 3D Visualization", variable=self.no_hands_in_3d)
        self.cb_no_hands_3d.pack(anchor=tk.W)
        self.toggle_3d_options()

        # Status/Info Area
        ttk.Label(controls_frame, text="Status / Info", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10,0))
        self.status_text = tk.Text(controls_frame, height=8, width=32, state=tk.DISABLED, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, pady=(0,10))

        # --- Display Panels ---
        display_frame = ttk.Frame(self)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.depth_panel = tk.Label(display_frame, text="Depth View", bg="#222", fg="#fff")
        self.depth_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.ir_panel = tk.Label(display_frame, text="IR View", bg="#222", fg="#fff")
        self.ir_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.viz3d_panel = tk.Label(display_frame, text="3D Visualization", bg="#222", fg="#fff")
        self.viz3d_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    def toggle_advanced_options(self):
        if self.advanced_depth.get():
            self.cb_no_edge.state(["!disabled"])
            self.cb_no_highlight.state(["!disabled"])
        else:
            self.cb_no_edge.state(["disabled"])
            self.cb_no_highlight.state(["disabled"])

    def toggle_3d_options(self):
        if not self.no_3d_viz.get():
            self.cb_no_hands_3d.state(["!disabled"])
        else:
            self.cb_no_hands_3d.state(["disabled"])

    def collect_config(self):
        return {
            'resolution': self.resolution.get(),
            'fps': self.fps.get(),
            'min_depth': self.min_depth.get(),
            'max_depth': self.max_depth.get(),
            'enable_color': self.enable_color.get(),
            'no_auto_range': self.no_auto_range.get(),
            'no_auto_exposure': self.no_auto_exposure.get(),
            'no_hand_tracking': self.no_hand_tracking.get(),
            'enable_object_detection': self.enable_object_detection.get(),
            'object_model': self.object_model.get(),
            'object_confidence': self.object_confidence.get(),
            'colormap': self.colormap.get(),
            'advanced_depth': self.advanced_depth.get(),
            'no_edge_enhancement': self.no_edge_enhancement.get(),
            'no_highlight': self.no_highlight.get(),
            'no_3d_viz': self.no_3d_viz.get(),
            'no_hands_in_3d': self.no_hands_in_3d.get(),
        }

    def config_changed(self, *args):
        if self.camera_running and self.core:
            config = self.collect_config()
            self.core.update_config(config)

    def start_camera(self):
        if self.camera_running:
            self.append_status("[INFO] Camera already running.")
            return
        config = self.collect_config()
        self.core = RealSenseAppCore(config)
        self.core.start()
        self.camera_running = True
        self.append_status("[INFO] Camera started.")
        self.after(self.update_interval, self.update_frames)

    def stop_camera(self):
        if self.core:
            self.core.stop()
            self.core = None
        self.camera_running = False
        self.append_status("[INFO] Camera stopped.")

    def update_frames(self):
        if not self.camera_running or not self.core:
            return
        depth, ir, viz3d = self.core.get_latest_frames()
        self.depth_imgtk = self.cv_to_imgtk(depth)
        self.ir_imgtk = self.cv_to_imgtk(ir)
        self.viz3d_imgtk = self.cv_to_imgtk(viz3d)
        self.depth_panel.config(image=self.depth_imgtk)
        self.ir_panel.config(image=self.ir_imgtk)
        self.viz3d_panel.config(image=self.viz3d_imgtk)
        status, objects = self.core.get_status_info()
        self.set_status(status, objects)
        self.after(self.update_interval, self.update_frames)

    def cv_to_imgtk(self, img):
        if img is None or not isinstance(img, np.ndarray):
            img = np.zeros((360, 640, 3), dtype=np.uint8)
        img = np.ascontiguousarray(img)
        im = Image.fromarray(img)
        return ImageTk.PhotoImage(im)

    def set_status(self, status, objects):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, status + "\n")
        if objects:
            for obj in objects:
                self.status_text.insert(tk.END, str(obj) + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def append_status(self, msg):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, msg + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def on_close(self):
        self.stop_camera()
        self.destroy()

if __name__ == "__main__":
    app = RealSenseGUI()
    app.mainloop() 