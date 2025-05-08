import cv2
import numpy as np
import threading
import time
import logging
import sys
import os
from realsense.wrapper import PyRealSense

# Optionally import mediapipe if available
try:
    import mediapipe as mp
except ImportError:
    mp = None

# Helper functions (copied and adapted from example.py)
def draw_hand_landmarks(image, results, color=(0, 255, 0)):
    if mp is None or results is None:
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    return image

def detect_gesture(ir_frame, hands):
    if mp is None or hands is None:
        return None
    ir_rgb = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2RGB)
    results = hands.process(ir_rgb)
    return results

def draw_subtitle(image: np.ndarray, text: str, position: tuple = (10, 30)) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        test_line = ' '.join(current_line)
        (width, height), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if width > image.shape[1] - 40:
            lines.append(' '.join(current_line[:-1]))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    y = position[1]
    for line in lines:
        (width, height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(image, (position[0] - 5, y - height - 5), (position[0] + width + 5, y + 5), bg_color, -1)
        cv2.putText(image, line, (position[0], y), font, font_scale, color, thickness)
        y += height + 10
    return image

def create_enhanced_depth_colormap(depth_frame, min_depth, max_depth, colormap_type=cv2.COLORMAP_TURBO):
    """Basic, reliable depth colormap with minimal processing."""
    if depth_frame is None or not isinstance(depth_frame, np.ndarray) or depth_frame.size == 0:
        h, w = 480, 640
        if isinstance(depth_frame, np.ndarray) and depth_frame.ndim >= 2:
            h, w = depth_frame.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)
    min_depth_mm = min_depth * 1000.0
    max_depth_mm = max_depth * 1000.0
    depth_normalized = np.zeros_like(depth_frame, dtype=np.uint8)
    valid_mask = (depth_frame > 0) & (depth_frame >= min_depth_mm) & (depth_frame <= max_depth_mm)
    if np.any(valid_mask):
        depth_range = max_depth_mm - min_depth_mm
        if depth_range > 0:
            depth_normalized[valid_mask] = np.clip(
                255 * (max_depth_mm - depth_frame[valid_mask]) / depth_range,
                0, 255
            ).astype(np.uint8)
    colormap = cv2.applyColorMap(depth_normalized, colormap_type)
    colormap[~valid_mask] = [30, 30, 30]
    edges = cv2.Canny(depth_normalized, 50, 150)
    edge_mask = edges > 0
    colormap[edge_mask] = [255, 255, 255]
    return colormap

def create_advanced_depth_colormap(depth_frame, min_depth, max_depth, colormap_type=cv2.COLORMAP_TURBO, enhance_edges=True, highlight_closest=True):
    """Advanced depth colormap with edge enhancement and closest point highlighting."""
    if depth_frame is None or not isinstance(depth_frame, np.ndarray) or depth_frame.size == 0:
        h, w = 480, 640
        if isinstance(depth_frame, np.ndarray) and depth_frame.ndim >= 2:
            h, w = depth_frame.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)
    min_depth_mm = min_depth * 1000.0
    max_depth_mm = max_depth * 1000.0
    depth_normalized = np.zeros_like(depth_frame, dtype=np.uint8)
    valid_mask = (depth_frame > 0) & (depth_frame >= min_depth_mm) & (depth_frame <= max_depth_mm)
    if np.any(valid_mask):
        depth_range = max_depth_mm - min_depth_mm
        if depth_range > 0:
            depth_normalized[valid_mask] = np.clip(
                255 * (max_depth_mm - depth_frame[valid_mask]) / depth_range,
                0, 255
            ).astype(np.uint8)
    colormap = cv2.applyColorMap(depth_normalized, colormap_type)
    colormap[~valid_mask] = [30, 30, 30]
    if enhance_edges:
        try:
            filtered = cv2.bilateralFilter(depth_normalized, 5, 50, 50)
            edges = cv2.Canny(filtered, 30, 100)
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edge_mask = edges > 0
            colormap[edge_mask] = [255, 255, 255]
        except Exception:
            edges = cv2.Canny(depth_normalized, 50, 150)
            edge_mask = edges > 0
            colormap[edge_mask] = [255, 255, 255]
    if highlight_closest and np.any(valid_mask):
        try:
            close_threshold = np.percentile(depth_frame[valid_mask], 5)
            closest_mask = (depth_frame <= close_threshold) & valid_mask
            if np.any(closest_mask):
                import time
                pulse = (np.sin(time.time() * 5) + 1) / 2
                highlight_color = np.array([0, int(155 + 100 * pulse), 255], dtype=np.uint8)
                highlight_mask = np.zeros_like(depth_frame, dtype=bool)
                highlight_mask[closest_mask] = True
                kernel = np.ones((5, 5), np.uint8)
                highlight_mask = cv2.dilate(highlight_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                alpha = 0.7
                colormap[highlight_mask] = (colormap[highlight_mask] * (1 - alpha) + highlight_color * alpha).astype(np.uint8)
        except Exception:
            pass
    return colormap

def get_depth_info(depth_frame):
    """Get simple depth information from the depth frame"""
    if depth_frame is None:
        return "No depth data available"
    valid_depth = depth_frame[depth_frame > 0]
    if len(valid_depth) > 0:
        avg_depth = np.mean(valid_depth) / 1000.0
        min_depth = np.min(valid_depth) / 1000.0
        max_depth = np.max(valid_depth) / 1000.0
        return f"Objects at {min_depth:.2f}-{max_depth:.2f}m, average {avg_depth:.2f}m"
    else:
        return "No valid depth data"

def create_3d_visualization(hand_results=None, depth_frame=None, detected_objects=None, width=640, height=480):
    visualization = np.zeros((height, width, 3), dtype=np.uint8)
    visualization[:, :] = [30, 30, 30]
    cv2.putText(visualization, "3D View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    axes_length = 100
    origin = (width // 2, height // 2)
    cv2.line(visualization, origin, (origin[0] + axes_length, origin[1]), (0, 0, 255), 2)
    cv2.putText(visualization, "X", (origin[0] + axes_length + 10, origin[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.line(visualization, origin, (origin[0], origin[1] - axes_length), (0, 255, 0), 2)
    cv2.putText(visualization, "Y", (origin[0], origin[1] - axes_length - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.line(visualization, origin, (origin[0] + axes_length//2, origin[1] + axes_length//2), (255, 0, 0), 2)
    cv2.putText(visualization, "Z", (origin[0] + axes_length//2 + 5, origin[1] + axes_length//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if (hand_results is None or not hasattr(hand_results, 'multi_hand_landmarks') or not hand_results.multi_hand_landmarks) and (detected_objects is None or len(detected_objects) == 0):
        cv2.putText(visualization, "No hands or objects detected", (width//2 - 120, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return visualization
    if hand_results is not None and hasattr(hand_results, 'multi_hand_landmarks') and hand_results.multi_hand_landmarks:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (0, 5), (5, 9), (9, 13), (13, 17)
        ]
        for hand_landmarks in hand_results.multi_hand_landmarks:
            points_3d = []
            for i, landmark in enumerate(hand_landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                if depth_frame is not None:
                    pixel_x = int(x * depth_frame.shape[1])
                    pixel_y = int(y * depth_frame.shape[0])
                    if 0 <= pixel_x < depth_frame.shape[1] and 0 <= pixel_y < depth_frame.shape[0]:
                        pixel_depth = depth_frame[pixel_y, pixel_x]
                        if pixel_depth > 0:
                            z = pixel_depth / 5000.0
                proj_x = int(origin[0] + x * width * 0.4 - z * width * 0.1)
                proj_y = int(origin[1] + y * height * 0.4)
                points_3d.append((proj_x, proj_y))
                cv2.circle(visualization, (proj_x, proj_y), 5, (100, 100, 255), -1)
                if i == 0 or i == 4 or i == 8 or i == 12 or i == 16 or i == 20:
                    cv2.putText(visualization, str(i), (proj_x + 5, proj_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(points_3d) and end_idx < len(points_3d):
                    cv2.line(visualization, points_3d[start_idx], points_3d[end_idx], (0, 255, 0), 2)
    if detected_objects is not None and len(detected_objects) > 0 and depth_frame is not None:
        object_colors = {
            "person": (255, 200, 0),
            "chair": (0, 255, 255),
            "bottle": (255, 0, 255),
            "cup": (200, 100, 255),
            "default": (0, 200, 200)
        }
        for label, confidence, bbox in detected_objects:
            point_cloud = extract_object_point_cloud(depth_frame, bbox)
            if point_cloud is not None:
                x_coords, y_coords, z_coords = point_cloud
                color = object_colors.get(label.lower(), object_colors["default"])
                for i in range(len(x_coords)):
                    x_norm = x_coords[i] / depth_frame.shape[1]
                    y_norm = y_coords[i] / depth_frame.shape[0]
                    z_norm = z_coords[i] / 5.0
                    proj_x = int(origin[0] + x_norm * width * 0.4 - z_norm * width * 0.1)
                    proj_y = int(origin[1] + y_norm * height * 0.4)
                    if 0 <= proj_x < width and 0 <= proj_y < height:
                        cv2.circle(visualization, (proj_x, proj_y), 1, color, -1)
                label_pos_x = int(origin[0] + (np.mean(x_coords) / depth_frame.shape[1]) * width * 0.4)
                label_pos_y = int(origin[1] + (np.mean(y_coords) / depth_frame.shape[0]) * height * 0.4)
                cv2.putText(visualization, label, (label_pos_x, label_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    legend_y = height - 120
    cv2.putText(visualization, "Legend:", (width - 150, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    legend_y += 20
    if hand_results is not None and hasattr(hand_results, 'multi_hand_landmarks') and hand_results.multi_hand_landmarks:
        cv2.putText(visualization, "Hand Joints", (width - 150, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        legend_y += 20
        cv2.putText(visualization, "Connections", (width - 150, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        legend_y += 20
    if detected_objects is not None and len(detected_objects) > 0:
        for label, _, _ in detected_objects:
            if legend_y < height - 10:
                color = object_colors.get(label.lower(), object_colors["default"])
                cv2.putText(visualization, label, (width - 150, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                legend_y += 20
    return visualization

def detect_objects(frame, confidence_threshold=0.5, model_name="mobilenet-ssd"):
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return []
    if len(frame.shape) == 2:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # For now, only support OpenCV DNN-based models (mobilenet-ssd, yolo-tiny)
    global object_detection_model, object_detection_labels
    if model_name in ["mobilenet-ssd", "yolo-tiny"]:
        if ('current_model' not in globals() or globals()['current_model'] != model_name or 
            'object_detection_model' not in globals() or object_detection_model is None):
            try:
                model_path = os.path.join(os.path.dirname(__file__), "models")
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                if model_name == "mobilenet-ssd":
                    model_weights = os.path.join(model_path, "mobilenet_ssd.caffemodel")
                    model_config = os.path.join(model_path, "mobilenet_ssd.prototxt")
                    if not os.path.exists(model_weights) or not os.path.exists(model_config):
                        weight_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
                        config_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt"
                        import urllib.request
                        def download_file(url, destination):
                            try:
                                urllib.request.urlretrieve(url, destination)
                                return True
                            except Exception as e:
                                print(f"Error downloading {url}: {e}")
                                return False
                        if not download_file(weight_url, model_weights) or not download_file(config_url, model_config):
                            raise Exception("Failed to download model files")
                    object_detection_model = cv2.dnn.readNetFromCaffe(model_config, model_weights)
                    object_detection_labels = ["background", "aeroplane", "bicycle", "bird", "boat",
                                             "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                             "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                             "sofa", "train", "tvmonitor"]
                elif model_name == "yolo-tiny":
                    model_weights = os.path.join(model_path, "yolov4-tiny.weights")
                    model_config = os.path.join(model_path, "yolov4-tiny.cfg")
                    class_names = os.path.join(model_path, "coco.names")
                    if not os.path.exists(model_weights) or not os.path.exists(model_config) or not os.path.exists(class_names):
                        weight_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
                        config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
                        names_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
                        import urllib.request
                        def download_file(url, destination):
                            try:
                                urllib.request.urlretrieve(url, destination)
                                return True
                            except Exception as e:
                                print(f"Error downloading {url}: {e}")
                                return False
                        if (not download_file(weight_url, model_weights) or 
                            not download_file(config_url, model_config) or 
                            not download_file(names_url, class_names)):
                            raise Exception("Failed to download model files")
                    object_detection_model = cv2.dnn.readNetFromDarknet(model_config, model_weights)
                    with open(class_names, 'rt') as f:
                        object_detection_labels = f.read().rstrip('\n').split('\n')
                try:
                    print(f"Using CPU for {model_name} (GPU acceleration disabled)")
                except:
                    print(f"GPU acceleration not available, using CPU for {model_name}")
                globals()['current_model'] = model_name
                print(f"Object detection model {model_name} loaded successfully")
            except Exception as e:
                print(f"Error initializing object detection model {model_name}: {e}")
                object_detection_model = None
                object_detection_labels = []
                return []
        if 'object_detection_model' not in globals() or object_detection_model is None:
            return []
        height, width = frame_rgb.shape[:2]
        if model_name == "mobilenet-ssd":
            blob = cv2.dnn.blobFromImage(frame_rgb, 0.007843, (300, 300), 127.5)
            object_detection_model.setInput(blob)
            detections = object_detection_model.forward()
            detected_objects = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    class_id = int(detections[0, 0, i, 1])
                    if class_id < len(object_detection_labels):
                        label = object_detection_labels[class_id]
                        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                        (startX, startY, endX, endY) = box.astype("int")
                        detected_objects.append((label, confidence, (startX, startY, endX, endY)))
        elif model_name == "yolo-tiny":
            blob = cv2.dnn.blobFromImage(frame_rgb, 1/255.0, (416, 416), swapRB=True, crop=False)
            object_detection_model.setInput(blob)
            layer_names = object_detection_model.getLayerNames()
            output_layers = [layer_names[i - 1] for i in object_detection_model.getUnconnectedOutLayers()]
            outputs = object_detection_model.forward(output_layers)
            boxes = []
            confidences = []
            class_ids = []
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
            detected_objects = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    label = object_detection_labels[class_ids[i]]
                    confidence = confidences[i]
                    detected_objects.append((label, confidence, (x, y, x + w, y + h)))
        return detected_objects
    print(f"Unknown model: {model_name}. Use 'mobilenet-ssd' or 'yolo-tiny'")
    return []

def draw_detected_objects(image, detected_objects, depth_frame=None):
    for label, confidence, bbox in detected_objects:
        startX, startY, endX, endY = bbox
        depth_text = ""
        if depth_frame is not None:
            height, width = depth_frame.shape
            valid_startX = max(0, startX)
            valid_startY = max(0, startY)
            valid_endX = min(width, endX)
            valid_endY = min(height, endY)
            roi_depth = depth_frame[valid_startY:valid_endY, valid_startX:valid_endX]
            valid_depths = roi_depth[roi_depth > 0]
            if len(valid_depths) > 0:
                avg_depth = np.mean(valid_depths) / 1000.0
                depth_text = f" @ {avg_depth:.2f}m"
        color = (0, 255, 0)
        if depth_text:
            depth_value = float(depth_text.split('@')[1].split('m')[0])
            if depth_value < 1.0:
                color = (0, 0, 255)
            elif depth_value < 2.0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        text = f"{label}: {confidence:.2f}{depth_text}"
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def extract_object_point_cloud(depth_frame, bbox, max_points=1000):
    if depth_frame is None:
        return None
    startX, startY, endX, endY = bbox
    height, width = depth_frame.shape
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(width, endX)
    endY = min(height, endY)
    roi_depth = depth_frame[startY:endY, startX:endX]
    valid_mask = roi_depth > 0
    if not np.any(valid_mask):
        return None
    valid_depths = roi_depth[valid_mask]
    y_indices, x_indices = np.where(valid_mask)
    x_coords = x_indices + startX
    y_coords = y_indices + startY
    z_coords = valid_depths / 1000.0
    if len(x_coords) > max_points:
        indices = np.random.choice(len(x_coords), max_points, replace=False)
        x_coords = x_coords[indices]
        y_coords = y_coords[indices]
        z_coords = z_coords[indices]
    return x_coords, y_coords, z_coords

# Main RealSenseAppCore class
class RealSenseAppCore:
    def __init__(self, config):
        self.config = config
        self.camera_thread = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_depth = np.zeros((360, 640, 3), dtype=np.uint8)
        self.latest_ir = np.zeros((360, 640, 3), dtype=np.uint8)
        self.latest_3d = np.zeros((360, 640, 3), dtype=np.uint8)
        self.status_info = "Not started"
        self.detected_objects = []
        self._init_helpers()
        self._init_logger()
        self._reset_state()

    def _init_helpers(self):
        # Import or define all helper functions here
        global draw_hand_landmarks, detect_gesture, draw_subtitle
        # ... (other helpers as above) ...
        pass

    def _init_logger(self):
        self.logger = logging.getLogger('depth_app_gui')
        self.logger.setLevel(logging.INFO)

    def _reset_state(self):
        self.hands = None
        self.rs = None
        self.colormap_choices = {
            'TURBO': cv2.COLORMAP_TURBO,
            'JET': cv2.COLORMAP_JET,
            'PLASMA': cv2.COLORMAP_PLASMA,
            'VIRIDIS': cv2.COLORMAP_VIRIDIS,
            'HOT': cv2.COLORMAP_HOT,
            'RAINBOW': cv2.COLORMAP_RAINBOW
        }
        self.selected_colormap = self.colormap_choices.get(self.config.get('colormap', 'RAINBOW').upper(), cv2.COLORMAP_RAINBOW)
        self.depth_min_running = self.config.get('min_depth', 0.1)
        self.depth_max_running = self.config.get('max_depth', 5.0)
        self.auto_range = not self.config.get('no_auto_range', False)
        self.latest_hand_landmarks = None
        self.latest_detected_objects = []

    def start(self):
        if self.running:
            return
        self.running = True
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()

    def stop(self):
        self.running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        # Cleanup
        with self.lock:
            if self.rs:
                self.rs.stop()
                self.rs = None
            if self.hands:
                self.hands.close()
                self.hands = None

    def update_config(self, config):
        with self.lock:
            self.config = config
            # Update live settings
            self.depth_min_running = config.get('min_depth', self.depth_min_running)
            self.depth_max_running = config.get('max_depth', self.depth_max_running)
            self.selected_colormap = self.colormap_choices.get(self.config.get('colormap', 'RAINBOW').upper(), cv2.COLORMAP_RAINBOW)
            self.auto_range = not self.config.get('no_auto_range', False)
            # Optionally, add more live-updatable settings here

    def get_latest_frames(self):
        with self.lock:
            return self.latest_depth.copy(), self.latest_ir.copy(), self.latest_3d.copy()

    def get_status_info(self):
        with self.lock:
            return self.status_info, self.latest_detected_objects

    def _camera_loop(self):
        # Parse config
        width, height = map(int, self.config.get('resolution', '640x360').split('x'))
        fps = self.config.get('fps', 30)
        min_depth = self.config.get('min_depth', 0.1)
        max_depth = self.config.get('max_depth', 5.0)
        enable_color = self.config.get('enable_color', False)
        no_hand_tracking = self.config.get('no_hand_tracking', False)
        advanced_depth = self.config.get('advanced_depth', False)
        no_edge_enhancement = self.config.get('no_edge_enhancement', False)
        no_highlight = self.config.get('no_highlight', False)
        no_3d_viz = self.config.get('no_3d_viz', False)
        enable_object_detection = self.config.get('enable_object_detection', False)
        object_model = self.config.get('object_model', 'mobilenet-ssd')
        object_confidence = self.config.get('object_confidence', 0.5)
        no_hands_in_3d = self.config.get('no_hands_in_3d', False)
        # ... (other config as needed) ...

        # Initialize RealSense
        try:
            self.rs = PyRealSense(
                width=width,
                height=height,
                framerate=fps,
                enable_color=enable_color,
                enable_ir=True,
                min_depth=min_depth,
                max_depth=max_depth
            )
            self.rs.start()
        except Exception as e:
            with self.lock:
                self.status_info = f"Error initializing RealSense: {e}"
            return

        # Initialize MediaPipe Hands if enabled
        if not no_hand_tracking and mp is not None:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.hands = None

        # Main loop
        while self.running:
            try:
                frames = self.rs.get_frames()
                if frames is None:
                    continue
                depth_frame = frames.get('depth')
                ir_frame = frames.get('infrared')
                color_frame = frames.get('color')
                if depth_frame is None or ir_frame is None:
                    continue
                if not isinstance(depth_frame, np.ndarray) or not isinstance(ir_frame, np.ndarray):
                    continue
                if depth_frame.size == 0 or ir_frame.size == 0:
                    continue
                if np.isnan(depth_frame).any() or np.isnan(ir_frame).any():
                    depth_frame = np.nan_to_num(depth_frame)
                    ir_frame = np.nan_to_num(ir_frame)
                # Auto-range
                if self.auto_range:
                    valid_depths = depth_frame[depth_frame > 0]
                    if len(valid_depths) > 0:
                        valid_depths_m = valid_depths / 1000.0
                        min_depth_scene = np.percentile(valid_depths_m, 2)
                        max_depth_scene = np.percentile(valid_depths_m, 98)
                        scene_range = max_depth_scene - min_depth_scene
                        if min_depth_scene < self.depth_min_running:
                            self.depth_min_running = 0.7 * self.depth_min_running + 0.3 * min_depth_scene
                        else:
                            self.depth_min_running = 0.9 * self.depth_min_running + 0.1 * min_depth_scene
                        if max_depth_scene > self.depth_max_running:
                            self.depth_max_running = 0.7 * self.depth_max_running + 0.3 * max_depth_scene
                        else:
                            self.depth_max_running = 0.9 * self.depth_max_running + 0.1 * max_depth_scene
                        min_range = max(0.5, scene_range * 0.2)
                        if self.depth_max_running - self.depth_min_running < min_range:
                            mid_point = (self.depth_max_running + self.depth_min_running) / 2
                            self.depth_min_running = mid_point - min_range / 2
                            self.depth_max_running = mid_point + min_range / 2
                else:
                    self.depth_min_running = min_depth
                    self.depth_max_running = max_depth
                # Depth info
                current_description = get_depth_info(depth_frame)
                # Depth colormap
                if advanced_depth:
                    depth_colormap = create_advanced_depth_colormap(
                        depth_frame,
                        self.depth_min_running,
                        self.depth_max_running,
                        self.selected_colormap,
                        enhance_edges=not no_edge_enhancement,
                        highlight_closest=not no_highlight
                    )
                else:
                    depth_colormap = create_enhanced_depth_colormap(
                        depth_frame,
                        self.depth_min_running,
                        self.depth_max_running,
                        self.selected_colormap
                    )
                ir_viz = ir_frame.copy()
                if len(ir_viz.shape) == 2:
                    ir_viz = cv2.cvtColor(ir_viz, cv2.COLOR_GRAY2BGR)
                depth_viz = depth_colormap.copy()
                # Hand tracking
                hand_results = None
                if not no_hand_tracking and self.hands is not None:
                    hand_results = detect_gesture(ir_frame, self.hands)
                    ir_viz = draw_hand_landmarks(ir_viz, hand_results)
                    depth_viz = draw_hand_landmarks(depth_viz, hand_results)
                # Object detection
                latest_detected_objects = []
                if enable_object_detection:
                    detection_frame = color_frame if color_frame is not None and color_frame.size > 0 else ir_viz
                    if detection_frame is not None and detection_frame.size > 0:
                        latest_detected_objects = detect_objects(
                            detection_frame,
                            object_confidence,
                            object_model
                        )
                        if latest_detected_objects:
                            ir_viz = draw_detected_objects(ir_viz, latest_detected_objects, depth_frame)
                # 3D visualization
                viz_3d = create_3d_visualization(
                    hand_results=None if no_hands_in_3d else hand_results,
                    depth_frame=depth_frame,
                    detected_objects=latest_detected_objects if enable_object_detection else None,
                    width=depth_viz.shape[1],
                    height=depth_viz.shape[0]
                )
                # Update shared state
                with self.lock:
                    self.latest_depth = depth_viz
                    self.latest_ir = ir_viz
                    self.latest_3d = viz_3d
                    self.status_info = current_description
                    self.latest_detected_objects = latest_detected_objects
            except Exception as e:
                with self.lock:
                    self.status_info = f"Error in camera loop: {e}"
                time.sleep(0.1)
        # Cleanup
        with self.lock:
            if self.rs:
                self.rs.stop()
                self.rs = None
            if self.hands:
                self.hands.close()
                self.hands = None 