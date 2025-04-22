import cv2
import numpy as np
import argparse
import os
import time
from realsense.wrapper import PyRealSense
import logging
import sys
import threading
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hand_visualization import create_hand_figure, update_hand_visualization, default_coordinate_conversion

# Add new imports for object detection
# Note: We'll use MediaPipe's object detection if it's available
# If not, we'll fall back to OpenCV's DNN module with MobileNet SSD

# Parse arguments first to check if hand tracking is disabled
parser = argparse.ArgumentParser(description='RealSense Depth Camera Example')
parser.add_argument('--no-hand-tracking', action='store_true', help='Disable hand tracking and gesture detection')
parser.add_argument('--no-3d-viz', action='store_true', help='Disable 3D visualization of hand tracking')
parser.add_argument('--enable-object-detection', action='store_true', help='Enable object detection and point cloud visualization')
parser.add_argument('--no-hands-in-3d', action='store_true', help='Disable hands in 3D visualization (objects only)')
args, _ = parser.parse_known_args()

# Import mediapipe only if hand tracking is enabled
mp = None
if not args.no_hand_tracking:
    try:
        import mediapipe as mp
        print("MediaPipe imported successfully")
    except ImportError:
        print("WARNING: MediaPipe could not be imported. Hand tracking will be disabled.")
        args.no_hand_tracking = True
        mp = None

# Configure logging to suppress most messages
logging.basicConfig(level=logging.ERROR)

# Redirect stdout for the realsense module to prevent debug prints
class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        # Only write complete lines
        for line in buf.rstrip().splitlines():
            if line.strip():
                self.level(line.rstrip())
    
    def flush(self):
        pass

# Only show ERROR and above from realsense module
realsense_logger = logging.getLogger('realsense')
realsense_logger.setLevel(logging.ERROR)

# Create custom logger for the script
logger = logging.getLogger('depth_app')
logger.setLevel(logging.INFO)

# Global variables for sharing data between threads and storing detected objects
latest_hand_landmarks = None
latest_depth_frame = None
latest_detected_objects = []  # Will store (label, confidence, bounding_box) tuples
app_running = True

def draw_hand_landmarks(image, results, color=(0, 255, 0)):
    """Draw hand landmarks and connections on the image."""
    if mp is None or results is None:
        # If MediaPipe is not available, return the image as is
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
        
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Convert single channel to BGR if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    
    return image

def detect_gesture(ir_frame, hands):
    """Detect hand gestures using MediaPipe Hands."""
    if mp is None or hands is None:
        # If MediaPipe is not available, return None
        return None
        
    # Convert IR frame to RGB for MediaPipe
    ir_rgb = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2RGB)
    
    # Process frame
    results = hands.process(ir_rgb)
    
    # Debug print for hand detection
    if results.multi_hand_landmarks:
        logger.debug(f"Detected {len(results.multi_hand_landmarks)} hands")
    
    return results

def draw_subtitle(image: np.ndarray, text: str, position: tuple = (10, 30)) -> np.ndarray:
    """Draw subtitle text on the image with a background for better visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)     # Black background
    
    # Split text into lines if it's too long
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        test_line = ' '.join(current_line)
        (width, height), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if width > image.shape[1] - 40:  # Leave some margin
            lines.append(' '.join(current_line[:-1]))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Draw each line with background
    y = position[1]
    for line in lines:
        (width, height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        # Draw background rectangle
        cv2.rectangle(image, 
                     (position[0] - 5, y - height - 5),
                     (position[0] + width + 5, y + 5),
                     bg_color, -1)
        # Draw text
        cv2.putText(image, line, (position[0], y), font, font_scale, color, thickness)
        y += height + 10
    
    return image

def create_enhanced_depth_colormap(depth_frame, min_depth, max_depth, colormap_type=cv2.COLORMAP_TURBO):
    """Basic, reliable depth colormap with minimal processing."""
    # Just in case depth_frame is None or invalid
    if depth_frame is None or not isinstance(depth_frame, np.ndarray) or depth_frame.size == 0:
        # Return a black image of specified size or default
        h, w = 480, 640
        if isinstance(depth_frame, np.ndarray) and depth_frame.ndim >= 2:
            h, w = depth_frame.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)

    # Convert min/max from meters to millimeters
    min_depth_mm = min_depth * 1000.0
    max_depth_mm = max_depth * 1000.0
    
    # Make a copy to avoid modifying the original
    depth_normalized = np.zeros_like(depth_frame, dtype=np.uint8)
    
    # Only include pixels with valid depth and in range
    valid_mask = (depth_frame > 0) & (depth_frame >= min_depth_mm) & (depth_frame <= max_depth_mm)
    
    # Check if we have any valid data
    if np.any(valid_mask):
        # Apply very simple linear normalization (avoids any complicated calculations)
        # Scale from min_depth to max_depth linearly to 0-255
        depth_range = max_depth_mm - min_depth_mm
        if depth_range > 0:  # Protect against division by zero
            # Map depth values to 0-255 range
            depth_normalized[valid_mask] = np.clip(
                255 * (max_depth_mm - depth_frame[valid_mask]) / depth_range,
                0, 255
            ).astype(np.uint8)
    
    # Apply chosen colormap
    colormap = cv2.applyColorMap(depth_normalized, colormap_type)
    
    # Mark invalid areas as dark gray
    colormap[~valid_mask] = [30, 30, 30]
    
    # Optional: Apply a slight edge enhancement to emphasize object boundaries
    edges = cv2.Canny(depth_normalized, 50, 150)
    edge_mask = edges > 0
    colormap[edge_mask] = [255, 255, 255]  # Highlight edges in white
    
    return colormap

# Add a new depth enhancement function with customizable options
def create_advanced_depth_colormap(depth_frame, min_depth, max_depth, colormap_type=cv2.COLORMAP_TURBO, 
                                  enhance_edges=True, highlight_closest=True):
    """Advanced depth colormap with edge enhancement and closest point highlighting."""
    # Just in case depth_frame is None or invalid
    if depth_frame is None or not isinstance(depth_frame, np.ndarray) or depth_frame.size == 0:
        # Return a black image of specified size or default
        h, w = 480, 640
        if isinstance(depth_frame, np.ndarray) and depth_frame.ndim >= 2:
            h, w = depth_frame.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)

    # Convert min/max from meters to millimeters
    min_depth_mm = min_depth * 1000.0
    max_depth_mm = max_depth * 1000.0
    
    # Make a copy to avoid modifying the original
    depth_normalized = np.zeros_like(depth_frame, dtype=np.uint8)
    
    # Only include pixels with valid depth and in range
    valid_mask = (depth_frame > 0) & (depth_frame >= min_depth_mm) & (depth_frame <= max_depth_mm)
    
    # Check if we have any valid data
    if np.any(valid_mask):
        # Apply very simple linear normalization (avoids any complicated calculations)
        # Scale from min_depth to max_depth linearly to 0-255
        depth_range = max_depth_mm - min_depth_mm
        if depth_range > 0:  # Protect against division by zero
            # Map depth values to 0-255 range
            depth_normalized[valid_mask] = np.clip(
                255 * (max_depth_mm - depth_frame[valid_mask]) / depth_range,
                0, 255
            ).astype(np.uint8)
    
    # Apply chosen colormap
    colormap = cv2.applyColorMap(depth_normalized, colormap_type)
    
    # Mark invalid areas as dark gray
    colormap[~valid_mask] = [30, 30, 30]
    
    # Optional edge enhancement
    if enhance_edges:
        try:
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(depth_normalized, 5, 50, 50)
            # Detect edges using Canny
            edges = cv2.Canny(filtered, 30, 100)
            # Dilate edges to make them more visible
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            # Overlay edges on colormap
            edge_mask = edges > 0
            colormap[edge_mask] = [255, 255, 255]  # White edges
        except Exception as e:
            # Fallback to simpler edge detection if advanced method fails
            edges = cv2.Canny(depth_normalized, 50, 150)
            edge_mask = edges > 0
            colormap[edge_mask] = [255, 255, 255]
    
    # Optional highlight of closest points
    if highlight_closest and np.any(valid_mask):
        try:
            # Find the closest 5% of valid points
            close_threshold = np.percentile(depth_frame[valid_mask], 5)
            closest_mask = (depth_frame <= close_threshold) & valid_mask
            
            # Only proceed if we have closest points
            if np.any(closest_mask):
                # Apply a pulsating highlight effect based on time
                pulse = (np.sin(time.time() * 5) + 1) / 2  # Value between 0 and 1
                highlight_color = np.array([0, int(155 + 100 * pulse), 255], dtype=np.uint8)
                
                # Create a dilated mask for the highlight area
                highlight_mask = np.zeros_like(depth_frame, dtype=bool)
                highlight_mask[closest_mask] = True
                kernel = np.ones((5, 5), np.uint8)
                highlight_mask = cv2.dilate(highlight_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                
                # Apply highlight with alpha blending
                alpha = 0.7
                colormap[highlight_mask] = (colormap[highlight_mask] * (1 - alpha) + highlight_color * alpha).astype(np.uint8)
        except Exception as e:
            # Silently fail if highlighting doesn't work
            pass
    
    return colormap

def get_depth_info(depth_frame):
    """Get simple depth information from the depth frame"""
    if depth_frame is None:
        return "No depth data available"
    
    valid_depth = depth_frame[depth_frame > 0]
    if len(valid_depth) > 0:
        # Convert to meters
        avg_depth = np.mean(valid_depth) / 1000.0
        min_depth = np.min(valid_depth) / 1000.0
        max_depth = np.max(valid_depth) / 1000.0
        
        return f"Objects at {min_depth:.2f}-{max_depth:.2f}m, average {avg_depth:.2f}m"
    else:
        return "No valid depth data"

# Update create_3d_visualization function to fix upside-down display
def create_3d_visualization(hand_results=None, depth_frame=None, detected_objects=None, width=640, height=480):
    """Create a 3D visualization of hands and detected objects"""
    # Create a blank image with dark background
    visualization = np.zeros((height, width, 3), dtype=np.uint8)
    visualization[:, :] = [30, 30, 30]  # Dark gray background
    
    # Add a title
    cv2.putText(visualization, "3D View", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Define the coordinate system axes
    axes_length = 100
    origin = (width // 2, height // 2)
    
    # Draw coordinate axes - CORRECTED ORIENTATION
    # X-axis (red) - points right
    cv2.line(visualization, origin, (origin[0] + axes_length, origin[1]), (0, 0, 255), 2)
    cv2.putText(visualization, "X", (origin[0] + axes_length + 10, origin[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Y-axis (green) - points UP (not down as before)
    cv2.line(visualization, origin, (origin[0], origin[1] - axes_length), (0, 255, 0), 2)
    cv2.putText(visualization, "Y", (origin[0], origin[1] - axes_length - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Z-axis (blue) - points out of screen (towards viewer)
    cv2.line(visualization, origin, (origin[0] + axes_length//2, origin[1] + axes_length//2), (255, 0, 0), 2)
    cv2.putText(visualization, "Z", (origin[0] + axes_length//2 + 5, origin[1] + axes_length//2 + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # If nothing to display, show a message
    if (hand_results is None or not hasattr(hand_results, 'multi_hand_landmarks') or not hand_results.multi_hand_landmarks) and (detected_objects is None or len(detected_objects) == 0):
        cv2.putText(visualization, "No hands or objects detected", (width//2 - 120, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return visualization
    
    # Draw hand landmarks if available
    if hand_results is not None and hasattr(hand_results, 'multi_hand_landmarks') and hand_results.multi_hand_landmarks:
        # Create connections for hand landmarks (similar to MediaPipe connections)
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (0, 5), (5, 9), (9, 13), (13, 17)
        ]
        
        # For each hand
        for hand_landmarks in hand_results.multi_hand_landmarks:
            points_3d = []
            
            # Extract 3D coordinates and project them to the visualization
            for i, landmark in enumerate(hand_landmarks.landmark):
                # Get 3D coordinates (normalized)
                x, y, z = landmark.x, landmark.y, landmark.z
                
                # Get depth from depth frame if available
                if depth_frame is not None:
                    # Convert normalized coordinates to pixel coordinates
                    pixel_x = int(x * depth_frame.shape[1])
                    pixel_y = int(y * depth_frame.shape[0])
                    
                    # Check if within bounds
                    if 0 <= pixel_x < depth_frame.shape[1] and 0 <= pixel_y < depth_frame.shape[0]:
                        pixel_depth = depth_frame[pixel_y, pixel_x]
                        if pixel_depth > 0:  # Valid depth
                            # Scale z based on real depth (convert mm to normalized value)
                            z = pixel_depth / 5000.0  # Adjust scaling as needed
                
                # FIXED: Project 3D to 2D for visualization (simple orthographic projection)
                # Flipped y-axis orientation as requested
                proj_x = int(origin[0] + x * width * 0.4 - z * width * 0.1)
                proj_y = int(origin[1] + y * height * 0.4)  # Flipped y-axis (+ instead of -)
                
                # Store the projected point
                points_3d.append((proj_x, proj_y))
                
                # Draw the landmark point
                cv2.circle(visualization, (proj_x, proj_y), 5, (100, 100, 255), -1)
                
                # Add point labels for key points
                if i == 0 or i == 4 or i == 8 or i == 12 or i == 16 or i == 20:
                    cv2.putText(visualization, str(i), (proj_x + 5, proj_y + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw connections between landmarks
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(points_3d) and end_idx < len(points_3d):
                    cv2.line(visualization, points_3d[start_idx], points_3d[end_idx], (0, 255, 0), 2)
    
    # Draw detected objects if available
    if detected_objects is not None and len(detected_objects) > 0 and depth_frame is not None:
        # Define colors for different object types (to distinguish them visually)
        object_colors = {
            "person": (255, 200, 0),    # Yellow
            "chair": (0, 255, 255),     # Cyan
            "bottle": (255, 0, 255),    # Magenta
            "cup": (200, 100, 255),     # Purple
            "default": (0, 200, 200)    # Teal (for other objects)
        }
        
        # For each detected object
        for label, confidence, bbox in detected_objects:
            # Extract point cloud for the object
            point_cloud = extract_object_point_cloud(depth_frame, bbox)
            
            if point_cloud is not None:
                x_coords, y_coords, z_coords = point_cloud
                
                # Get color for this object type
                color = object_colors.get(label.lower(), object_colors["default"])
                
                # Project and draw each point in the point cloud
                for i in range(len(x_coords)):
                    # Normalize coordinates to 0-1 range
                    x_norm = x_coords[i] / depth_frame.shape[1]
                    y_norm = y_coords[i] / depth_frame.shape[0]
                    z_norm = z_coords[i] / 5.0  # Normalize depth to reasonable range
                    
                    # FIXED: Project with correct orientation and flipped y-axis
                    proj_x = int(origin[0] + x_norm * width * 0.4 - z_norm * width * 0.1)
                    proj_y = int(origin[1] + y_norm * height * 0.4)  # Flipped y-axis (+ instead of -)
                    
                    # Ensure point is within display bounds
                    if 0 <= proj_x < width and 0 <= proj_y < height:
                        # Draw the point
                        cv2.circle(visualization, (proj_x, proj_y), 1, color, -1)
                
                # Draw object label
                label_pos_x = int(origin[0] + (np.mean(x_coords) / depth_frame.shape[1]) * width * 0.4)
                label_pos_y = int(origin[1] + (np.mean(y_coords) / depth_frame.shape[0]) * height * 0.4)  # Flipped y-axis
                cv2.putText(visualization, label, (label_pos_x, label_pos_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Add legend
    legend_y = height - 120
    cv2.putText(visualization, "Legend:", (width - 150, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    legend_y += 20
    
    if hand_results is not None and hasattr(hand_results, 'multi_hand_landmarks') and hand_results.multi_hand_landmarks:
        cv2.putText(visualization, "Hand Joints", (width - 150, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        legend_y += 20
        cv2.putText(visualization, "Connections", (width - 150, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        legend_y += 20
    
    if detected_objects is not None and len(detected_objects) > 0:
        # Add object types to legend
        for label, _, _ in detected_objects:
            if legend_y < height - 10:  # Ensure we don't go off the bottom of the image
                color = object_colors.get(label.lower(), object_colors["default"])
                cv2.putText(visualization, label, (width - 150, legend_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                legend_y += 20
    
    return visualization

def detect_objects(frame, confidence_threshold=0.5, model_name="mobilenet-ssd"):
    """
    Detect objects in a frame using various object detection models.
    
    Args:
        frame: The input image frame
        confidence_threshold: Minimum confidence threshold for detections
        model_name: The model to use ("mobilenet-ssd", "yolo-tiny", or "mp-objects" for MediaPipe)
    
    Returns:
        List of (label, confidence, bounding_box) tuples
    """
    # Check if frame is valid
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return []
    
    # Ensure frame is in RGB format for processing
    if len(frame.shape) == 2:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Try MediaPipe object detection if selected and available
    if model_name == "mp-objects" and mp is not None:
        try:
            # Initialize MediaPipe object detection if not already done
            global mp_object_detector
            if 'mp_object_detector' not in globals():
                mp_objectron = mp.solutions.objectron
                mp_object_detector = mp_objectron.Objectron(
                    static_image_mode=False,
                    max_num_objects=5,
                    min_detection_confidence=confidence_threshold,
                    model_name="Shoe"  # Can also be "Cup", "Chair", "Camera"
                )
                print("MediaPipe Objectron initialized")
            
            # Process image
            results = mp_object_detector.process(frame_rgb)
            
            detected_objects = []
            if results.detected_objects:
                for i, detected_object in enumerate(results.detected_objects):
                    # Get bounding box
                    height, width = frame_rgb.shape[:2]
                    landmarks_2d = detected_object.landmarks_2d
                    
                    # Calculate bounding box from landmark points
                    x_coords = [landmark.x * width for landmark in landmarks_2d.landmark]
                    y_coords = [landmark.y * height for landmark in landmarks_2d.landmark]
                    startX, startY = int(min(x_coords)), int(min(y_coords))
                    endX, endY = int(max(x_coords)), int(max(y_coords))
                    
                    # Add to detected objects
                    detected_objects.append(("Object", detected_object.score, (startX, startY, endX, endY)))
            
            return detected_objects
        
        except Exception as e:
            print(f"Error using MediaPipe object detection: {e}")
            # Fall back to OpenCV models
    
    # For OpenCV DNN-based models
    if model_name in ["mobilenet-ssd", "yolo-tiny"]:
        # Initialize model if not already done
        global object_detection_model, object_detection_labels
        
        # Load new model if different from current or not loaded
        if ('current_model' not in globals() or globals()['current_model'] != model_name or 
            'object_detection_model' not in globals() or object_detection_model is None):
            
            try:
                model_path = os.path.join(os.path.dirname(__file__), "models")
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                
                if model_name == "mobilenet-ssd":
                    # MobileNet SSD model
                    model_weights = os.path.join(model_path, "mobilenet_ssd.caffemodel")
                    model_config = os.path.join(model_path, "mobilenet_ssd.prototxt")
                    
                    if not os.path.exists(model_weights) or not os.path.exists(model_config):
                        print("Downloading MobileNet SSD model files...")
                        weight_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
                        config_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt"
                        
                        # Simple download function
                        import urllib.request
                        def download_file(url, destination):
                            try:
                                urllib.request.urlretrieve(url, destination)
                                return True
                            except Exception as e:
                                print(f"Error downloading {url}: {e}")
                                return False
                        
                        # Download the files
                        if not download_file(weight_url, model_weights) or not download_file(config_url, model_config):
                            raise Exception("Failed to download model files")
                    
                    # Load the model
                    object_detection_model = cv2.dnn.readNetFromCaffe(model_config, model_weights)
                    
                    # List of class names
                    object_detection_labels = ["background", "aeroplane", "bicycle", "bird", "boat",
                                             "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                             "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                             "sofa", "train", "tvmonitor"]
                
                elif model_name == "yolo-tiny":
                    # YOLOv4 Tiny model
                    model_weights = os.path.join(model_path, "yolov4-tiny.weights")
                    model_config = os.path.join(model_path, "yolov4-tiny.cfg")
                    class_names = os.path.join(model_path, "coco.names")
                    
                    if not os.path.exists(model_weights) or not os.path.exists(model_config) or not os.path.exists(class_names):
                        print("Downloading YOLOv4 Tiny model files...")
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
                        
                        # Download the files
                        if (not download_file(weight_url, model_weights) or 
                            not download_file(config_url, model_config) or 
                            not download_file(names_url, class_names)):
                            raise Exception("Failed to download model files")
                    
                    # Load YOLO model
                    object_detection_model = cv2.dnn.readNetFromDarknet(model_config, model_weights)
                    
                    # Read class names
                    with open(class_names, 'rt') as f:
                        object_detection_labels = f.read().rstrip('\n').split('\n')
                
                # For GPU acceleration if available
                try:
                    # Disable GPU acceleration for now due to compatibility issues
                    # object_detection_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    # object_detection_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    # print(f"Using GPU acceleration for {model_name}")
                    print(f"Using CPU for {model_name} (GPU acceleration disabled)")
                except:
                    print(f"GPU acceleration not available, using CPU for {model_name}")
                
                # Store current model name
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
            # MobileNet SSD processing
            blob = cv2.dnn.blobFromImage(frame_rgb, 0.007843, (300, 300), 127.5)
            object_detection_model.setInput(blob)
            detections = object_detection_model.forward()
            
            # Process results
            detected_objects = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    # Get the class label
                    class_id = int(detections[0, 0, i, 1])
                    if class_id < len(object_detection_labels):
                        label = object_detection_labels[class_id]
                        
                        # Get bounding box coordinates
                        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        # Add to detected objects
                        detected_objects.append((label, confidence, (startX, startY, endX, endY)))
            
        elif model_name == "yolo-tiny":
            # YOLO processing
            blob = cv2.dnn.blobFromImage(frame_rgb, 1/255.0, (416, 416), swapRB=True, crop=False)
            object_detection_model.setInput(blob)
            
            # Get output layer names
            layer_names = object_detection_model.getLayerNames()
            output_layers = [layer_names[i - 1] for i in object_detection_model.getUnconnectedOutLayers()]
            
            # Run forward pass
            outputs = object_detection_model.forward(output_layers)
            
            # Process results
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > confidence_threshold:
                        # YOLO returns center, width, height
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
            
            detected_objects = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    label = object_detection_labels[class_ids[i]]
                    confidence = confidences[i]
                    
                    # Convert to the expected format (label, confidence, (startX, startY, endX, endY))
                    detected_objects.append((label, confidence, (x, y, x + w, y + h)))
        
        return detected_objects
    
    # Fallback to no detections if invalid model name
    print(f"Unknown model: {model_name}. Use 'mobilenet-ssd', 'yolo-tiny', or 'mp-objects'")
    return []

def draw_detected_objects(image, detected_objects, depth_frame=None):
    """Draw detected objects on an image with depth information if available"""
    for label, confidence, bbox in detected_objects:
        startX, startY, endX, endY = bbox
        
        # Calculate average depth for this object if depth frame is available
        depth_text = ""
        if depth_frame is not None:
            # Ensure coordinates are within frame boundaries
            height, width = depth_frame.shape
            valid_startX = max(0, startX)
            valid_startY = max(0, startY)
            valid_endX = min(width, endX)
            valid_endY = min(height, endY)
            
            # Get depth values in the bounding box
            roi_depth = depth_frame[valid_startY:valid_endY, valid_startX:valid_endX]
            valid_depths = roi_depth[roi_depth > 0]
            
            if len(valid_depths) > 0:
                # Convert from mm to meters
                avg_depth = np.mean(valid_depths) / 1000.0
                depth_text = f" @ {avg_depth:.2f}m"
        
        # Draw bounding box with color based on depth if available
        color = (0, 255, 0)  # Default green
        if depth_text:
            # Vary color based on depth (red for close, green for mid, blue for far)
            depth_value = float(depth_text.split('@')[1].split('m')[0])
            if depth_value < 1.0:
                color = (0, 0, 255)  # Red (BGR)
            elif depth_value < 2.0:
                color = (0, 255, 0)  # Green
            else:
                color = (255, 0, 0)  # Blue
        
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        
        # Draw label, confidence, and depth
        y = startY - 15 if startY - 15 > 15 else startY + 15
        text = f"{label}: {confidence:.2f}{depth_text}"
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def extract_object_point_cloud(depth_frame, bbox, max_points=1000):
    """Extract point cloud for an object based on its bounding box"""
    if depth_frame is None:
        return None
    
    startX, startY, endX, endY = bbox
    
    # Ensure bounding box is within frame boundaries
    height, width = depth_frame.shape
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(width, endX)
    endY = min(height, endY)
    
    # Extract depth values within the bounding box
    roi_depth = depth_frame[startY:endY, startX:endX]
    
    # Filter out invalid depth values
    valid_mask = roi_depth > 0
    if not np.any(valid_mask):
        return None
    
    # Get valid depths
    valid_depths = roi_depth[valid_mask]
    
    # Create x, y coordinates grid for valid points
    y_indices, x_indices = np.where(valid_mask)
    
    # Adjust coordinates to image space
    x_coords = x_indices + startX
    y_coords = y_indices + startY
    
    # Convert depth from mm to meters
    z_coords = valid_depths / 1000.0
    
    # Print depth statistics for debugging
    avg_depth = np.mean(z_coords)
    min_depth = np.min(z_coords)
    max_depth = np.max(z_coords)
    print(f"Object depth range: {min_depth:.2f}m - {max_depth:.2f}m, avg: {avg_depth:.2f}m")
    
    # If there are too many points, sample them
    if len(x_coords) > max_points:
        indices = np.random.choice(len(x_coords), max_points, replace=False)
        x_coords = x_coords[indices]
        y_coords = y_coords[indices]
        z_coords = z_coords[indices]
    
    return x_coords, y_coords, z_coords

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RealSense Depth Camera Example')
    parser.add_argument('--min-depth', type=float, default=0.1, help='Minimum depth in meters')
    parser.add_argument('--max-depth', type=float, default=5.0, help='Maximum depth in meters')
    parser.add_argument('--enable-color', action='store_true', help='Enable color stream')
    parser.add_argument('--colormap', type=str, default='RAINBOW', choices=['TURBO', 'JET', 'PLASMA', 'VIRIDIS', 'HOT', 'RAINBOW'], 
                      help='Colormap to use for depth visualization')
    parser.add_argument('--no-auto-range', action='store_true', help='Disable automatic depth range adjustment')
    parser.add_argument('--no-auto-exposure', action='store_true', help='Disable auto exposure for IR and color cameras')
    # Add resolution and frame rate parameters
    parser.add_argument('--resolution', type=str, default='640x360', 
                        choices=['1280x720', '848x480', '640x360', '480x270', '424x240'],
                        help='Camera resolution (width x height)')
    parser.add_argument('--fps', type=int, default=30, 
                        choices=[5, 15, 30, 60, 90],
                        help='Camera frame rate (fps)')
    parser.add_argument('--no-hand-tracking', action='store_true', help='Disable hand tracking and gesture detection')
    parser.add_argument('--advanced-depth', action='store_true', help='Use advanced depth visualization')
    parser.add_argument('--no-edge-enhancement', action='store_true', help='Disable edge enhancement in advanced depth mode')
    parser.add_argument('--no-highlight', action='store_true', help='Disable closest point highlighting in advanced depth mode')
    parser.add_argument('--no-3d-viz', action='store_true', help='Disable 3D visualization of hand tracking')
    parser.add_argument('--enable-object-detection', action='store_true', help='Enable object detection and point cloud visualization')
    parser.add_argument('--object-confidence', type=float, default=0.5, help='Confidence threshold for object detection')
    parser.add_argument('--object-model', type=str, default='mobilenet-ssd', 
                      choices=['mobilenet-ssd', 'yolo-tiny', 'mp-objects'],
                      help='Object detection model to use')
    parser.add_argument('--no-hands-in-3d', action='store_true', help='Disable hands in 3D visualization (objects only)')
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Initialize MediaPipe Hands if hand tracking is enabled
    hands = None
    if not args.no_hand_tracking and mp is not None:
        print("Initializing hand tracking...")
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    else:
        print("Hand tracking disabled")
        args.no_hand_tracking = True
    
    # Create RealSense instance with depth filtering
    try:
        rs = PyRealSense(
            width=width,
            height=height,
            framerate=args.fps,
            enable_color=args.enable_color,
            enable_ir=True,
            min_depth=args.min_depth,
            max_depth=args.max_depth
        )
    except TypeError as e:
        print(f"Error initializing RealSense: {e}")
        print("Trying without auto_exposure parameter...")
        rs = PyRealSense(
            width=width,
            height=height,
            framerate=args.fps,
            enable_color=args.enable_color,
            enable_ir=True,
            min_depth=args.min_depth,
            max_depth=args.max_depth
        )
    
    # Set colormap based on argument
    COLORMAP_CHOICES = {
        'TURBO': cv2.COLORMAP_TURBO,
        'JET': cv2.COLORMAP_JET,
        'PLASMA': cv2.COLORMAP_PLASMA,
        'VIRIDIS': cv2.COLORMAP_VIRIDIS,
        'HOT': cv2.COLORMAP_HOT,
        'RAINBOW': cv2.COLORMAP_RAINBOW
    }
    selected_colormap = COLORMAP_CHOICES.get(args.colormap.upper(), cv2.COLORMAP_RAINBOW)
    
    try:
        # Redirect stdout temporarily to capture debug messages from realsense
        original_stdout = sys.stdout
        sys.stdout = LoggerWriter(realsense_logger.debug)
        
        # Start the camera
        rs.start()
        
        # Restore stdout
        sys.stdout = original_stdout
        
        # Only print essential camera info
        logger.info(f"Camera ready - {rs.camera_model_name}, {rs.frame_width}x{rs.frame_height} @ {rs.frame_rate}fps")
        logger.info(f"Auto-range: {'Disabled' if args.no_auto_range else 'Enabled'}, Colormap: {args.colormap}")
        logger.info(f"Hand Tracking: {'Disabled' if args.no_hand_tracking else 'Enabled'}")
        logger.info(f"Depth Visualization: {'Advanced' if args.advanced_depth else 'Standard'}")
        
        print("\nControls:")
        print("1. Press 'q' to quit")
        
        current_description = "Initializing..."
        
        # Variables for auto-range
        depth_min_running = args.min_depth
        depth_max_running = args.max_depth
        auto_range = not args.no_auto_range
        
        while True:
            # Redirect stdout during frame capture to suppress debug messages
            original_stdout = sys.stdout
            sys.stdout = LoggerWriter(realsense_logger.debug)
            
            frames = rs.get_frames()
            
            # Restore stdout
            sys.stdout = original_stdout
            
            if frames is None:
                continue
            
            depth_frame = frames.get('depth')
            ir_frame = frames.get('infrared')
            color_frame = frames.get('color')
            
            # Update global depth frame for 3D visualization
            global latest_depth_frame
            latest_depth_frame = depth_frame
            
            # Safety check for missing frames
            if depth_frame is None or ir_frame is None:
                continue
                
            # Safety check for corrupted data
            if not isinstance(depth_frame, np.ndarray) or not isinstance(ir_frame, np.ndarray):
                continue
                
            # Safety check for empty arrays
            if depth_frame.size == 0 or ir_frame.size == 0:
                continue
                
            # Safety check for NaN values
            if np.isnan(depth_frame).any() or np.isnan(ir_frame).any():
                # Replace NaN with zeros
                depth_frame = np.nan_to_num(depth_frame)
                ir_frame = np.nan_to_num(ir_frame)
            
            # Auto-adjust depth range if enabled
            if auto_range:
                valid_depths = depth_frame[depth_frame > 0]
                if len(valid_depths) > 0:
                    # Convert depth values from millimeters to meters for processing
                    valid_depths_m = valid_depths / 1000.0
                    
                    # Use tighter percentiles for better focus on main objects
                    min_depth_scene = np.percentile(valid_depths_m, 2)  # 2nd percentile to better capture close objects
                    max_depth_scene = np.percentile(valid_depths_m, 98) # 98th percentile to better capture far objects
                    
                    # Calculate range in meters
                    scene_range = max_depth_scene - min_depth_scene
                    
                    # Different adaptation rates for min and max
                    if min_depth_scene < depth_min_running:
                        # Adapt faster to closer objects
                        depth_min_running = 0.7 * depth_min_running + 0.3 * min_depth_scene
                    else:
                        # Slower adjustment when objects move away
                        depth_min_running = 0.9 * depth_min_running + 0.1 * min_depth_scene
                        
                    if max_depth_scene > depth_max_running:
                        # Adapt faster to farther objects
                        depth_max_running = 0.7 * depth_max_running + 0.3 * max_depth_scene
                    else:
                        # Slower adjustment when max range decreases
                        depth_max_running = 0.9 * depth_max_running + 0.1 * max_depth_scene
                    
                    # Ensure minimum range based on scene characteristics
                    min_range = max(0.5, scene_range * 0.2)  # At least 20% of detected range or 0.5m
                    if depth_max_running - depth_min_running < min_range:
                        # Extend in both directions
                        mid_point = (depth_max_running + depth_min_running) / 2
                        depth_min_running = mid_point - min_range / 2
                        depth_max_running = mid_point + min_range / 2
            else:
                depth_min_running = args.min_depth
                depth_max_running = args.max_depth
            
            # Get depth information
            current_description = get_depth_info(depth_frame)
            
            # Create depth visualization with current range - use advanced mode if selected
            if args.advanced_depth:
                depth_colormap = create_advanced_depth_colormap(
                    depth_frame, 
                    depth_min_running, 
                    depth_max_running, 
                    selected_colormap,
                    enhance_edges=not args.no_edge_enhancement,
                    highlight_closest=not args.no_highlight
                )
            else:
                depth_colormap = create_enhanced_depth_colormap(
                    depth_frame, 
                    depth_min_running, 
                    depth_max_running, 
                    selected_colormap
                )
            
            try:
                # Initialize visualization images
                ir_viz = ir_frame.copy()
                if len(ir_viz.shape) == 2:  # Convert grayscale to BGR for display
                    ir_viz = cv2.cvtColor(ir_viz, cv2.COLOR_GRAY2BGR)
                
                depth_viz = depth_colormap.copy()
                
                # Process hand gestures and draw landmarks if hand tracking is enabled
                if not args.no_hand_tracking and hands is not None:
                    # Process hand gestures
                    hand_results = detect_gesture(ir_frame, hands)
                    
                    # Update global hand landmarks for 3D visualization
                    global latest_hand_landmarks
                    latest_hand_landmarks = hand_results
                    
                    # Draw hand landmarks on visualization images
                    try:
                        ir_viz = draw_hand_landmarks(ir_viz, hand_results)
                    except Exception as e:
                        print(f"Error drawing IR landmarks: {e}")
                    
                    try:
                        depth_viz = draw_hand_landmarks(depth_viz, hand_results)
                    except Exception as e:
                        print(f"Error drawing depth landmarks: {e}")
                
                # Safety check for visualization images
                if ir_viz is None or depth_viz is None or not isinstance(ir_viz, np.ndarray) or not isinstance(depth_viz, np.ndarray):
                    print("Warning: Invalid visualization images")
                    continue
                    
                if ir_viz.shape[0] != depth_viz.shape[0] or ir_viz.shape[1] != depth_viz.shape[1]:
                    # Resize to match if dimensions don't match
                    ir_viz = cv2.resize(ir_viz, (depth_viz.shape[1], depth_viz.shape[0]))
                
                # Run object detection if enabled and we have a color or IR frame
                if args.enable_object_detection:
                    # Prefer color frame if available, otherwise use IR
                    detection_frame = color_frame if color_frame is not None and color_frame.size > 0 else ir_viz
                    
                    if detection_frame is not None and detection_frame.size > 0:
                        # Detect objects using the specified model
                        latest_detected_objects = detect_objects(
                            detection_frame, 
                            args.object_confidence,
                            args.object_model
                        )
                        print(f"Detected {len(latest_detected_objects)} objects using {args.object_model}")
                        
                        # Draw detected objects on IR visualization with depth info
                        if latest_detected_objects and len(latest_detected_objects) > 0:
                            ir_viz = draw_detected_objects(ir_viz, latest_detected_objects, depth_frame)
                            
                            # Print detailed object information
                            for i, (label, confidence, bbox) in enumerate(latest_detected_objects):
                                print(f"Object {i+1}: {label} ({confidence:.2f}), bbox: {bbox}")
                
                # Create 3D visualization with both hands and objects (but optionally disable hands)
                viz_3d = create_3d_visualization(
                    hand_results=None if args.no_hands_in_3d else latest_hand_landmarks, 
                    depth_frame=depth_frame,
                    detected_objects=latest_detected_objects if args.enable_object_detection else None,
                    width=depth_viz.shape[1], 
                    height=depth_viz.shape[0]
                )
                
                # Create display image by stacking three visualizations horizontally
                try:
                    display_image = np.hstack((depth_viz, ir_viz, viz_3d))
                except Exception as e:
                    print(f"Error creating display image: {e}")
                    # Create a fallback display image with error message
                    display_image = np.zeros((480, 1920, 3), dtype=np.uint8)
                    cv2.putText(display_image, f"Error: {str(e)}", (400, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error in main loop: {e}")
                # Create a blank display image as fallback
                display_image = np.zeros((480, 1920, 3), dtype=np.uint8)
                cv2.putText(display_image, f"Error: {str(e)}", (400, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add depth information as subtitle
            try:
                display_image = draw_subtitle(display_image, current_description)
            except Exception as e:
                print(f"Error drawing subtitle: {e}")
                # Try to continue without adding subtitle
            
            # Display the image without any status text
            cv2.imshow('RealSense', display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    finally:
        # Stop the camera
        rs.stop()
        cv2.destroyAllWindows()
        
        # Clean up resources
        if hands is not None:
            hands.close()

if __name__ == "__main__":
    main() 