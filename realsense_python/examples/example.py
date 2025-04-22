import cv2
import numpy as np
import argparse
import os
import time
from datetime import datetime
from realsense.wrapper import PyRealSense
import mediapipe as mp
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
from typing import Optional
import logging
import sys

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

# Initialize BLIP model only if needed
print("Checking for LLM model requirement...")
processor = None
model = None
device = None

def initialize_blip_model():
    """Initialize BLIP model for scene description (only when needed)"""
    global processor, model, device
    if processor is None or model is None:
        print("Loading BLIP model...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)
        print(f"Using device: {device}")

def draw_hand_landmarks(image, results, color=(0, 255, 0)):
    """Draw hand landmarks and connections on the image."""
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
    # Convert IR frame to RGB for MediaPipe
    ir_rgb = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2RGB)
    
    # Process frame
    results = hands.process(ir_rgb)
    
    # Debug print for hand detection
    if results.multi_hand_landmarks:
        logger.debug(f"Detected {len(results.multi_hand_landmarks)} hands")
        
        # Check for thumbs up gesture
        if len(results.multi_hand_landmarks) == 2:
            thumbs_up_count = 0
            thumbs_down_count = 0
            open_palm_count = 0
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Get thumb tip and base
                thumb_tip = hand_landmarks.landmark[4]
                thumb_base = hand_landmarks.landmark[2]
                
                # Get middle finger pip (middle joint) for reference
                middle_pip = hand_landmarks.landmark[10]
                
                # Get all finger tips
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                ring_tip = hand_landmarks.landmark[16]
                pinky_tip = hand_landmarks.landmark[20]
                
                # Check if thumb is pointing up
                if thumb_tip.y < thumb_base.y and thumb_tip.y < middle_pip.y:
                    thumbs_up_count += 1
                # Check if thumb is pointing down
                elif thumb_tip.y > thumb_base.y and thumb_tip.y > middle_pip.y:
                    thumbs_down_count += 1
                
                # Check for open palm (all fingers extended)
                if (index_tip.y < middle_pip.y and 
                    middle_tip.y < middle_pip.y and 
                    ring_tip.y < middle_pip.y and 
                    pinky_tip.y < middle_pip.y):
                    open_palm_count += 1
            
            if thumbs_up_count == 2:
                return "thumbs_up", results
            elif thumbs_down_count == 2:
                return "thumbs_down", results
            elif open_palm_count == 2:
                return "open_palm", results
    
    return None, results

def save_frame_data(output_dir, frame_idx, depth_frame, ir_frame, metadata):
    """Save frame data and metadata to disk."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save depth and IR frames
    np.save(os.path.join(output_dir, f'depth_{frame_idx:06d}.npy'), depth_frame)
    np.save(os.path.join(output_dir, f'ir_{frame_idx:06d}.npy'), ir_frame)
    
    # Save metadata
    with open(os.path.join(output_dir, f'metadata_{frame_idx:06d}.txt'), 'w') as f:
        for key, value in metadata.items():
            f.write(f'{key}: {value}\n')

def get_scene_description(depth_frame: np.ndarray, ir_frame: np.ndarray, use_llm: bool = True) -> Optional[str]:
    """Generate a description of the scene using BLIP if LLM is enabled, otherwise just depth info."""
    try:
        if ir_frame is None:
            return "No IR feed available"
        
        # Always provide depth information
        valid_depth = depth_frame[depth_frame > 0] if depth_frame is not None else np.array([])
        if len(valid_depth) > 0:
            # Convert to meters
            avg_depth = np.mean(valid_depth) / 1000.0
            min_depth = np.min(valid_depth) / 1000.0
            max_depth = np.max(valid_depth) / 1000.0
            
            depth_info = f"Objects at {min_depth:.2f}-{max_depth:.2f}m, average {avg_depth:.2f}m"
        else:
            depth_info = "No valid depth data"
        
        # Return only depth info if LLM is disabled
        if not use_llm:
            return depth_info
        
        # Initialize LLM if needed
        initialize_blip_model()
        
        # Convert IR to 3 channels for BLIP
        ir_3ch = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(ir_3ch, cv2.COLOR_BGR2RGB))
        
        # Prepare inputs
        inputs = processor(pil_image, return_tensors="pt").to(device)
        
        # Generate description
        out = model.generate(**inputs, max_new_tokens=50)
        description = processor.decode(out[0], skip_special_tokens=True)
        
        # Add depth information
        description += f" ({depth_info})"
        
        # Print the description to terminal
        print(f"LLM: {description}")
        
        return description
    except Exception as e:
        print(f"Error in scene description: {e}")
        return "Scene description error"

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
    """Basic, reliable depth colormap with minimal processing.
    
    Args:
        depth_frame: Depth frame in millimeters
        min_depth: Minimum depth in meters
        max_depth: Maximum depth in meters
        colormap_type: OpenCV colormap to use
        
    Returns:
        Colorized depth map
    """
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
    
    return colormap

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RealSense Depth Camera Example')
    parser.add_argument('--min-depth', type=float, default=0.1, help='Minimum depth in meters')
    parser.add_argument('--max-depth', type=float, default=5.0, help='Maximum depth in meters')
    parser.add_argument('--enable-color', action='store_true', help='Enable color stream')
    parser.add_argument('--output-dir', type=str, default='recordings', help='Output directory for recordings')
    parser.add_argument('--colormap', type=str, default='RAINBOW', choices=['TURBO', 'JET', 'PLASMA', 'VIRIDIS', 'HOT', 'RAINBOW'], 
                      help='Colormap to use for depth visualization')
    parser.add_argument('--no-auto-range', action='store_true', help='Disable automatic depth range adjustment')
    parser.add_argument('--no-auto-exposure', action='store_true', help='Disable auto exposure for IR and color cameras')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM scene descriptions')
    # Add resolution and frame rate parameters
    parser.add_argument('--resolution', type=str, default='640x360', 
                        choices=['1280x720', '848x480', '640x360', '480x270', '424x240'],
                        help='Camera resolution (width x height)')
    parser.add_argument('--fps', type=int, default=30, 
                        choices=[5, 15, 30, 60, 90],
                        help='Camera frame rate (fps)')
    args = parser.parse_args()
    
    # Print LLM status
    if args.no_llm:
        print("LLM disabled - using simple depth information only")
    else:
        # Only initialize LLM if needed
        initialize_blip_model()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Create RealSense instance with depth filtering
    rs = PyRealSense(
        width=width,
        height=height,
        framerate=args.fps,
        enable_color=args.enable_color,
        enable_ir=True,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        auto_exposure=not args.no_auto_exposure
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
        logger.info(f"LLM Descriptions: {'Disabled' if args.no_llm else 'Enabled'}")
        
        print("\nGesture Controls:")
        print("1. Two thumbs up 👍👍 to start recording")
        print("2. Two thumbs down 👎👎 to stop recording (>5 seconds to save)")
        print("3. Two open palms 🖐️🖐️ to cancel recording")
        print("4. Press 'q' to quit")
        if not args.no_llm:
            print("\n--- Scene Descriptions ---")
        else:
            print("\n--- Depth Information ---")
        
        recording = False
        output_dir = None
        frame_idx = 0
        start_time = None
        last_description_time = 0
        current_description = "Initializing scene description..."
        
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
            
            # Auto-adjust depth range if enabled (quietly)
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
            
            # Generate new scene description every 3 seconds
            current_time = time.time()
            if not args.no_llm and current_time - last_description_time >= 3.0:
                # Redirect stdout during scene description to capture only explicit prints
                sys.stdout = original_stdout
                current_description = get_scene_description(depth_frame, ir_frame) or "Unable to generate scene description"
                last_description_time = current_time
            
            # Create simplified depth visualization with current range
            depth_colormap = create_enhanced_depth_colormap(
                depth_frame, 
                depth_min_running, 
                depth_max_running, 
                selected_colormap
            )
            
            try:
                # Process hand gestures and draw landmarks
                gesture, results = detect_gesture(ir_frame, hands)
                
                # Create visualization images (with error handling)
                try:
                    ir_viz = draw_hand_landmarks(ir_frame.copy(), results)
                except Exception as e:
                    print(f"Error drawing IR landmarks: {e}")
                    ir_viz = ir_frame.copy()
                    if len(ir_viz.shape) == 2:  # Convert grayscale to BGR for display
                        ir_viz = cv2.cvtColor(ir_viz, cv2.COLOR_GRAY2BGR)
                
                try:
                    depth_viz = draw_hand_landmarks(depth_colormap.copy(), results)
                except Exception as e:
                    print(f"Error drawing depth landmarks: {e}")
                    depth_viz = depth_colormap.copy()
                
                # Handle gestures with minimal printing
                if gesture == "thumbs_up" and not recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = os.path.join(args.output_dir, timestamp)
                    print(f"\n▶️ Recording started")
                    recording = True
                    frame_idx = 0
                    start_time = time.time()
                elif gesture == "thumbs_down" and recording:
                    recording = False
                    duration = time.time() - start_time
                    if duration >= 5.0:
                        print(f"\n⏹️ Recording saved ({duration:.1f}s, {frame_idx} frames)")
                    else:
                        print(f"\n❌ Recording discarded (too short: {duration:.1f}s)")
                        if output_dir and os.path.exists(output_dir):
                            for file in os.listdir(output_dir):
                                os.remove(os.path.join(output_dir, file))
                            os.rmdir(output_dir)
                        output_dir = None
                elif gesture == "open_palm" and recording:
                    recording = False
                    print(f"\n🚫 Recording cancelled")
                    if output_dir and os.path.exists(output_dir):
                        for file in os.listdir(output_dir):
                            os.remove(os.path.join(output_dir, file))
                        os.rmdir(output_dir)
                    output_dir = None
                
                # Safety check for visualization images
                if ir_viz is None or depth_viz is None or not isinstance(ir_viz, np.ndarray) or not isinstance(depth_viz, np.ndarray):
                    print("Warning: Invalid visualization images")
                    continue
                    
                if ir_viz.shape[0] != depth_viz.shape[0] or ir_viz.shape[1] != depth_viz.shape[1]:
                    # Resize to match if dimensions don't match
                    ir_viz = cv2.resize(ir_viz, (depth_viz.shape[1], depth_viz.shape[0]))
                
                # Create display image by stacking horizontally
                try:
                    display_image = np.hstack((depth_viz, ir_viz))
                except Exception as e:
                    print(f"Error creating display image: {e}")
                    # Create a fallback image
                    display_image = np.zeros((480, 1280, 3), dtype=np.uint8)
                    cv2.putText(display_image, "Display Error", (500, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Add scene description as subtitle
                try:
                    display_image = draw_subtitle(display_image, current_description)
                except Exception as e:
                    print(f"Error drawing subtitle: {e}")
                
                # Save frame data if recording
                if recording:
                    try:
                        metadata = {
                            'timestamp': time.time() - start_time,
                            'frame_index': frame_idx,
                            'min_depth': depth_min_running,
                            'max_depth': depth_max_running,
                            'frame_rate': rs.frame_rate,
                            'resolution': f"{rs.frame_width}x{rs.frame_height}",
                            'scene_description': current_description
                        }
                        save_frame_data(output_dir, frame_idx, depth_frame, ir_frame, metadata)
                        frame_idx += 1
                        print(f"Recording frame {frame_idx}", end='\r')
                    except Exception as e:
                        print(f"Error saving frame data: {e}")
                
                # Show recording status
                status_text = "Recording..." if recording else "Ready"
                status_color = (0, 0, 255) if recording else (0, 255, 0)
                
                if recording:
                    cv2.rectangle(display_image, (0, 0), (display_image.shape[1]-1, display_image.shape[0]-1), status_color, 10)
                
                text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (display_image.shape[1] - text_size[0]) // 2
                cv2.rectangle(display_image, (text_x-10, 10), (text_x+text_size[0]+10, 50), (0, 0, 0), -1)
                cv2.putText(display_image, status_text, (text_x, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                if recording:
                    frame_text = f"Frame: {frame_idx}"
                    frame_text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    frame_text_x = (display_image.shape[1] - frame_text_size[0]) // 2
                    cv2.rectangle(display_image, (frame_text_x-10, 60), (frame_text_x+frame_text_size[0]+10, 90), (0, 0, 0), -1)
                    cv2.putText(display_image, frame_text, (frame_text_x, 85), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                cv2.imshow('RealSense', display_image)
                
            except Exception as e:
                print(f"Error in main loop: {e}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        rs.stop()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main() 