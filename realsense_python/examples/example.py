import cv2
import numpy as np
import argparse
import os
import time
from datetime import datetime
from realsense.wrapper import PyRealSense
import mediapipe as mp

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
        print(f"Detected {len(results.multi_hand_landmarks)} hands")
        
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RealSense Depth Camera Example')
    parser.add_argument('--min-depth', type=float, default=0.0, help='Minimum depth in meters')
    parser.add_argument('--max-depth', type=float, default=10.0, help='Maximum depth in meters')
    parser.add_argument('--enable-color', action='store_true', help='Enable color stream')
    parser.add_argument('--output-dir', type=str, default='recordings', help='Output directory for recordings')
    args = parser.parse_args()
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,  # Lower threshold for better detection
        min_tracking_confidence=0.5
    )
    
    # Create RealSense instance with depth filtering
    rs = PyRealSense(
        enable_color=False,  # Disable color stream
        enable_ir=True,      # Enable IR stream for gesture detection
        min_depth=args.min_depth,
        max_depth=args.max_depth
    )
    
    try:
        # Start the camera
        rs.start()
        
        # Print camera information
        print(f"Camera Model: {rs.camera_model_name}")
        print(f"USB Mode: {'3.1' if rs.is_usb3_mode else '2.0'}")
        print(f"Frame Rate: {rs.frame_rate}")
        print(f"Resolution: {rs.frame_width}x{rs.frame_height}")
        print(f"Depth Range: {args.min_depth:.2f}m to {args.max_depth:.2f}m")
        
        print("\nGesture Controls:")
        print("1. Two thumbs up to start recording")
        print("2. Two thumbs down to stop recording")
        print("3. Two open palms to cancel recording")
        
        recording = False
        output_dir = None
        frame_idx = 0
        start_time = None
        
        while True:
            # Get frames
            frames = rs.get_frames()
            
            # Get depth frame (always available)
            depth_frame = frames['depth']
            
            # Convert depth to color for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Process IR frame for gestures
            if 'infrared' in frames:
                ir_frame = frames['infrared']
                
                # Detect gestures and get hand landmarks
                gesture, results = detect_gesture(ir_frame, hands)
                
                # Draw landmarks on IR frame
                ir_viz = draw_hand_landmarks(ir_frame.copy(), results)
                
                # Draw landmarks on depth frame
                depth_viz = draw_hand_landmarks(depth_colormap.copy(), results)
                
                # Handle gestures
                if gesture == "thumbs_up" and not recording:
                    # Start recording
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = os.path.join(args.output_dir, timestamp)
                    print(f"\nStarted recording to: {output_dir}")
                    recording = True
                    frame_idx = 0
                    start_time = time.time()
                elif gesture == "thumbs_down" and recording:
                    # Stop recording
                    recording = False
                    duration = time.time() - start_time
                    if duration >= 5.0:
                        print(f"\nStopped recording. Saved {frame_idx} frames to {output_dir}")
                    else:
                        print(f"\nRecording too short ({duration:.1f}s). Deleting {frame_idx} frames...")
                        if output_dir and os.path.exists(output_dir):
                            for file in os.listdir(output_dir):
                                os.remove(os.path.join(output_dir, file))
                            os.rmdir(output_dir)
                        output_dir = None
                elif gesture == "open_palm" and recording:
                    # Cancel recording
                    recording = False
                    print(f"\nCancelled recording. Deleting {frame_idx} frames...")
                    if output_dir and os.path.exists(output_dir):
                        for file in os.listdir(output_dir):
                            os.remove(os.path.join(output_dir, file))
                        os.rmdir(output_dir)
                    output_dir = None
                
                # Create display image
                display_image = np.hstack((depth_viz, ir_viz))
                
                # Save frame data if recording
                if recording:
                    metadata = {
                        'timestamp': time.time() - start_time,
                        'frame_index': frame_idx,
                        'min_depth': args.min_depth,
                        'max_depth': args.max_depth,
                        'frame_rate': rs.frame_rate,
                        'resolution': f"{rs.frame_width}x{rs.frame_height}"
                    }
                    save_frame_data(output_dir, frame_idx, depth_frame, ir_frame, metadata)
                    frame_idx += 1
                    print(f"Recording frame {frame_idx}", end='\r')
            
            # Show recording status
            status_text = "Recording..." if recording else "Ready"
            status_color = (0, 0, 255) if recording else (0, 255, 0)  # Red for recording, Green for ready
            
            # Add colored border when recording
            if recording:
                cv2.rectangle(display_image, (0, 0), (display_image.shape[1]-1, display_image.shape[0]-1), status_color, 10)
            
            # Add status text with background for better visibility
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (display_image.shape[1] - text_size[0]) // 2
            cv2.rectangle(display_image, (text_x-10, 10), (text_x+text_size[0]+10, 50), (0, 0, 0), -1)
            cv2.putText(display_image, status_text, (text_x, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Show frame count when recording
            if recording:
                frame_text = f"Frame: {frame_idx}"
                frame_text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                frame_text_x = (display_image.shape[1] - frame_text_size[0]) // 2
                cv2.rectangle(display_image, (frame_text_x-10, 60), (frame_text_x+frame_text_size[0]+10, 90), (0, 0, 0), -1)
                cv2.putText(display_image, frame_text, (frame_text_x, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Show images
            cv2.imshow('RealSense', display_image)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Stop the camera
        rs.stop()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main() 