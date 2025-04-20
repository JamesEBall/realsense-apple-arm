import cv2
import numpy as np
import argparse
import time
from realsense.wrapper import PyRealSense

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RealSense Depth Camera Simple Example')
    parser.add_argument('--colormap', type=str, default='JET', 
                      choices=['JET', 'HOT', 'RAINBOW', 'OCEAN', 'VIRIDIS', 'PLASMA', 'INFERNO', 'MAGMA', 'BONE', 'AUTUMN'], 
                      help='Colormap for depth visualization')
    args = parser.parse_args()
    
    # Create RealSense instance
    rs = PyRealSense()
    
    try:
        # Start the camera
        rs.start()
        
        print(f"Camera started successfully")
        print(f"Frame Resolution: {rs.frame_width}x{rs.frame_height}")
        print(f"Press 'q' to quit")
        
        while True:
            # Get frames
            frames = rs.get_frames()
            if frames is None:
                print("No frames received, retrying...")
                time.sleep(0.1)
                continue
            
            # Extract depth and IR frames
            depth_frame = frames.get('depth')
            ir_frame = frames.get('infrared')
            
            if depth_frame is None:
                print("No depth frame received, retrying...")
                time.sleep(0.1)
                continue
            
            # Process depth frame for visualization
            # Normalize depth for better visualization
            depth_norm = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply colormap
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_norm), 
                getattr(cv2, f'COLORMAP_{args.colormap}')
            )
            
            # Display depth statistics
            valid_mask = depth_frame > 0
            if np.any(valid_mask):
                min_val = depth_frame[valid_mask].min() / 1000.0  # Convert to meters
                max_val = depth_frame[valid_mask].max() / 1000.0
                avg_val = depth_frame[valid_mask].mean() / 1000.0
                depth_text = f"Min: {min_val:.2f}m  Max: {max_val:.2f}m  Avg: {avg_val:.2f}m"
                cv2.putText(depth_colormap, depth_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display IR frame if available
            if ir_frame is not None:
                # Resize IR frame to match depth frame if needed
                if ir_frame.shape[0] != depth_frame.shape[0] or ir_frame.shape[1] != depth_frame.shape[1]:
                    ir_frame = cv2.resize(ir_frame, (depth_frame.shape[1], depth_frame.shape[0]))
                
                # Convert IR to BGR for display
                ir_colorized = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
                
                # Stack depth and IR images side by side
                display_image = np.hstack((depth_colormap, ir_colorized))
                cv2.imshow('RealSense Frames', display_image)
            else:
                # Display just depth frame
                cv2.imshow('RealSense Depth', depth_colormap)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up
        rs.stop()
        cv2.destroyAllWindows()
        print("Camera stopped")

if __name__ == "__main__":
    main() 