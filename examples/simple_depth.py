#!/usr/bin/env python3
"""
Simple RealSense depth camera example showing depth and infrared streams.
This script provides a minimal example of using the RealSense camera with depth visualization.
"""
import cv2
import numpy as np
import argparse
from realsense.wrapper import PyRealSense
import time

def create_depth_colormap(depth_frame, min_depth, max_depth, colormap=cv2.COLORMAP_TURBO):
    """Create a colormap from depth data"""
    # Convert min/max from meters to millimeters
    min_depth_mm = min_depth * 1000.0
    max_depth_mm = max_depth * 1000.0
    
    # Create a normalized depth map
    depth_normalized = np.zeros_like(depth_frame, dtype=np.uint8)
    
    # Only include pixels with valid depth and in range
    valid_mask = (depth_frame > 0) & (depth_frame >= min_depth_mm) & (depth_frame <= max_depth_mm)
    
    # Apply normalization if we have valid data
    if np.any(valid_mask):
        depth_range = max_depth_mm - min_depth_mm
        if depth_range > 0:
            depth_normalized[valid_mask] = np.clip(
                255 * (max_depth_mm - depth_frame[valid_mask]) / depth_range, 
                0, 255
            ).astype(np.uint8)
    
    # Apply colormap and mark invalid areas
    colormap = cv2.applyColorMap(depth_normalized, colormap)
    colormap[~valid_mask] = [30, 30, 30]  # Dark gray for invalid areas
    
    return colormap

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simple RealSense Depth Camera Example')
    parser.add_argument('--min-depth', type=float, default=0.1, help='Minimum depth in meters')
    parser.add_argument('--max-depth', type=float, default=4.0, help='Maximum depth in meters')
    parser.add_argument('--resolution', type=str, default='640x360', 
                      choices=['1280x720', '848x480', '640x360', '480x270', '424x240'],
                      help='Camera resolution (width x height)')
    parser.add_argument('--fps', type=int, default=30, 
                      choices=[5, 15, 30, 60, 90],
                      help='Frame rate (fps)')
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Initialize RealSense camera
    print(f"Initializing RealSense with resolution {width}x{height} @ {args.fps}fps")
    rs = PyRealSense(
        width=width,
        height=height,
        framerate=args.fps,
        enable_color=False,
        enable_ir=True,
        min_depth=args.min_depth,
        max_depth=args.max_depth
    )
    
    # Start the camera
    try:
        rs.start()
        print(f"Camera ready: {rs.camera_model_name}")
        print("Press 'q' to quit")
        
        # Auto-adjust depth range variables
        depth_min = args.min_depth
        depth_max = args.max_depth
        auto_range = True
        
        while True:
            # Get frames
            frames = rs.get_frames()
            if frames is None:
                continue
            
            # Extract depth and IR frames
            depth_frame = frames.get('depth')
            ir_frame = frames.get('infrared')
            
            # Skip if frames are invalid
            if depth_frame is None or ir_frame is None:
                continue
            
            # Auto-adjust depth range for better visualization
            if auto_range and depth_frame is not None:
                valid_depths = depth_frame[depth_frame > 0]
                if len(valid_depths) > 0:
                    # Convert to meters
                    valid_depths_m = valid_depths / 1000.0
                    
                    # Get 5th and 95th percentiles for min/max
                    min_depth_scene = np.percentile(valid_depths_m, 5)
                    max_depth_scene = np.percentile(valid_depths_m, 95)
                    
                    # Smooth adaptation
                    depth_min = 0.8 * depth_min + 0.2 * min_depth_scene
                    depth_max = 0.8 * depth_max + 0.2 * max_depth_scene
                    
                    # Ensure minimum range
                    if depth_max - depth_min < 0.5:
                        mid_point = (depth_max + depth_min) / 2
                        depth_min = mid_point - 0.25
                        depth_max = mid_point + 0.25
            
            # Create depth colormap
            depth_colormap = create_depth_colormap(depth_frame, depth_min, depth_max)
            
            # Convert IR to BGR for display
            if len(ir_frame.shape) == 2:
                ir_viz = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
            else:
                ir_viz = ir_frame.copy()
            
            # Create a combined display
            display = np.hstack((depth_colormap, ir_viz))
            
            # Add depth range info
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display, f"Depth range: {depth_min:.2f}m - {depth_max:.2f}m", 
                      (10, 30), font, 0.7, (255, 255, 255), 2)
            
            # Show the combined image
            cv2.imshow('RealSense', display)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Stop the camera
        rs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 