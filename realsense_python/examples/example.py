import cv2
import numpy as np
import argparse
from realsense.wrapper import PyRealSense

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RealSense Depth Camera Example')
    parser.add_argument('--min-depth', type=float, default=0.0, help='Minimum depth in meters')
    parser.add_argument('--max-depth', type=float, default=10.0, help='Maximum depth in meters')
    parser.add_argument('--enable-color', action='store_true', help='Enable color stream')
    args = parser.parse_args()
    
    # Create RealSense instance with depth filtering
    rs = PyRealSense(
        enable_color=args.enable_color,
        enable_ir=True,
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
            
            # Initialize display image
            display_image = depth_colormap
            
            # Add IR frame if available
            if 'infrared' in frames:
                ir_frame = frames['infrared']
                ir_colormap = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
                display_image = np.hstack((depth_colormap, ir_colormap))
            
            # Add color frame if available
            if 'color' in frames:
                color_frame = frames['color']
                # Convert YUYV/UYVY to BGR
                color_bgr = cv2.cvtColor(color_frame, cv2.COLOR_YUV2BGR_YUYV)
                display_image = np.hstack((display_image, color_bgr))
            
            # Show images
            cv2.imshow('RealSense', display_image)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Stop the camera
        rs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 