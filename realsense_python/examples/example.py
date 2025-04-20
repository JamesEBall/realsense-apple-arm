import cv2
import numpy as np
from realsense.wrapper import PyRealSense

def main():
    # Create RealSense instance with only depth and IR enabled
    rs = PyRealSense(enable_color=False, enable_ir=True)  # Disable color stream for now
    
    try:
        # Start the camera
        rs.start()
        
        # Print camera information
        print(f"Camera Model: {rs.camera_model_name}")
        print(f"USB Mode: {'3.1' if rs.is_usb3_mode else '2.0'}")
        print(f"Frame Rate: {rs.frame_rate}")
        print(f"Resolution: {rs.frame_width}x{rs.frame_height}")
        
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