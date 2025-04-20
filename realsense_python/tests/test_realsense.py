import numpy as np
import cv2
from realsense import PyRealSense

def main():
    # Initialize camera
    camera = PyRealSense()
    
    try:
        # Start the camera
        print("Starting camera...")
        camera.start()
        
        print("Press 'q' to quit")
        while True:
            # Get frames
            depth_frame, ir_frame = camera.get_frames()
            
            # Normalize depth for visualization
            depth_colormap = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)
            
            # Display frames
            cv2.imshow('Depth', depth_colormap)
            cv2.imshow('IR', ir_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean up
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 