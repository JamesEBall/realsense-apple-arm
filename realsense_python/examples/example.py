import cv2
import numpy as np
from realsense_wrapper import PyRealSense

def main():
    # Create RealSense instance
    rs = PyRealSense()
    
    try:
        # Start the camera
        rs.start()
        
        while True:
            # Get depth and IR frames
            depth_frame, ir_frame = rs.get_frames()
            
            # Convert depth to color for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Convert IR to color for visualization
            ir_colormap = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
            
            # Stack images horizontally
            images = np.hstack((depth_colormap, ir_colormap))
            
            # Show images
            cv2.imshow('RealSense', images)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Stop the camera
        rs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 