# distutils: language = c++
# distutils: include_dirs = /opt/homebrew/include
# distutils: libraries = realsense2

from libc.stdint cimport uint8_t, uint16_t, uint32_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr, shared_ptr
import numpy as np
cimport numpy as np
from libc.string cimport memcpy
from libcpp.string cimport string
from libcpp cimport bool
from cython.operator cimport dereference as deref

np.import_array()

# Define constants
cdef int RS2_API_MAJOR_VERSION = 2
cdef int RS2_API_MINOR_VERSION = 56
cdef int RS2_API_PATCH_VERSION = 3
cdef int RS2_API_VERSION = (RS2_API_MAJOR_VERSION * 10000) + (RS2_API_MINOR_VERSION * 100) + RS2_API_PATCH_VERSION

# Camera model definitions
cdef enum CameraModel:
    D410 = 0x0AD2
    D415 = 0x0AD3
    D421 = 0x1155  # D421 model - stereo depth camera with dual IR sensors
    D430 = 0x0AD4
    D435 = 0x0B07
    D435i = 0x0B3A
    D450 = 0x0B5C
    D455 = 0x0B5C
    D401 = 0x0B5B
    D405 = 0x0B5B
    D435f = 0x0B07
    D435if = 0x0B3A
    D455f = 0x0B5C
    D456 = 0x0B5C

# Declare the C++ API
cdef extern from "librealsense2/h/rs_types.h":
    ctypedef enum rs2_stream:
        RS2_STREAM_ANY = 0
        RS2_STREAM_DEPTH = 1
        RS2_STREAM_COLOR = 2
        RS2_STREAM_INFRARED = 3
        RS2_STREAM_GYRO = 4
        RS2_STREAM_ACCEL = 5
        
    ctypedef enum rs2_format:
        RS2_FORMAT_ANY = 0
        RS2_FORMAT_Z16 = 1
        RS2_FORMAT_DISPARITY16 = 2
        RS2_FORMAT_XYZ32F = 3
        RS2_FORMAT_YUYV = 4
        RS2_FORMAT_RGB8 = 5
        RS2_FORMAT_BGR8 = 6
        RS2_FORMAT_RGBA8 = 7
        RS2_FORMAT_BGRA8 = 8
        RS2_FORMAT_Y8 = 9
        RS2_FORMAT_Y16 = 10
        RS2_FORMAT_RAW10 = 11
        RS2_FORMAT_RAW16 = 12
        RS2_FORMAT_RAW8 = 13
        RS2_FORMAT_UYVY = 14
        RS2_FORMAT_MOTION_RAW = 15
        RS2_FORMAT_MOTION_XYZ32F = 16
        RS2_FORMAT_GPIO_RAW = 17
        RS2_FORMAT_6DOF = 18
        RS2_FORMAT_DISPARITY32 = 19
        RS2_FORMAT_Y10BPACK = 20
        RS2_FORMAT_DISTANCE = 21
        RS2_FORMAT_MJPEG = 22
        RS2_FORMAT_Y8I = 23
        RS2_FORMAT_Y12I = 24
        RS2_FORMAT_INZI = 25
        RS2_FORMAT_INVI = 26
        RS2_FORMAT_W10 = 27
        RS2_FORMAT_Z16H = 28
        RS2_FORMAT_FG = 29
        RS2_FORMAT_Y411 = 30
        RS2_FORMAT_COUNT = 31
        
    ctypedef enum rs2_camera_info:
        RS2_CAMERA_INFO_NAME = 0
        RS2_CAMERA_INFO_SERIAL_NUMBER = 1
        RS2_CAMERA_INFO_FIRMWARE_VERSION = 2
        RS2_CAMERA_INFO_RECOMMENDED_FIRMWARE_VERSION = 3
        RS2_CAMERA_INFO_PHYSICAL_PORT = 4
        RS2_CAMERA_INFO_DEBUG_OP_CODE = 5
        RS2_CAMERA_INFO_ADVANCED_MODE = 6
        RS2_CAMERA_INFO_PRODUCT_ID = 7
        RS2_CAMERA_INFO_CAMERA_LOCKED = 8
        RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR = 9
        RS2_CAMERA_INFO_PRODUCT_LINE = 10
        RS2_CAMERA_INFO_ASIC_SERIAL_NUMBER = 11
        RS2_CAMERA_INFO_FIRMWARE_UPDATE_ID = 12
        RS2_CAMERA_INFO_COUNT = 13

    ctypedef enum rs2_option:
        RS2_OPTION_BACKLIGHT_COMPENSATION = 0
        RS2_OPTION_BRIGHTNESS = 1
        RS2_OPTION_CONTRAST = 2
        RS2_OPTION_EXPOSURE = 3
        RS2_OPTION_GAIN = 4
        RS2_OPTION_GAMMA = 5
        RS2_OPTION_HUE = 6
        RS2_OPTION_SATURATION = 7
        RS2_OPTION_SHARPNESS = 8
        RS2_OPTION_WHITE_BALANCE = 9
        RS2_OPTION_ENABLE_AUTO_EXPOSURE = 10
        RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE = 11

cdef extern from "librealsense2/rs.hpp" namespace "rs2":
    cdef cppclass context:
        context() except +
        device_list query_devices() except +
        
    cdef cppclass device_list:
        int size() except +
        device operator[](int) except +
        
    cdef cppclass device:
        string get_info(rs2_camera_info) except +
        bool supports(rs2_camera_info) except +
        
    cdef cppclass sensor:
        void set_option(rs2_option option, float value) except +
        float get_option(rs2_option option) except +
        bool supports(rs2_option option) except +
        
    cdef cppclass pipeline:
        pipeline(context&) except +
        void start() except +
        void stop() except +
        frameset wait_for_frames(unsigned int timeout_ms) except +
        
    cdef cppclass config:
        config() except +
        void enable_stream(rs2_stream stream_type, int stream_index, int width, int height, rs2_format format, int framerate) except +
        void enable_all_streams() except +
        
    cdef cppclass frameset:
        frame get_depth_frame() except +
        frame get_infrared_frame() except +
        frame get_color_frame() except +
        
    cdef cppclass frame:
        const void* get_data() except +
        int get_width() except +
        int get_height() except +
        int get_stride_in_bytes() except +
        rs2_stream get_stream_type() except +
        rs2_format get_format() except +

# Create the wrapper class
cdef class PyRealSense:
    cdef context* ctx
    cdef pipeline* pipe
    cdef config* cfg
    cdef int width
    cdef int height
    cdef int framerate
    cdef bint running
    cdef bint enable_color
    cdef bint enable_ir
    cdef bint enable_imu
    cdef CameraModel camera_model
    cdef bint is_usb3
    cdef float min_depth
    cdef float max_depth
    cdef bint auto_exposure
    cdef bint is_initialized
    cdef int _width
    cdef int _height
    cdef int _frame_rate
    cdef int _camera_model
    cdef bint _is_usb3
    cdef bint _is_running
    cdef str _camera_model_name
    
    def __cinit__(self, width=640, height=480, framerate=30, enable_color=False, enable_ir=True, enable_imu=False, min_depth=0.0, max_depth=10.0, auto_exposure=True):
        print("Initializing RealSense...")  # Debug print
        self.width = width
        self.height = height
        self.framerate = framerate
        self.enable_color = enable_color
        self.enable_ir = enable_ir
        self.enable_imu = enable_imu
        self.running = False
        self.is_usb3 = True  # Default to USB 3.1 mode
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.auto_exposure = auto_exposure
        self.is_initialized = False
        self._width = width
        self._height = height
        self._frame_rate = framerate
        self._camera_model = 0
        self._is_usb3 = True
        self._is_running = False
        self._camera_model_name = ""
        
        try:
            self.ctx = new context()
            print("Context created")  # Debug print
            
            # Detect camera model and USB mode
            self._detect_camera()
            
            # Validate resolution and frame rate
            self._validate_resolution_and_fps()
            
            self.pipe = new pipeline(deref(self.ctx))
            print("Pipeline created")  # Debug print
            
            self.cfg = new config()
            print("Config created")  # Debug print
            
            # Configure streams based on camera model and USB mode
            self._configure_streams()
            
            print("Streams configured")  # Debug print
            
        except Exception as e:
            print(f"Error in initialization: {str(e)}")  # Debug print
            self.cleanup()
            raise
    
    cdef void _detect_camera(self):
        cdef device_list devices = self.ctx.query_devices()
        if devices.size() == 0:
            raise RuntimeError("No RealSense devices found")
            
        cdef device dev = devices[0]
        cdef string pid_str = dev.get_info(RS2_CAMERA_INFO_PRODUCT_ID)
        cdef int pid = int(pid_str.decode('utf-8'), 16)
        
        # Map PID to camera model
        if pid == D410:
            self.camera_model = D410
        elif pid == D415:
            self.camera_model = D415
        elif pid == D421:
            self.camera_model = D421
        elif pid == D430:
            self.camera_model = D430
        elif pid == D435 or pid == D435f:
            self.camera_model = D435
        elif pid == D435i or pid == D435if:
            self.camera_model = D435i
        elif pid == D450 or pid == D455 or pid == D455f or pid == D456:
            self.camera_model = D450
        elif pid == D401 or pid == D405:
            self.camera_model = D401
        else:
            raise RuntimeError(f"Unsupported camera model with PID: {pid}")
            
        # Check USB mode
        cdef string usb_type = dev.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR)
        self.is_usb3 = "3.1" in usb_type.decode('utf-8')
        
        print(f"Detected camera model: {self.camera_model}, USB mode: {'3.1' if self.is_usb3 else '2.0'}")
    
    cdef void _validate_resolution_and_fps(self):
        """Validate resolution and frame rate combination based on camera model and USB mode."""
        # Check if resolution is supported
        valid_resolutions = [
            (1280, 720),
            (848, 480),
            (640, 360),
            (480, 270),
            (424, 240)
        ]
        
        if (self.width, self.height) not in valid_resolutions:
            valid_res_str = ", ".join([f"{w}x{h}" for w, h in valid_resolutions])
            raise ValueError(f"Unsupported resolution: {self.width}x{self.height}. Supported resolutions: {valid_res_str}")
        
        # Validate frame rate based on resolution and USB mode
        if self.is_usb3:
            # USB 3.1 mode
            if self.width == 1280 and self.height == 720:
                if self.framerate not in [5, 15, 30]:
                    raise ValueError(f"Unsupported frame rate {self.framerate} for resolution {self.width}x{self.height} in USB 3.1 mode. Supported rates: 5, 15, 30")
                self.framerate = min(self.framerate, 30)
            elif (self.width == 848 and self.height == 480) or \
                 (self.width == 640 and self.height == 360) or \
                 (self.width == 480 and self.height == 270) or \
                 (self.width == 424 and self.height == 240):
                if self.framerate not in [5, 15, 30, 60, 90]:
                    raise ValueError(f"Unsupported frame rate {self.framerate} for resolution {self.width}x{self.height} in USB 3.1 mode. Supported rates: 5, 15, 30, 60, 90")
                self.framerate = min(self.framerate, 90)
        else:
            # USB 2.0 mode (more restricted)
            if self.width == 1280 and self.height == 720:
                # Lower FPS for USB 2.0
                valid_rates = [5]
                if self.framerate not in valid_rates:
                    raise ValueError(f"Unsupported frame rate {self.framerate} for resolution {self.width}x{self.height} in USB 2.0 mode. Supported rates: {', '.join(map(str, valid_rates))}")
                self.framerate = min(self.framerate, 5)
            elif self.width == 848 and self.height == 480:
                valid_rates = [5, 10]
                if self.framerate not in valid_rates:
                    raise ValueError(f"Unsupported frame rate {self.framerate} for resolution {self.width}x{self.height} in USB 2.0 mode. Supported rates: {', '.join(map(str, valid_rates))}")
                self.framerate = min(self.framerate, 10)
            elif (self.width == 640 and self.height == 360) or (self.width == 480 and self.height == 270) or (self.width == 424 and self.height == 240):
                valid_rates = [5, 15, 30]
                if self.framerate not in valid_rates:
                    raise ValueError(f"Unsupported frame rate {self.framerate} for resolution {self.width}x{self.height} in USB 2.0 mode. Supported rates: {', '.join(map(str, valid_rates))}")
                self.framerate = min(self.framerate, 30)
        
        print(f"Using resolution {self.width}x{self.height} at {self.framerate} FPS")
    
    cdef void _configure_streams(self):
        # Configure depth stream
        self.cfg.enable_stream(RS2_STREAM_DEPTH, 0, self.width, self.height, RS2_FORMAT_Z16, self.framerate)
        
        # Configure infrared stream if enabled
        if self.enable_ir:
            # For D421, enable the left IR stream (index 1) as the main IR stream
            # This ensures get_infrared_frame() will return the left camera
            if self.camera_model == D421:
                self.cfg.enable_stream(RS2_STREAM_INFRARED, 1, self.width, self.height, RS2_FORMAT_Y8, self.framerate)  # Left IR only
            else:
                self.cfg.enable_stream(RS2_STREAM_INFRARED, 0, self.width, self.height, RS2_FORMAT_Y8, self.framerate)
        
        # Configure color stream if enabled and supported
        if self.enable_color and self.camera_model not in [D421]:  # D421 doesn't have color
            if self.camera_model in [D415, D435, D435i, D435f, D435if, D450, D455, D455f, D456]:
                # Use YUYV for RGB camera
                self.cfg.enable_stream(RS2_STREAM_COLOR, 0, self.width, self.height, RS2_FORMAT_YUYV, self.framerate)
            else:
                # Use UYVY for left imager
                self.cfg.enable_stream(RS2_STREAM_COLOR, 0, self.width, self.height, RS2_FORMAT_UYVY, self.framerate)
        
        # Configure IMU if enabled and supported
        if self.enable_imu and self.camera_model in [D435i, D435if, D455, D455f, D456]:
            self.cfg.enable_stream(RS2_STREAM_GYRO, 0, 0, 0, RS2_FORMAT_MOTION_XYZ32F, 200)  # 200 Hz for gyro
            self.cfg.enable_stream(RS2_STREAM_ACCEL, 0, 0, 0, RS2_FORMAT_MOTION_XYZ32F, 63)  # 63 Hz for accelerometer
    
    def __dealloc__(self):
        self.cleanup()
    
    cdef void cleanup(self):
        if self.pipe and self.running:
            try:
                self.pipe.stop()
            except:
                pass
            
        if self.cfg:
            del self.cfg
            self.cfg = NULL
            
        if self.pipe:
            del self.pipe
            self.pipe = NULL
            
        if self.ctx:
            del self.ctx
            self.ctx = NULL
    
    def start(self):
        print("Starting pipeline...")  # Debug print
        if not self.running:
            try:
                self.pipe.start()
                self.running = True
                print("Pipeline started successfully")  # Debug print
                
                # Configure auto-exposure settings
                if not self.auto_exposure:
                    self._disable_auto_exposure()
                
            except Exception as e:
                print(f"Error starting pipeline: {str(e)}")  # Debug print
                raise
    
    def stop(self):
        if self.running:
            try:
                self.pipe.stop()
            finally:
                self.running = False
    
    def get_frames(self):
        cdef frameset frames
        cdef frame depth_frame
        cdef frame ir_frame
        cdef frame color_frame
        cdef const uint16_t* depth_data = NULL
        cdef const uint8_t* ir_data = NULL
        cdef const uint8_t* color_data = NULL
        cdef np.ndarray[np.uint16_t, ndim=2] depth_array
        cdef np.ndarray[np.uint8_t, ndim=2] ir_array
        cdef np.ndarray[np.uint8_t, ndim=3] color_array
        
        if not self.running:
            raise RuntimeError("Camera not started")
        
        try:
            # Get frames
            frames = deref(self.pipe).wait_for_frames(5000)  # 5 second timeout
            
            try:
                # Get depth frame
                try:
                    depth_frame = frames.get_depth_frame()
                    depth_data = <const uint16_t*>depth_frame.get_data()
                except Exception as e:
                    print(f"Warning: Failed to get depth frame: {e}")
                    return None
                    
                if depth_data == NULL:
                    print("Warning: Null depth data")
                    return None
                    
                depth_array = np.zeros((self.height, self.width), dtype=np.uint16)
                memcpy(depth_array.data, depth_data, self.width * self.height * sizeof(uint16_t))
                
                # Apply depth filtering
                if self.min_depth > 0 or self.max_depth < 10.0:
                    # Convert depth values to meters (assuming Z16 format)
                    depth_meters = depth_array.astype(np.float32) / 1000.0
                    # Create mask for values outside the range
                    mask = (depth_meters < self.min_depth) | (depth_meters > self.max_depth)
                    # Set invalid values to 0
                    depth_array[mask] = 0
                
                result = {'depth': depth_array}
                
                # Get IR frame if enabled
                if self.enable_ir:
                    try:
                        # For all camera models, use the standard get_infrared_frame method
                        # D421 will need special handling in stream configuration, not here
                        ir_frame = frames.get_infrared_frame()
                        ir_data = <const uint8_t*>ir_frame.get_data()
                    except Exception as e:
                        print(f"Warning: Failed to get IR frame: {e}")
                        ir_data = NULL
                        
                    if ir_data != NULL:
                        ir_array = np.zeros((self.height, self.width), dtype=np.uint8)
                        memcpy(ir_array.data, ir_data, self.width * self.height * sizeof(uint8_t))
                        result['infrared'] = ir_array
                
                # Get color frame if enabled
                if self.enable_color:
                    try:
                        color_frame = frames.get_color_frame()
                        color_data = <const uint8_t*>color_frame.get_data()
                    except Exception as e:
                        print(f"Warning: Failed to get color frame: {e}")
                        color_data = NULL
                        
                    if color_data != NULL:
                        if self.camera_model in [D415, D435, D435i, D435f, D435if, D450, D455, D455f, D456]:
                            # YUYV format for RGB camera
                            color_array = np.zeros((self.height, self.width, 2), dtype=np.uint8)
                            memcpy(color_array.data, color_data, self.width * self.height * 2 * sizeof(uint8_t))
                        else:
                            # UYVY format for left imager
                            color_array = np.zeros((self.height, self.width, 2), dtype=np.uint8)
                            memcpy(color_array.data, color_data, self.width * self.height * 2 * sizeof(uint8_t))
                        result['color'] = color_array
                
                return result
                
            except Exception as e:
                print(f"Error processing frames: {str(e)}")
                return None
            
        except Exception as e:
            print(f"Error in get_frames: {str(e)}")
            return None
    
    @property
    def frame_width(self):
        return self.width
    
    @property
    def frame_height(self):
        return self.height
    
    @property
    def frame_rate(self):
        return self.framerate
    
    @property
    def camera_model_name(self):
        if self.camera_model == D410:
            return "D410"
        elif self.camera_model == D415:
            return "D415"
        elif self.camera_model == D421:
            return "D421"
        elif self.camera_model == D430:
            return "D430"
        elif self.camera_model == D435:
            return "D435"
        elif self.camera_model == D435i:
            return "D435i"
        elif self.camera_model == D450:
            return "D450"
        elif self.camera_model == D401:
            return "D401"
        elif self.camera_model == D405:
            return "D405"
        else:
            return "Unknown"
    
    @property
    def is_usb3_mode(self):
        return self.is_usb3 

    cdef void _disable_auto_exposure(self):
        """Configure auto-exposure settings for all sensors."""
        print("Disabling auto-exposure...")
        
        try:
            # Access devices directly through librealsense C API
            # Since we can't use create_sensor, we'll need to add code to set auto-exposure after pipeline starts
            # This is a simplified version that doesn't use sensors directly
            # In a real implementation, you would need to get access to sensor objects
            print("Auto-exposure features not fully implemented yet")
            print("Setting manual exposure requires direct access to sensor objects")
            
        except Exception as e:
            print(f"Error configuring auto-exposure: {str(e)}")
            # Don't raise, just log the error and continue 