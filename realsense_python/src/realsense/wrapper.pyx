# distutils: language = c++
# distutils: include_dirs = /opt/homebrew/include
# distutils: libraries = realsense2

from libc.stdint cimport uint8_t, uint16_t
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

# Declare the C++ API
cdef extern from "librealsense2/h/rs_types.h":
    ctypedef enum rs2_stream:
        RS2_STREAM_ANY = 0
        RS2_STREAM_DEPTH = 1
        RS2_STREAM_COLOR = 2
        RS2_STREAM_INFRARED = 3
        
    ctypedef enum rs2_format:
        RS2_FORMAT_ANY = 0
        RS2_FORMAT_Z16 = 1
        RS2_FORMAT_DISPARITY16 = 2
        RS2_FORMAT_Y8 = 3

cdef extern from "librealsense2/rs.hpp" namespace "rs2":
    cdef cppclass context:
        context() except +
        
    cdef cppclass pipeline:
        pipeline(context&) except +
        void start() except +
        void stop() except +
        frameset wait_for_frames(unsigned int timeout_ms) except +
        
    cdef cppclass config:
        config() except +
        void enable_stream(rs2_stream stream_type, int stream_index, int width, int height, rs2_format format, int framerate) except +
        
    cdef cppclass frameset:
        frame get_depth_frame() except +
        frame get_infrared_frame() except +
        
    cdef cppclass frame:
        const void* get_data() except +
        int get_width() except +
        int get_height() except +
        int get_stride_in_bytes() except +

# Create the wrapper class
cdef class PyRealSense:
    cdef context* ctx
    cdef pipeline* pipe
    cdef config* cfg
    cdef int width
    cdef int height
    cdef bint running
    
    def __cinit__(self):
        print("Initializing RealSense...")  # Debug print
        self.width = 640
        self.height = 480
        self.running = False
        
        try:
            self.ctx = new context()
            print("Context created")  # Debug print
            
            self.pipe = new pipeline(deref(self.ctx))
            print("Pipeline created")  # Debug print
            
            self.cfg = new config()
            print("Config created")  # Debug print
            
            # Configure streams
            self.cfg.enable_stream(RS2_STREAM_DEPTH, 0, self.width, self.height, RS2_FORMAT_Z16, 30)
            self.cfg.enable_stream(RS2_STREAM_INFRARED, 0, self.width, self.height, RS2_FORMAT_Y8, 30)
            print("Streams configured")  # Debug print
            
        except Exception as e:
            print(f"Error in initialization: {str(e)}")  # Debug print
            self.cleanup()
            raise
    
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
        cdef const uint16_t* depth_data
        cdef const uint8_t* ir_data
        cdef np.ndarray[np.uint16_t, ndim=2] depth_array
        cdef np.ndarray[np.uint8_t, ndim=2] ir_array
        
        if not self.running:
            raise RuntimeError("Camera not started")
        
        print("Waiting for frames...")  # Debug print
        try:
            # Get frames
            frames = deref(self.pipe).wait_for_frames(5000)  # 5 second timeout
            print("Got frameset")  # Debug print
            
            depth_frame = frames.get_depth_frame()
            ir_frame = frames.get_infrared_frame()
            print("Extracted individual frames")  # Debug print
            
            # Get frame data
            depth_data = <const uint16_t*>depth_frame.get_data()
            ir_data = <const uint8_t*>ir_frame.get_data()
            print("Got frame data pointers")  # Debug print
            
            # Create numpy arrays
            depth_array = np.zeros((self.height, self.width), dtype=np.uint16)
            ir_array = np.zeros((self.height, self.width), dtype=np.uint8)
            
            # Copy data
            memcpy(depth_array.data, depth_data, self.width * self.height * sizeof(uint16_t))
            memcpy(ir_array.data, ir_data, self.width * self.height * sizeof(uint8_t))
            print("Data copied to numpy arrays")  # Debug print
            
            return depth_array, ir_array
            
        except Exception as e:
            print(f"Error in get_frames: {str(e)}")  # Debug print
            raise
    
    @property
    def frame_width(self):
        return self.width
    
    @property
    def frame_height(self):
        return self.height 