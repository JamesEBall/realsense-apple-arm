# RealSense for Apple Silicon

A high-performance Intel RealSense camera implementation for Apple Silicon (M) Macs, featuring:

- Optimized Cython wrapper for Python applications
- Direct memory access for efficient frame capture
- Native support for depth and infrared streams
- OpenCV integration for real-time visualization
- C++ test suite for hardware validation

This project bridges the gap between Intel RealSense SDK and ARM-based Mac systems, providing a seamless experience for computer vision applications on Apple Silicon.

## Example Output

![Example RealSense Output](realsense_python/docs/example_image.png)

## Features

### Python Wrapper

- Efficient Cython-based interface to RealSense SDK
- Direct memory mapping for fast frame access
- NumPy array integration for data processing
- Real-time visualization with OpenCV
- Simple, intuitive API

### C++ Implementation

- Native performance test suite
- Hardware validation tools
- Direct SDK integration
- Performance benchmarking

## Installation

### Prerequisites

- macOS on Apple Silicon (M1/M2)
- Python 3.8+
- Intel RealSense SDK 2.0
- OpenCV
- Cython

### Setup

1. Install the Intel RealSense SDK 2.0:

```bash
brew install librealsense
```

2. Install the Python package:

```bash
cd realsense_python
python -m venv venv
source venv/bin/activate
pip install -e .
```

> **Important Note**: When running RealSense applications with pyenv on macOS, you must use `sudo` or you will encounter permission errors. This is because RealSense cameras require elevated permissions to access USB devices. For example:
>
> ```bash
> sudo pyenv exec python example.py
> ```
>
> Without sudo, you may get "Access denied" or "Device not found" errors when trying to access the camera.

## Usage

### Python Interface

```python
from realsense import PyRealSense

# Initialize camera
camera = PyRealSense()

# Start the camera
camera.start()

try:
    while True:
        # Get depth and IR frames
        depth_frame, ir_frame = camera.get_frames()
    
        # Process frames (example with OpenCV)
        cv2.imshow('Depth', depth_frame)
        cv2.imshow('IR', ir_frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.stop()
```

### C++ Test Application

Build and run the C++ test suite:

```bash
cd cpp_test
mkdir build && cd build
cmake ..
make
./realsense_test
```

## Project Structure

```
realsense_applesilicon/
├── realsense_python/       # Python package
│   ├── src/               # Source code
│   │   └── realsense/    # Python module
│   ├── tests/            # Python tests
│   ├── examples/         # Example scripts
│   └── docs/             # Documentation
├── cpp_test/             # C++ test application
│   ├── CMakeLists.txt
│   └── realsense_test.cpp
└── README.md
```

## Development

### Python Development

```bash
cd realsense_python
python -m venv venv
source venv/bin/activate
pip install -e .
python tests/test_realsense.py
```

### C++ Development

```bash
cd cpp_test
mkdir build && cd build
cmake ..
make
./realsense_test
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License
