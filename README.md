# RealSense Depth & IR Camera Test

This C++ application demonstrates the use of an Intel RealSense depth camera (D421) to capture and display depth and infrared (IR) images in real-time. The application shows both the depth map (color-coded) and IR image side by side, with a depth scale visualization.

## Prerequisites

- Intel RealSense SDK 2.0 (C++ API)
- OpenCV 4.x (C++ API)
- CMake 3.10 or higher
- Modern C++ compiler with C++14 support
  - macOS: Apple Clang
  - Linux: GCC 5+ or Clang
  - Windows: MSVC 2017+

### Installing Dependencies on macOS

```bash
# Install OpenCV
brew install opencv

# Install RealSense SDK
brew install librealsense
```

## Building the Application

1. Create a build directory and navigate to it:
```bash
mkdir build && cd build
```

2. Generate the build files:
```bash
cmake ..
```

3. Build the application:
```bash
make
```

## Running the Application

The application requires elevated permissions to access the RealSense camera. Run it using:

```bash
sudo ./realsense_test
```

### Display

The application shows:
- Left side: Depth image (color-coded)
- Right side: Infrared image
- Bottom: Depth scale visualization
- Top-left: FPS counter and maximum depth information

### Controls

- Press 'q' or 'Q' to quit the application

## Features

- Real-time depth and IR image display using modern C++ and OpenCV
- Depth colorization using a rainbow color map
- Bilateral filtering for depth noise reduction
- Histogram equalization for IR image enhancement
- Dynamic depth range adjustment
- FPS counter
- Depth scale visualization
- Exception-safe RAII resource management
- Modern C++ features:
  - Smart pointers for resource management
  - Auto type deduction
  - Range-based for loops
  - Standard library containers and algorithms

## Implementation Details

The application uses modern C++ features and best practices:
- RealSense SDK's C++ API for camera interaction
- OpenCV's C++ API for image processing
- RAII principles for resource management
- Exception handling for robust error management
- Standard library containers and algorithms

## Troubleshooting

If you encounter "No RealSense devices found!", check:
1. Camera is properly connected
2. Camera permissions are granted
3. USB connection is stable

For OpenCV include errors, ensure OpenCV is properly installed and CMake can find it:
```bash
brew install opencv
```

For RealSense SDK errors, reinstall the SDK:
```bash
brew reinstall librealsense
``` 