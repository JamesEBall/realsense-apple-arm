# RealSense Depth & IR Camera Test

This C++ application demonstrates the use of an Intel RealSense depth camera (D421) to capture and display depth and infrared (IR) images in real-time. The application shows both the depth map (color-coded) and IR image side by side, with a depth scale visualization.

## Supported Platforms

- macOS (Apple Silicon & Intel)
- Linux (x86_64)
- Windows 10/11 (x86_64)

## Prerequisites

- Intel RealSense SDK 2.0 (librealsense2 >= 2.53.1)
- OpenCV 4.x (>= 4.8.0)
- CMake 3.10 or higher
- Modern C++ compiler with C++14 support
  - macOS: Apple Clang 14+ or GCC 11+
  - Linux: GCC 9+ or Clang 10+
  - Windows: MSVC 2019+ (v142)

### Installing Dependencies on macOS

```bash
# Install build tools
brew install cmake

# Install OpenCV
brew install opencv

# Install RealSense SDK
brew install librealsense

# Verify installations
cmake --version
pkg-config --modversion opencv4
pkg-config --modversion realsense2
```

### Installing Dependencies on Linux (Ubuntu/Debian)

```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y cmake build-essential pkg-config

# Install OpenCV dependencies
sudo apt-get install -y libopencv-dev

# Install RealSense SDK (see https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
sudo apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev
```

### Installing Dependencies on Windows

1. Install Visual Studio 2019 or later with C++ development tools
2. Install CMake from https://cmake.org/download/
3. Install OpenCV using vcpkg:
   ```bash
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   ./bootstrap-vcpkg.bat
   ./vcpkg install opencv:x64-windows
   ```
4. Install RealSense SDK from https://www.intelrealsense.com/sdk-2/

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