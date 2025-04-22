# RealSense for Apple Silicon

This repository contains a Python wrapper for Intel RealSense depth cameras that works natively on Apple Silicon Macs. It provides access to the core functionality of RealSense cameras including depth, infrared, and color streams.

## Features

- Native support for Apple Silicon (M1/M2/M3)
- Access to depth, infrared, and RGB streams
- Advanced depth visualization options
- Hand tracking using MediaPipe (optional)
- Auto depth range adjustment
- Compatible with most RealSense depth cameras

## Installation

### Requirements

- macOS on Apple Silicon
- Python 3.9+ (recommended)
- pip package manager

### Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/realsense_applesilicon.git
   cd realsense_applesilicon
   ```

2. Create a Python virtual environment:
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

4. Install additional requirements for hand tracking (optional):
   ```bash
   pip install mediapipe opencv-python numpy
   ```

## Usage

### Basic Example

Run the example script to see the depth camera in action:

```bash
cd examples
python example.py --advanced-depth
```

### Command Line Options

- `--advanced-depth`: Use advanced depth visualization
- `--no-hand-tracking`: Disable hand tracking
- `--min-depth`: Set minimum depth in meters (default: 0.1)
- `--max-depth`: Set maximum depth in meters (default: 5.0)
- `--no-auto-range`: Disable automatic depth range adjustment
- `--colormap`: Select colormap (TURBO, JET, PLASMA, VIRIDIS, HOT, RAINBOW)
- `--resolution`: Camera resolution (1280x720, 848x480, 640x360, 480x270, 424x240)
- `--fps`: Camera frame rate (5, 15, 30, 60, 90)
- `--enable-color`: Enable color stream

## Project Structure

- `realsense_python/`: Core Python package
  - `src/`: C++ implementation
  - `include/`: Header files
- `examples/`: Example scripts and applications
- `build/`: Build artifacts

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Intel RealSense SDK
- MediaPipe Project
