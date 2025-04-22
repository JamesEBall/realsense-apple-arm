# Setting up RealSense for Apple Silicon

This guide provides step-by-step instructions for setting up the RealSense Python wrapper on Apple Silicon Macs.

## Prerequisites

Make sure you have the following before you begin:

- macOS on Apple Silicon (M1/M2/M3)
- An Intel RealSense depth camera (D415, D435, D435i, D455, etc.)
- Python installed (3.9 or later recommended)
- Command Line Tools for Xcode (can be installed with `xcode-select --install`)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/realsense_applesilicon.git
cd realsense_applesilicon
```

### 2. Create a Python Virtual Environment

For optimal compatibility with MediaPipe and other dependencies, we recommend using Python 3.9:

```bash
# Check available Python versions
which -a python3.9 python3.10 python3.11

# Create a virtual environment with Python 3.9
python3.9 -m venv venv39
source venv39/bin/activate
```

### 3. Install the Package

```bash
pip install -e .
```

### 4. Install Hand Tracking Dependencies (Optional)

If you want to use hand tracking features:

```bash
pip install mediapipe opencv-python numpy
```

### 5. Run the Examples

```bash
# Simple depth visualization example
cd examples
python simple_depth.py

# Advanced example with hand tracking
python example.py --advanced-depth
```

## Troubleshooting

### Camera Not Detected

If your camera is not detected:

1. Ensure the camera is properly plugged in
2. Try a different USB port or cable
3. Restart the application
4. Check system permissions for USB devices

### Error: "Permission denied accessing USB device"

On macOS, you might need to grant permissions for terminal applications to access the camera:

1. Go to System Preferences → Security & Privacy → Privacy → Camera
2. Ensure Terminal (or your IDE) has permission to access the camera

### MediaPipe Not Installing

MediaPipe has specific Python version requirements. It works best with Python 3.9 or 3.10 on Apple Silicon. If you're having issues:

```bash
# Try creating a new environment with Python 3.9
python3.9 -m venv venv39
source venv39/bin/activate
pip install mediapipe
```

## Performance Tips

- For optimal performance, use a high-quality USB-C cable
- USB 3.1 or Thunderbolt ports provide the best performance
- Lower the resolution and frame rate if you experience lag
- Close other applications that might use the camera or GPU
- On MacBook laptops, keep the device plugged in for best performance

## Questions and Support

If you encounter any issues, please open an issue on the repository or contact the project maintainers. 