# RealSense Examples for Apple Silicon

## Cold Setup: Apple Silicon + pyenv + MediaPipe

**Important:** For best compatibility (especially with MediaPipe), use Python 3.9 (or 3.10) via [pyenv](https://github.com/pyenv/pyenv) on Apple Silicon. MediaPipe does **not** support Python 3.12+ as of mid-2024.

### Step-by-step cold setup

1. **Install pyenv** (if not already):
   ```sh
   brew install pyenv
   ```
2. **Install Python 3.9:**
   ```sh
   pyenv install 3.9.18
   ```
3. **Set local Python version:**
   ```sh
   pyenv local 3.9.18
   ```
   This creates a `.python-version` file in your project root.
4. **Create a virtual environment using pyenv's Python:**
   ```sh
   ~/.pyenv/versions/3.9.18/bin/python -m venv venv
   source venv/bin/activate
   ```
   *Do not use `python3 -m venv venv` unless you are sure `python3` points to pyenv's Python 3.9!*
5. **Upgrade pip and install dependencies:**
   ```sh
   pip install --upgrade pip
   pip install -e ../realsense_python
   pip install -r ../realsense_python/requirements.txt
   pip install mediapipe plotly
   ```
6. **Run the example:**
   ```sh
   cd examples
   python example.py
   ```

### Troubleshooting
- If `mediapipe` fails to install, check your Python version (`python --version`). It must be 3.9 or 3.10.
- If you see errors about missing RealSense modules, make sure you installed your local package with `pip install -e ../realsense_python` from your venv.
- If you get OpenCV or numpy errors, try upgrading pip and reinstalling dependencies.
- If you have an old venv, delete it and recreate it with the correct Python version.

---

This directory contains examples demonstrating the capabilities of Intel RealSense cameras on Apple Silicon Macs.

## Basic Example

For a simple demonstration of depth camera functionality:

```bash
python simple_depth.py
```

## Advanced Example (example.py)

The `example.py` script provides a comprehensive demonstration of the RealSense camera's capabilities including depth visualization, object detection, hand tracking, and 3D point cloud visualization.

### Basic Usage

Run the example with default settings:

```bash
python example.py
```

### Enabling Object Detection

To enable object detection and point cloud visualization:

```bash
python example.py --enable-object-detection
```

This will detect objects in the scene, draw bounding boxes in the infrared view, and create 3D point cloud visualizations of detected objects.

#### Object Detection Options

- Change the detection model:
  ```bash
  python example.py --enable-object-detection --object-model mobilenet-ssd
  ```
  Available models: `mobilenet-ssd` (default), `yolo-tiny`, `mp-objects` (MediaPipe)

- Adjust confidence threshold:
  ```bash
  python example.py --enable-object-detection --object-confidence 0.6
  ```
  Higher values reduce false positives (default: 0.5)

### Hand Tracking Features

The example includes MediaPipe-based hand tracking:

```bash
# Run with hand tracking (enabled by default)
python example.py

# Disable hand tracking
python example.py --no-hand-tracking
```

To use both hand tracking and object detection:

```bash
python example.py --enable-object-detection
```

### 3D Visualization Options

Control what appears in the 3D visualization panel:

```bash
# Only show objects (no hands) in 3D visualization
python example.py --enable-object-detection --no-hands-in-3d

# Disable 3D visualization completely
python example.py --no-3d-viz
```

### Depth Visualization Options

```bash
# Use advanced depth visualization
python example.py --advanced-depth

# Select a different colormap
python example.py --colormap TURBO  # Options: TURBO, JET, PLASMA, VIRIDIS, HOT, RAINBOW

# Disable auto depth range adjustment
python example.py --no-auto-range

# Set custom depth range
python example.py --min-depth 0.3 --max-depth 3.0
```

### Resolution and Performance Options

```bash
# Set camera resolution
python example.py --resolution 848x480  # Options: 1280x720, 848x480, 640x360, 480x270, 424x240

# Set frame rate
python example.py --fps 60  # Options: 5, 15, 30, 60, 90 (availability depends on resolution and USB mode)

# Enable color stream
python example.py --enable-color
```

## MediaPipe Installation

For hand tracking functionality, you need MediaPipe:

```bash
# Best to install in a Python 3.9 or 3.10 environment
pip install mediapipe opencv-python
```

If you encounter MediaPipe installation issues, create a specific Python 3.9 environment:

```bash
python3.9 -m venv venv-py39
source venv-py39/bin/activate
pip install -e .
pip install mediapipe opencv-python numpy
```

## Hardware Requirements

- Intel RealSense camera (tested with D415, D435i, D421, D455)
- Apple Silicon Mac (M1/M2/M3)
- USB 3.1 port recommended for best performance

## Model Files for Object Detection

The first time you run object detection, the script will automatically download the required model files to the `examples/models/` directory.

If you experience issues with automatic downloads, you can manually download:

- MobileNet-SSD: [mobilenet_iter_73000.caffemodel](https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel) and [deploy.prototxt](https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt)
- YOLOv4-Tiny: [yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights), [yolov4-tiny.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg), and [coco.names](https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names)

Place these files in the `examples/models/` directory. 