# realsense-applesilicon

Python wrapper for Intel RealSense cameras on Apple Silicon

## Installation

### System Requirements

- macOS running on Apple Silicon (M1/M2)
- Homebrew package manager
- Python 3.8 or higher

### Installing System Dependencies

```bash
# Install librealsense2 (required)
brew install librealsense2
```

### Installing the Python Package

```bash
# Basic installation
pip install realsense-applesilicon

```

## Dependencies

### Core Dependencies

- Python 3.8+
- librealsense
- numpy>=1.19.0,<2.0.0
- opencv-python>=4.5.0,<5.0.0
- cython>=0.29.0,<1.0.0

## Usage

```python
from realsense.wrapper import PyRealSense

# Initialize the camera
rs = PyRealSense(width=640, height=480, framerate=30)

# Start the camera
rs.start()

try:
    # Get frames
    frames = rs.get_frames()
    depth_frame = frames.get('depth')
    color_frame = frames.get('color')
    ir_frame = frames.get('infrared')
  
    # Process frames...
  
finally:
    # Stop the camera
    rs.stop()
```

## Development

### Setting up development environment

```bash
# Clone the repository
git clone https://github.com/yourusername/realsense-applesilicon.git
cd realsense-applesilicon

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate
```

### Running tests

```bash
pytest tests/
```

### License

MIT License
