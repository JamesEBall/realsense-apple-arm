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

# With test dependencies
pip install realsense-applesilicon[test]

# With development tools
pip install realsense-applesilicon[dev]

# With documentation tools
pip install realsense-applesilicon[docs]

# With all optional dependencies
pip install realsense-applesilicon[all]
```

## Dependencies

### Core Dependencies
- Python 3.8+
- librealsense2 (system library)
- numpy>=1.19.0,<2.0.0
- opencv-python>=4.5.0,<5.0.0
- cython>=0.29.0,<1.0.0

### Optional Dependencies

#### Testing (install with `pip install realsense-applesilicon[test]`)
- pytest>=6.0.0,<7.0.0
- pytest-cov>=2.10.0,<3.0.0

#### Development (install with `pip install realsense-applesilicon[dev]`)
- black>=22.0.0 (code formatting)
- isort>=5.0.0 (import sorting)
- flake8>=3.9.0 (code linting)
- mypy>=0.900 (type checking)

#### Documentation (install with `pip install realsense-applesilicon[docs]`)
- sphinx>=4.0.0
- sphinx-rtd-theme>=1.0.0

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

# Install development dependencies
pip install -e ".[dev,test,docs]"
```

### Running tests
```bash
pytest tests/
```

### Code formatting
```bash
# Format code
black .

# Sort imports
isort .

# Run linter
flake8 .
```

## License

MIT License 