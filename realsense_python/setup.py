from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
import os

def get_ext_modules():
    # Get the directory containing setup.py
    here = os.path.abspath(os.path.dirname(__file__))
    
    # Convert absolute paths to relative
    wrapper_path = os.path.join("src", "realsense", "wrapper.pyx")
    
    ext_modules = [
        Extension(
            "realsense.wrapper",
            [wrapper_path],
            include_dirs=[np.get_include(), "/opt/homebrew/include"],
            library_dirs=["/opt/homebrew/lib"],
            libraries=["realsense2"],
            language="c++",
            extra_compile_args=["-std=c++11"]
        )
    ]
    return cythonize(ext_modules, language_level="3")

setup(
    name="realsense-applesilicon",
    version="0.1.0",
    description="Python wrapper for Intel RealSense cameras on Apple Silicon",
    author="James Ball",
    author_email="James@istarirobotics.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=get_ext_modules(),
    install_requires=[
        "numpy>=1.19.0,<2.0.0",
        "opencv-python>=4.5.0,<5.0.0",
        "cython>=0.29.0,<1.0.0"
    ],
    setup_requires=[
        "cython>=0.29.0",
        "numpy>=1.19.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 