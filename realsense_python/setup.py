from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "realsense.wrapper",
        ["src/realsense/wrapper.pyx"],
        include_dirs=[np.get_include(), "/opt/homebrew/include"],
        library_dirs=["/opt/homebrew/lib"],
        libraries=["realsense2"],
        language="c++",
        extra_compile_args=["-std=c++11"]
    )
]

setup(
    name="realsense",
    version="0.1.0",
    description="Python wrapper for Intel RealSense cameras on Apple Silicon",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(ext_modules),
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "cython>=0.29.0"
    ],
    setup_requires=[
        "cython>=0.29.0",
        "numpy>=1.19.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 