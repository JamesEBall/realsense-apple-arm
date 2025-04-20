cmake_minimum_required(VERSION 3.10)
project(realsense_test)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV first.")
endif()

# Find RealSense2 package
find_package(realsense2 REQUIRED)
if(NOT realsense2_FOUND)
    message(FATAL_ERROR "RealSense SDK 2.0 not found. Please install librealsense2 first.")
endif()

# Add executable
add_executable(realsense_test realsense_test.cpp)

# Include directories
target_include_directories(realsense_test PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${realsense2_INCLUDE_DIRS}
    /usr/local/include
    /usr/include
    /opt/homebrew/include
)

# Link directories
target_link_directories(realsense_test PRIVATE
    ${OpenCV_LIB_DIRS}
    ${realsense2_LIBRARY_DIRS}
    /usr/local/lib
    /usr/lib
    /opt/homebrew/lib
)

# Link libraries
target_link_libraries(realsense_test PRIVATE
    ${OpenCV_LIBS}
    ${realsense2_LIBRARIES}
    realsense2
    pthread
) 