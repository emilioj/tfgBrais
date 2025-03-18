cmake_minimum_required(VERSION 3.15)
project(opencv_test)
# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
# Find OpenCV
find_package(OpenCV REQUIRED)
# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
# Source files
file(GLOB SOURCES
    src/*.cpp
    src/*.hpp
)
# Create executable
add_executable(opencv_test ${SOURCES})
# Link OpenCV
target_link_libraries(opencv_test ${OpenCV_LIBS})


add_custom_target(copy_calibration_files ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/data/camera_calibration"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_SOURCE_DIR}/data/camera_calibration"
      "${CMAKE_BINARY_DIR}/data/camera_calibration"
    COMMENT "Copying camera calibration files to build directory"
)
