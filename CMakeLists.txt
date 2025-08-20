cmake_minimum_required(VERSION 3.15)
project(ar_obj_renderer LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# -----------------------------
# Core libraries
# -----------------------------
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)

# -----------------------------
# Optional graphics libraries
# -----------------------------
find_package(GLEW CONFIG QUIET)
find_package(glfw3 CONFIG QUIET)
find_package(glm CONFIG QUIET)

# -----------------------------
# Collect optional libraries
# -----------------------------
set(EXTRA_LIBS "")

if(TARGET GLEW::GLEW)
  list(APPEND EXTRA_LIBS GLEW::GLEW)
elseif(DEFINED GLEW_LIBRARIES)
  list(APPEND EXTRA_LIBS ${GLEW_LIBRARIES})
endif()

if(TARGET glfw)
  list(APPEND EXTRA_LIBS glfw)
elseif(TARGET glfw3::glfw)
  list(APPEND EXTRA_LIBS glfw3::glfw)
endif()

if(TARGET glm::glm)
  list(APPEND EXTRA_LIBS glm::glm)
endif()

# -----------------------------
# Include directories
# -----------------------------
include_directories(${OpenCV_INCLUDE_DIRS})

# -----------------------------
# Source files
# -----------------------------
set(COMMON_SOURCES
    src/obj_renderer.cpp
    src/obj_renderer.h
    src/marker_tracker.cpp
    src/marker_tracker.h
    src/obj_loader.h
    src/image_data.h
)

# -----------------------------
# Data copying
# -----------------------------
add_custom_target(copy_calibration_files ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/data/camera_calibration"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_SOURCE_DIR}/data/camera_calibration"
      "${CMAKE_BINARY_DIR}/data/camera_calibration"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/data/models"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_SOURCE_DIR}/data/models"
      "${CMAKE_BINARY_DIR}/data/models"
    COMMENT "Copying camera calibration files and models to build directory"
)



# -----------------------------
# Executables
# -----------------------------
# Main AR application executable
add_executable(ar_app 
    src/ar_app.cpp
    ${COMMON_SOURCES}
)

# Link libraries
target_link_libraries(ar_app PRIVATE ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${EXTRA_LIBS})

# Make sure data is copied before building the executable
add_dependencies(ar_app copy_calibration_files)
