cmake_minimum_required(VERSION 3.15)
project(opencv_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES
    src/*.cpp
    src/*.hpp
)

# Enable automatic symbol export
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(opencv_test SHARED ${SOURCES})
target_link_libraries(opencv_test ${OpenCV_LIBS})
set_target_properties(opencv_test PROPERTIES DEBUG_POSTFIX "_d")

install(TARGETS opencv_test
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

file(GLOB HEADERS src/*.h src/*.hpp)
install(FILES ${HEADERS} DESTINATION include/opencv_test)
