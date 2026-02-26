cmake_minimum_required(VERSION 4.1)
project(opencv_demo)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCV REQUIRED)


add_executable(opencv_demo main.cpp)

target_link_libraries(opencv_demo ${OpenCV_LIBS})

message(STATUS "OpenCV version: ${OpenCV_VERSION}")