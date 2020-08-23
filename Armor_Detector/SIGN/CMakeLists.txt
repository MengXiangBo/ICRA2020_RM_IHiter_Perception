cmake_minimum_required(VERSION 3.13)
project(SIGN)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SIGN main.cpp)
target_link_libraries(SIGN ${OpenCV_LIBS})