cmake_minimum_required(VERSION 3.13)
project(color_test)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} -Wall -O3 -march=native")
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Eigen3)

set(ARMOR_DETECT_SRC
        lightbar.cpp
        lightbar.h
        camera.cpp
        camera.h
        enemy.cpp
        enemy.h
        optimize.cpp
        predict.cpp
        predict.h
        )

add_library(ArmorDetector SHARED
        ${ARMOR_DETECT_SRC}
        )

include_directories(${OpenCV_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS})
add_executable(color_test main.cpp
        lightbar.cpp
		lightbar.h
		camera.cpp
		camera.h
		enemy.cpp
		enemy.h
		optimize.cpp
		optimize.h
		predict.h
		predict.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
