cmake_minimum_required(VERSION 3.5)

project(Yolov8CPPInference VERSION 0.1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.0/")
find_package(CUDA 11 REQUIRED)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# !CUDA
set(opencv opencv4.7.0cuda)

# OpenCV
set(OpenCV_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/${opencv}/include/opencv4
)

set(OpenCV_LINK_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/${opencv}/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/${opencv}/lib/opencv4/3rdparty
)
set(OpenCV_LIBS
    opencv_world
    m tiff jpeg lzma jasper webp ippiw ippicv png z rt resolv dl
)
# find_package(OpenCV REQUIRED)

# !OpenCV


# ffmpeg
set(FFmpeg_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/ffmpeg/include
)
set(FFmpeg_LINK_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/ffmpeg/lib
)
set(FFmpeg_LIBS
     avcodec   avformat  avutil  avdevice 
)
# !ffmpeg

include_directories(${OpenCV_INCLUDE_DIRS} ${FFmpeg_INCLUDE_DIRS})
set(PROJECT_SOURCES
    test_video.cpp
    inference.cpp
)

add_executable(Yolov8CPPInference ${PROJECT_SOURCES})

target_link_directories(Yolov8CPPInference PRIVATE ${OpenCV_LINK_DIRS} ${FFmpeg_LINK_DIRS})
target_link_libraries(Yolov8CPPInference PRIVATE ${OpenCV_LIBS} ${FFmpeg_LIBS} pthread)



