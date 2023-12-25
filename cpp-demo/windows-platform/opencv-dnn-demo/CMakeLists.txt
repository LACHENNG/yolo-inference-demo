cmake_minimum_required(VERSION 3.5)
set(CMAKE_BUILD_TYPE Debug)
set(PROJECT_NAME demo)
project(${PROJECT_NAME}  VERSION 0.1)

# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)


# -------------- OpenCV  ------------------#
find_package(OpenCV 4.8.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# -------------- Compile CUDA for FP16 inference if needed  ------------------#
option(USE_CUDA "Enable CUDA support" OFF)
if (NOT APPLE AND USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
else ()
    set(USE_CUDA OFF)
endif ()

set(PROJECT_SOURCES
        test_video.cpp
        inference.h
        inference.cpp
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
if (WIN32)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (LINUX)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS}) 
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (APPLE)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif ()


#copy onnx file and classes.txt file to the same folder of the executable file
configure_file(data/yolov8s_hand256x320.onnx ${CMAKE_CURRENT_BINARY_DIR}/model.onnx COPYONLY)

