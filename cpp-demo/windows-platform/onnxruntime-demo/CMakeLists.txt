cmake_minimum_required(VERSION 3.5)

set(CMAKE_BUILD_TYPE Debug) # Debug or Release
project(Yolov8Onnx VERSION 0.1)


set(CMAKE_INCLUDE_CURRENT_DIR ON)

# -------------- Compile CUDA for FP16 inference if needed  ------------------#
option(USE_CUDA "Enable CUDA support" OFF)
if (NOT APPLE AND USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
else ()
    set(USE_CUDA OFF)
endif ()


# -------------- OpenCV  ------------------#
# Any OpenCV version that support basic  cv::dnn::blobFromImages is ok
# find_package(OpenCV 4.8.1 REQUIRED)

set(OpenCV_VERSION 481)
set(OpenCV_BUILD_TYPE ${CMAKE_BUILD_TYPE}) 

if(OpenCV_BUILD_TYPE STREQUAL "Debug")
    set(OpenCV_LIB_PREFIX "d")
else()
    set(OpenCV_LIB_PREFIX "")
endif()


set(OpenCV_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/opencv${OpenCV_VERSION}${OpenCV_BUILD_TYPE}/include/
)

set(OpenCV_LINK_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/opencv${OpenCV_VERSION}${OpenCV_BUILD_TYPE}/x64/vc16/lib
)
set(OpenCV_LIBS
    opencv_world${OpenCV_VERSION}${OpenCV_LIB_PREFIX}
)


# -------------- ONNXRUNTIME  ------------------#
set(ONNXRUNTIME_VERSION 1.16.3)

if (WIN32)
    if (USE_CUDA)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third-party/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}")
    else ()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third-party/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}")
    endif ()
elseif (LINUX)
    if (USE_CUDA)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third-party/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}")
    else ()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third-party/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
    endif ()
elseif (APPLE)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third-party/onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}")
    # Apple X64 binary
    # set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-x64-${ONNXRUNTIME_VERSION}")
    # Apple Universal binary
    # set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}")
endif ()

include_directories(${PROJECT_NAME} ${ONNXRUNTIME_ROOT}/include ${OpenCV_INCLUDE_DIRS})


set(PROJECT_SOURCES
    usage_demo.cpp
    inference.cpp
    utils.cpp
)

add_executable(Yolov8Onnx ${PROJECT_SOURCES})

# ------------------ Add include and link directories to target ----------------------------#
# target_include_directories(Yolov8Onnx PRIVATE ${ONNX_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_directories(Yolov8Onnx PRIVATE ${ONNX_LINK_DIRS} ${OpenCV_LINK_DIRS})
if (WIN32)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (LINUX)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (APPLE)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)
endif ()

# For windows system, copy onnxruntime.dll and opencv_world481.dll to the same folder of the executable file to prevent strange error
if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/third-party/opencv${OpenCV_VERSION}${OpenCV_BUILD_TYPE}/x64/vc16/bin/opencv_world481${OpenCV_LIB_PREFIX}.dll" 
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif ()


#copy onnx file and classes.txt file to the same folder of the executable file
configure_file(data/yolov8s_handposeDemo480x640.onnx ${CMAKE_CURRENT_BINARY_DIR}/model.onnx COPYONLY)
configure_file(data/classes.txt ${CMAKE_CURRENT_BINARY_DIR}/classes.txt COPYONLY)
