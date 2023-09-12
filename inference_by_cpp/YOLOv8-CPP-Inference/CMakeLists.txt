cmake_minimum_required(VERSION 3.5)

project(Yolov8CPPInference VERSION 0.1)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY D:/code/onnx_study/inference_by_cpp/YOLOv8-CPP-Inference/output)

# 如果设置为 ON，则在构建过程中自动将当前处理的 CMakeLists.txt 所在的目录添加到包含路径中。
set(CMAKE_INCLUDE_CURRENT_DIR ON)


# CUDA
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
find_package(CUDA 11 REQUIRED)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# !CUDA

# OpenCV
set(OpenCV_DIR D:/dependency/opencv/build)  # add
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

set(PROJECT_SOURCES
    main.cpp
    inference.h
    inference.cpp
)

add_executable(Yolov8CPPInference ${PROJECT_SOURCES})
target_link_libraries(Yolov8CPPInference ${OpenCV_LIBS})
