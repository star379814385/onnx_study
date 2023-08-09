cmake_minimum_required(VERSION 3.27)
project(YOLOV8_RUN)

# SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")
# set(CMAKE_CXX_FLAGS "-O3")

# set(DEFAULT_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY D:/code/onnx_study/inference_by_cpp/YOLOv8-ONNXRuntime-CPP/output)

set(INCLUDE_PATHS ${PROJECT_SOURCE_DIR}/include)
list(APPEND CMAKE_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include)
file(GLOB LIB_PATHS ${PROJECT_SOURCE_DIR}/lib/*)
file(GLOB SOURCE_PATHS ${PROJECT_SOURCE_DIR}/src/*.cpp)
list(APPEND CMAKE_SOURCE_DIR ${SOURCE_PATHS})
list(APPEND CMAKE_DL_LIBS ${LIB_PATHS})


# main
list(APPEND CMAKE_SOURCE_DIR ${PROJECT_SOURCE_DIR}/main.cpp)


# opencv
set(OpenCV_DIR D:/dependency/opencv/build/x64/vc16/lib)
FIND_PACKAGE(OpenCV REQUIRED)
list(APPEND CMAKE_INCLUDE_PATH ${OpenCV_INCLUDE_DIRS})
list(APPEND CMAKE_DL_LIBS ${OpenCV_LIBS})

# onnxruntime-gpu
set(ORT_ROOT D:/dependency/onnxruntime-win-x64-gpu-1.15.1)
# set(ORT_ROOT D:/dependency/onnxruntime-win-x64-1.15.1)
set(ORT_INCLUDE ${ORT_ROOT}/include)
# set(ORT_LIBS ${ORT_ROOT}/lib/onnxruntime_providers_cuda.dll)
# set(ORT_LIBS ${ORT_ROOT}/lib/onnxruntime.lib)
file(GLOB ORT_LIBS ${ORT_ROOT}/lib/*.lib)
list(APPEND CMAKE_INCLUDE_PATH ${ORT_INCLUDE})
list(APPEND CMAKE_DL_LIBS ${ORT_LIBS})


# show
message(STATUS "INDLUDE:")
# message(STATUS ${CMAKE_INCLUDE_PATH})
FOREACH(M ${CMAKE_INCLUDE_PATH}) 
  message(${M})
ENDFOREACH(M)   
message(STATUS "LIBS:")
# message(STATUS ${CMAKE_DL_LIBS})
FOREACH(M ${CMAKE_DL_LIBS}) 
  message(${M})
ENDFOREACH(M)  


include_directories(${CMAKE_INCLUDE_PATH})

add_executable(main ${CMAKE_SOURCE_DIR})
target_link_libraries(main ${CMAKE_DL_LIBS})

