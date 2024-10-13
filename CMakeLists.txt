cmake_minimum_required(VERSION 3.10)
project(MyCppProject)

set(CMAKE_CXX_STANDARD 14)

# Include directories for source files
include_directories(${PROJECT_SOURCE_DIR}/inference)

# Set the directory for ONNX Runtime headers
set(ONNX_RUNTIME_INCLUDE_DIR "/usr/local/onnxruntime-linux-x64-1.19.2/include")
set(ONNX_RUNTIME_LIB_DIR "/usr/local/onnxruntime-linux-x64-1.19.2/lib")

# Find ONNX Runtime library
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNX_RUNTIME_LIB_DIR})

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found")
endif()
include_directories(${ONNX_RUNTIME_INCLUDE_DIR})

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV library not found")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# Add source files to the project
file(GLOB INFERENCE_SRC "${PROJECT_SOURCE_DIR}/inference/*.cpp")

# Create executable
add_executable(my_program ${INFERENCE_SRC})

# Link libraries
target_link_libraries(my_program ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})

# Thêm tệp Python (nếu có cần chạy)
add_custom_target(run_python_scripts ALL
    COMMAND ${CMAKE_COMMAND} -E env python3 /home/danz/NHAP/onnx/export_onnx.py
    COMMAND ${CMAKE_COMMAND} -E env python3 /home/danz/NHAP/training/export_pth.py
)
# Đảm bảo rằng tệp Python chạy sau khi biên dịch chương trình C++
add_dependencies(my_program run_python_scripts)



