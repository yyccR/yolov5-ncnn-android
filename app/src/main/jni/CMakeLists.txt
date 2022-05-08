project(yolov5ncnnandroid)

cmake_minimum_required(VERSION 3.10.2)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20220420-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(yolov5ncnnandroid SHARED yolov5_head_ncnn.cpp)

target_link_libraries(yolov5ncnnandroid
    ncnn
    jnigraphics
)
