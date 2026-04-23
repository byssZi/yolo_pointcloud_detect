cmake_minimum_required(VERSION 3.0.2)
project(yolo_pointcloud_detect)

# ================= 基础配置 =================
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

# ================= ROS =================
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  pcl_ros
  pcl_conversions
  message_filters
  visualization_msgs
)

catkin_package()

# ================= 依赖 =================
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# ================= TensorRT =================
set(TENSORRT_ROOT /home/cf206-2/czj/TensorRT-10.1.0.27)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_ROOT}/include
)

link_directories(
  ${TENSORRT_ROOT}/lib
)

# ================= CUDA 设置（关键修复点） =================
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")

# ================= YOLO 推理库 =================
file(GLOB YOLO_SRCS
  src/yolo_infer/*.cpp
  src/yolo_infer/*.cu
)

cuda_add_library(detect3dlib SHARED ${YOLO_SRCS})

target_link_libraries(detect3dlib
  nvinfer
  nvonnxparser
  nvinfer_plugin
  ${OpenCV_LIBS}
)

# ================= 主节点 =================
add_executable(yolo_pointcloud_detect_node
  src/yolo_pointcloud_detect_node.cpp
)

target_link_libraries(yolo_pointcloud_detect_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  detect3dlib
)
