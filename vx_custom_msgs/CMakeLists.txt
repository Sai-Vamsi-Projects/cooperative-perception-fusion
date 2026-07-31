cmake_minimum_required(VERSION 3.5)
project(vx_custom_msgs)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrafficSignalState.msg"
  "msg/GoalPoint.msg"
  "msg/FusedObject.msg"
  "msg/FusedObjects.msg"
  "msg/DetectedObject.msg"
  "msg/ObjectDetections.msg"
  "msg/CamLaser.msg"
  "msg/FusionResult.msg"
  DEPENDENCIES geometry_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
