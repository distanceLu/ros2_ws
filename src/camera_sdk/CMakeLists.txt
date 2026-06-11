cmake_minimum_required(VERSION 3.8)
project(camera_sdk)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(
  include
  include/camera_sdk
)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_executable(camera_node src/camera_node.cpp)
ament_target_dependencies(camera_node rclcpp sensor_msgs)
target_link_libraries(camera_node MVSDK yaml-cpp)

install(TARGETS
  camera_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

set(CAMERA_VENDOR_LIB "${CMAKE_CURRENT_SOURCE_DIR}/lib/libMVSDK.so")
if(EXISTS "${CAMERA_VENDOR_LIB}")
  install(FILES "${CAMERA_VENDOR_LIB}" DESTINATION lib)
else()
  message(WARNING "MVSDK library not found at ${CAMERA_VENDOR_LIB}")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
