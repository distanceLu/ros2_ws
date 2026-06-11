cmake_minimum_required(VERSION 3.8)
project(robot_control)

# 使用 C++17
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_srvs REQUIRED)

# 生成消息和服务
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/JointPos.msg"
  "msg/JogPos.msg"
  "msg/FreqCfg.msg"
  "msg/SpdSts.msg"
  "srv/Move.srv"
  "srv/Speedl.srv"
  "srv/Speed.srv"
  "srv/SpecialSpeedl.srv"
  "srv/Track.srv"
  "srv/SrvRgbd.srv"
  "srv/SetValue.srv"
  DEPENDENCIES
    geometry_msgs
    std_msgs
    sensor_msgs
)

# 获取类型支持 target
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
rosidl_get_typesupport_target(cpp_typesupport_introspection_target ${PROJECT_NAME} "rosidl_typesupport_introspection_cpp")

# 公共的包含目录
set(INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${rclcpp_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${tf2_INCLUDE_DIRS}
  ${tf2_msgs_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
  ${tf2_geometry_msgs_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${std_srvs_INCLUDE_DIRS}
)

# 设置库路径
set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# robot_control_node
add_executable(robot_control_node src/driver.cpp)
target_include_directories(robot_control_node PRIVATE ${INCLUDE_DIRS})
target_compile_features(robot_control_node PUBLIC c_std_99 cxx_std_17)
target_link_directories(robot_control_node PRIVATE ${LIB_DIR})
target_link_libraries(robot_control_node
  ${rclcpp_LIBRARIES}
  ${tf2_ros_LIBRARIES}
  ${tf2_msgs_LIBRARIES}
  ${tf2_LIBRARIES}
  ${std_srvs_LIBRARIES}
  ${cpp_typesupport_target}
  ${cpp_typesupport_introspection_target}
  DucoCobotAPI
  MegmeetComm
)

# # control_demo_node
# add_executable(control_demo_node src/demo.cpp)
# target_include_directories(control_demo_node PRIVATE ${INCLUDE_DIRS})
# target_compile_features(control_demo_node PUBLIC c_std_99 cxx_std_17)
# target_link_directories(control_demo_node PRIVATE ${LIB_DIR})
# target_link_libraries(control_demo_node
#   ${rclcpp_LIBRARIES}
#   ${cpp_typesupport_target}
#   DucoCobotAPI
#   MegmeetComm
# )

# # control_test_node
# add_executable(control_test_node src/test.cpp)
# target_include_directories(control_test_node PRIVATE ${INCLUDE_DIRS})
# target_compile_features(control_test_node PUBLIC c_std_99 cxx_std_17)
# target_link_directories(control_test_node PRIVATE ${LIB_DIR})
# target_link_libraries(control_test_node
#   ${rclcpp_LIBRARIES}
#   ${cpp_typesupport_target}
#   DucoCobotAPI
#   MegmeetComm
# )

# # demo_test_node
# add_executable(demo_test_node src/demo_test.cpp)
# target_include_directories(demo_test_node PRIVATE ${INCLUDE_DIRS})
# target_compile_features(demo_test_node PUBLIC c_std_99 cxx_std_17)
# target_link_directories(demo_test_node PRIVATE ${LIB_DIR})
# target_link_libraries(demo_test_node
#   ${rclcpp_LIBRARIES}
#   ${cpp_typesupport_target}
#   DucoCobotAPI
#   MegmeetComm
# )

# # test_speedl_node
# add_executable(test_speedl_node src/test_speedl.cpp)
# target_include_directories(test_speedl_node PRIVATE ${INCLUDE_DIRS})
# target_compile_features(test_speedl_node PUBLIC c_std_99 cxx_std_17)
# target_link_directories(test_speedl_node PRIVATE ${LIB_DIR})
# target_link_libraries(test_speedl_node
#   ${rclcpp_LIBRARIES}
#   ${cpp_typesupport_target}
#   DucoCobotAPI
#   MegmeetComm
# )

# 安装 Python 脚本
install(PROGRAMS
  scripts/megmeet.py
  scripts/data_collect_node.py
  scripts/haply_inverse3_node.py
  scripts/haply_handle_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# 安装库文件
install(FILES
  lib/libDucoCobotAPI.so
  lib/libMegmeetComm.so
  DESTINATION lib
)

# 安装可执行文件
install(TARGETS
  robot_control_node
  # control_demo_node
  # control_test_node
  # demo_test_node
  # test_speedl_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
