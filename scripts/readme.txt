当前数据采集已迁移到新 workspace：

  /home/shugen/yanjie/ros2_ws

一键启动：

  /home/shugen/yanjie/ros2_ws/scripts/collect_data.sh

进入 tmux 后切到 session 窗口，输入：

  r
  连续几轮? 10

每轮流程：

  1. 自动回初始位
  2. 自动开始采集
  3. 遥操完成一条轨迹
  4. 在提示处按 Enter 停止本轮

常用命令：

  /home/shugen/yanjie/ros2_ws/scripts/collect_data.sh attach
  /home/shugen/yanjie/ros2_ws/scripts/collect_data.sh kill

数据保存：

  /home/shugen/yanjie/ros2_ws/data_collect/YYYY-MM-DD/HH-MM-SS/

不要再运行 ~/ros2_ws/scripts/training_collect.sh；那是旧 workspace。
