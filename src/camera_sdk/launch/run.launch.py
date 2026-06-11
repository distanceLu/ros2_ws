from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    cfg_default = PathJoinSubstitution(
        [FindPackageShare('camera_sdk'), 'config', 'config.yaml']
    )

    # 相机配置参数
    cfg_file_arg = DeclareLaunchArgument(
        'cfg_file', default_value=cfg_default, description='camera config yaml path'
    )
    fre_arg = DeclareLaunchArgument(
        'fre', default_value='20', description='publish frequency (Hz)'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false', description='Use simulation time'
    )

    # 相机节点
    camera_node = Node(
        package='camera_sdk',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            'cfg_file': LaunchConfiguration('cfg_file'),
            'fre': LaunchConfiguration('fre'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )

    # 机器人控制节点
    robot_control_node = Node(
        package='robot_control',
        executable='robot_control_node',
        name='robot_control_node',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )

    # 数据采集脚本（直接使用绝对路径）
    data_collect_script = ExecuteProcess(
        cmd=['python3', '/home/shugen/ros2_ws/scripts/data_collect.py'],
        output='screen',
    )

    return LaunchDescription([
        cfg_file_arg, 
        fre_arg,
        use_sim_time_arg,
        camera_node,
        robot_control_node,
        data_collect_script,
    ])