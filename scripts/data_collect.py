#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Int32MultiArray
from cv_bridge import CvBridge
from robot_control.msg import JogPos
import cv2
import os
import csv
import yaml
from datetime import datetime

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_elapsed_microseconds(target_date):
    """计算从指定日期0点开始到现在的微秒数"""
    current = datetime.now()
    target = datetime.strptime(target_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    delta = current - target
    return int(delta.total_seconds() * 1_000_000)

class DataCollectNode(Node):
    def __init__(self):
        super().__init__('data_collect_node')
        
        # 保存路径
        self.save_dir_root = self.declare_parameter('save_dir_root', '/home/shugen/ros2_ws/data_collect/').value
        ensure_dir(self.save_dir_root)
        self.get_logger().info(f"[DataCollect] Saving data to {self.save_dir_root}")
        
        # YAML配置文件路径
        self.yaml_config_path = self.declare_parameter(
            'yaml_config_path', 
            '/home/shugen/ros2_ws/src/camera_sdk/config/base.yaml'
        ).value
        
        self.run_mode = False
        self.bridge = CvBridge()
        self.save_date = None
        self._img_count = 0
        self._joint_count = 0
        self._tool_count = 0
        
        # 存储从YAML读取的标定参数
        self.calibration_params = {
            'falan_tcp': None,
            'camera_falan': None,
            'other_cameras': []
        }
        
        # 订阅图像话题
        self.sub_image = self.create_subscription(Image, '/image_topic0', self.cb_save_image, 10)
        
        # 订阅关节状态
        self.sub_joint_state = self.create_subscription(
            Float64MultiArray, '/joint_pos', self.cb_save_joint_state, 10
        )
        
        # 订阅工具位姿
        self.sub_tool_pose = self.create_subscription(
            JogPos, '/tool_pos', self.cb_save_tool_pose, 10
        )
        
        # 订阅焊接状态
        self.sub_weld_state = self.create_subscription(
            Int32MultiArray, '/welding_state', self.cb_save_weld_state, 10
        )
        
        # 服务
        self.srv_activate = self.create_service(
            Trigger, 'data_collect_activate', self.data_collect_activate_callback
        )
        self.srv_deactivate = self.create_service(
            Trigger, 'data_collect_deactivate', self.data_collect_deactivate_callback
        )
        
        self.get_logger().info("[DataCollect] init success.")
        self.get_logger().info("  Subscribed to: /image_topic0, /joint_pos, /tool_pos, /welding_state")
    
    def load_calibration_from_yaml(self, camera_sn):
        """从YAML文件读取标定参数"""
        try:
            if not os.path.exists(self.yaml_config_path):
                self.get_logger().warn(f"YAML config file not found: {self.yaml_config_path}")
                return False
            
            with open(self.yaml_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # 读取 flange_tool (T_falan_tcp)
            if 'flange_tool' in config:
                ft = config['flange_tool']
                self.calibration_params['falan_tcp'] = (
                    ft.get('x', 0.0), ft.get('y', 0.0), ft.get('z', 0.0),
                    ft.get('rx', 0.0), ft.get('ry', 0.0), ft.get('rz', 0.0)
                )
                self.get_logger().info(f"Loaded flange_tool: x={ft.get('x', 0.0)}, y={ft.get('y', 0.0)}, z={ft.get('z', 0.0)}")
            else:
                self.get_logger().warn("flange_tool not found in YAML, using default")
                self.calibration_params['falan_tcp'] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            # 读取 camera (T_camera_falan)
            if 'camera' in config:
                cam = config['camera']
                self.calibration_params['camera_falan'] = (
                    cam.get('x', 0.0), cam.get('y', 0.0), cam.get('z', 0.0),
                    cam.get('rx', 0.0), cam.get('ry', 0.0), cam.get('rz', 0.0)
                )
                self.get_logger().info(f"Loaded camera: x={cam.get('x', 0.0)}, y={cam.get('y', 0.0)}, z={cam.get('z', 0.0)}")
            else:
                self.get_logger().warn("camera not found in YAML, using default")
                self.calibration_params['camera_falan'] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            # 可选：读取其他相机的标定参数（如果需要）
            # 这里可以根据 camera_sn 来选择使用哪个相机的参数
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to load YAML config: {e}")
            return False
    
    def save_calibration_file(self, root_dir, camera_sn):
        """保存标定文件 calibration.csv，使用YAML中的参数"""
        try:
            calib_path = os.path.join(self.save_dir_robot_state, "calibration.csv")
            
            # 从YAML加载标定参数
            if not self.load_calibration_from_yaml(camera_sn):
                self.get_logger().warn("Using default calibration values")
                # 使用默认值
                falan_tcp_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                camera_falan_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            else:
                falan_tcp_values = self.calibration_params['falan_tcp']
                camera_falan_values = self.calibration_params['camera_falan']
            
            # 格式化字符串
            falan_tcp_str = f"{falan_tcp_values[0]} {falan_tcp_values[1]} {falan_tcp_values[2]} " \
                           f"{falan_tcp_values[3]} {falan_tcp_values[4]} {falan_tcp_values[5]}"
            
            camera_falan_str = f"{camera_falan_values[0]} {camera_falan_values[1]} {camera_falan_values[2]} " \
                              f"{camera_falan_values[3]} {camera_falan_values[4]} {camera_falan_values[5]}"
            
            lines = [
                f"T_falan_tcp {falan_tcp_str}",
                f"T_camera-{camera_sn}_falan {camera_falan_str}",
                # 其他相机的标定可以保留为0或也从YAML读取
                "T_camera-044092320668_falan 0 0 0 0 0 0",
                "T_camera-044003322233_falan 0 0 0 0 0 0"
            ]
            
            with open(calib_path, 'w') as f:
                f.write("\n".join(lines) + "\n")
            
            self.get_logger().info(f"[DataCollect] Saved calibration.csv with values from YAML")
            self.get_logger().info(f"  T_falan_tcp: {falan_tcp_str}")
            self.get_logger().info(f"  T_camera-{camera_sn}_falan: {camera_falan_str}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save calibration.csv: {e}")
    
    def data_collect_activate_callback(self, request, response):
        if self.run_mode:
            response.success = False
            response.message = "Already active"
            return response
        
        # 创建保存目录
        self.save_date = datetime.now().strftime('%Y-%m-%d')
        timestamp = datetime.now().strftime('%H-%M-%S')
        root_dir = os.path.join(self.save_dir_root, self.save_date, timestamp)
        self.get_logger().info(f"[DataCollect] current save dir {root_dir}")
        ensure_dir(root_dir)
        
        # 相机SN（可从参数获取）
        camera_sn = "044030420162"
        
        # 创建子目录
        self.save_dir_camera = os.path.join(root_dir, f'camera_{camera_sn}')
        self.save_dir_robot_state = os.path.join(root_dir, 'robot_state')
        self.save_dir_welding_state = os.path.join(root_dir, 'welding_state')
        
        for folder in [self.save_dir_camera, self.save_dir_robot_state, self.save_dir_welding_state]:
            ensure_dir(folder)
        
        # 保存标定文件（现在会从YAML读取参数）
        self.save_calibration_file(root_dir, camera_sn)
        
        self.run_mode = True
        self._img_count = 0
        self._joint_count = 0
        self._tool_count = 0
        self.get_logger().info(f"[DataCollect] Data collection activated. Saving to {root_dir}")
        
        response.success = True
        response.message = f"Started saving to {root_dir}"
        return response
    
    def data_collect_deactivate_callback(self, request, response):
        self.get_logger().info(f"[DataCollect] Data collection deactivated.")
        self.get_logger().info(f"  Images saved: {self._img_count}")
        self.get_logger().info(f"  Joint states: {self._joint_count}")
        self.get_logger().info(f"  Tool poses: {self._tool_count}")
        self.run_mode = False
        response.success = True
        response.message = f"Data collection stopped. Saved {self._img_count} images"
        return response
    
    def cb_save_image(self, msg):
        if not self.run_mode:
            return
        
        timestamp = get_elapsed_microseconds(self.save_date)
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        image_path = os.path.join(self.save_dir_camera, f'{timestamp}.jpg')
        cv2.imwrite(image_path, cv_img)
        
        self._img_count += 1
        if self._img_count % 30 == 0:
            self.get_logger().info(f"Saved {self._img_count} images")
    
    def cb_save_joint_state(self, msg):
        """保存关节状态到 joint_state.csv"""
        if not self.run_mode:
            return
        
        timestamp = get_elapsed_microseconds(self.save_date)
        joint_state_path = os.path.join(self.save_dir_robot_state, 'joint_state.csv')
        
        # 获取关节数据
        if hasattr(msg, 'data'):
            joints = msg.data[:6] if len(msg.data) >= 6 else list(msg.data) + [0]*(6-len(msg.data))
        else:
            joints = [0]*6
        
        # 写入CSV
        file_exists = os.path.exists(joint_state_path)
        with open(joint_state_path, 'a') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
            writer.writerow([timestamp] + list(joints))
        
        self._joint_count += 1
        if self._joint_count % 100 == 0:
            self.get_logger().info(f"Saved {self._joint_count} joint states")
    
    def cb_save_tool_pose(self, msg):
        """保存工具位姿到 tool_pose.csv"""
        if not self.run_mode:
            return
        
        timestamp = get_elapsed_microseconds(self.save_date)
        tool_pose_path = os.path.join(self.save_dir_robot_state, 'tool_pose.csv')
        
        # 写入CSV
        file_exists = os.path.exists(tool_pose_path)
        with open(tool_pose_path, 'a') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'x', 'y', 'z', 'rx', 'ry', 'rz'])
            writer.writerow([timestamp, msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz])
        
        self._tool_count += 1
    
    def cb_save_weld_state(self, msg):
        """保存焊接状态到 weld_state.csv"""
        if not self.run_mode:
            return
        
        timestamp = get_elapsed_microseconds(self.save_date)
        weld_state_path = os.path.join(self.save_dir_welding_state, 'weld_state.csv')
        
        # 只在第一次写入（与ROS1一致）
        if os.path.exists(weld_state_path):
            return
        
        if hasattr(msg, 'data') and len(msg.data) >= 2:
            with open(weld_state_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, msg.data[0], msg.data[1]])
            self.get_logger().info(f"Saved welding state: {msg.data[0]}, {msg.data[1]}")

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()