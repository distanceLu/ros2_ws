import os
import time
import random
import math
import cv2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from robot_control.srv import Move
from robot_control.msg import JogPos
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor


class ImageSubscriber:
    """图片订阅器类，用于接收ROS2话题图片"""
    def __init__(self, node, topic_name="/image_topic0"):
        self.node = node
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_received = False
        # 使用可靠的QoS配置
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.subscription = node.create_subscription(
            Image,
            topic_name,
            self.image_callback,
            qos
        )
        
    def image_callback(self, msg):
        """图片回调函数"""
        try:
            # 将ROS图片消息转换为OpenCV格式
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_received = True
        except Exception as e:
            self.node.get_logger().error(f"图片转换失败: {str(e)}")
    
    def get_latest_image(self, timeout_sec=3.0):
        """获取最新的图片，带超时等待"""
        start_time = time.time()
        while not self.image_received and (time.time() - start_time) < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        
        if self.image_received:
            return self.latest_image
        else:
            self.node.get_logger().error(f"等待图片超时 ({timeout_sec} 秒)")
            return None


def get_random_multiplier():
    return random.uniform(0.8, 1.1)


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_swing_capture_ros2')
        
        # 创建服务客户端，使用回调组以支持并发调用
        self.callback_group = ReentrantCallbackGroup()
        self.move_tcp_client = self.create_client(
            Move, 
            '/mov_tcp',
            callback_group=self.callback_group
        )
        
        # 等待服务可用
        while not self.move_tcp_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('等待服务 /mov_tcp 可用...')
        self.get_logger().info('已连接到 /mov_tcp 服务')
    
    def move_to_meter(self, x, y, z, rx, ry, rz, z_toward=1):
        """相对移动函数"""
        if not self.move_tcp_client.service_is_ready():
            self.get_logger().error('服务 /mov_tcp 不可用')
            return False
        
        request = Move.Request()
        request.a = x
        request.b = y
        request.c = z * z_toward
        request.d = rx
        request.e = ry
        request.f = rz
        request.block = True
        request.name = ""
        
        future = self.move_tcp_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.result() is not None:
            return True
        else:
            self.get_logger().error(f'服务调用失败: {str(future.exception())}')
            return False


def write_floats_to_file(filename, float_list):
    if len(float_list) % 6 != 0:
        raise ValueError("float_list的长度必须是6的倍数，否则无法正确格式化")

    with open(filename, 'a') as f:
        for i in range(0, len(float_list), 6):
            line_numbers = float_list[i:i + 6]
            line = ' '.join(f"{num:.6f}" for num in line_numbers)
            f.write(line + '\n')


def main(args=None):
    rclpy.init(args=args)
    
    # 创建主节点
    node = RobotController()
    
    save_dir = "pic"
    pose_path = os.path.join(save_dir, "falan_pose.txt")
    os.makedirs(save_dir, exist_ok=True)

    # 初始化图片订阅器
    node.get_logger().info("等待图片话题 /image_topic0 ...")
    image_subscriber = ImageSubscriber(node, topic_name="/image_topic0")
    
    # 等待话题连接
    time.sleep(1.0)
    
    # 测试是否能收到图片
    test_img = image_subscriber.get_latest_image(timeout_sec=5.0)
    if test_img is None:
        node.get_logger().error("无法从 /image_topic0 接收到图片，请检查相机节点是否正常运行")
        return
    else:
        node.get_logger().info("成功连接到图片话题！")

    # 等待获取当前机器人位姿作为起始中心点
    node.get_logger().info("等待获取初始位姿...")
    
    # 获取初始位姿
    init_msg = None
    init_received = False
    
    def init_callback(msg):
        nonlocal init_msg, init_received
        init_msg = msg
        init_received = True
    
    # 创建临时订阅
    init_sub = node.create_subscription(JogPos, "/tool_pos", init_callback, 10)
    
    start_time = time.time()
    timeout = 5.0
    while not init_received and (time.time() - start_time) < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
    
    # 清理订阅
    node.destroy_subscription(init_sub)
    
    if not init_msg:
        node.get_logger().error("无法获取初始位姿，退出")
        return
    
    # 记录初始位姿（螺旋中心点和初始姿态）
    cx, cy, cz = init_msg.x, init_msg.y, init_msg.z
    crx, cry, crz = init_msg.rx, init_msg.ry, init_msg.rz
    node.get_logger().info(f"初始位姿: x={cx:.4f}, y={cy:.4f}, z={cz:.4f}")

    # 螺旋参数
    R = 0.01                    # 旋转半径 (米)
    total_rounds = 2.0          # 总旋转圈数
    total_steps = 300           # 总步数
    z_increment_total = -0.01   # 总上升高度 (米)
    
    # 姿态变化幅度参数
    rx_amplitude = 0.03         # rx抖动幅度 (弧度)
    ry_amplitude = 0.03         # ry抖动幅度 (弧度)
    rz_amplitude = 0.04         # rz抖动幅度 (弧度)
    
    # 计算每步的角位移和高度增量
    d_theta = 2 * math.pi * total_rounds / total_steps
    d_z = z_increment_total / total_steps

    # 记录上一步的位置（用于计算相对位移）
    prev_x, prev_y, prev_z = cx, cy, cz
    prev_rx, prev_ry, prev_rz = crx, cry, crz

    # 执行螺旋运动
    for step in range(1, total_steps + 1):
        # 计算当前目标绝对位置（绕中心点旋转 + z上升）
        theta = step * d_theta
        target_x = cx + R * math.cos(theta)
        target_y = cy + R * math.sin(theta)
        target_z = cz + step * d_z
        
        # 计算姿态变化
        phase = step * 0.5
        target_rx = crx + rx_amplitude * math.sin(phase)
        target_ry = cry + ry_amplitude * math.cos(phase * 1.3)
        target_rz = crz + rz_amplitude * math.sin(phase * 1.7)
        
        # 计算相对位移
        dx = target_x - prev_x
        dy = target_y - prev_y
        dz = target_z - prev_z
        drx = target_rx - prev_rx
        dry = target_ry - prev_ry
        drz = target_rz - prev_rz
        
        node.get_logger().info(f"Step {step}/{total_steps}:")
        node.get_logger().info(f"  相对移动: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")

        # 发送相对位移命令
        move_success = node.move_to_meter(dx, dy, dz, drx, dry, drz, z_toward=1)
        if not move_success:
            node.get_logger().error(f"Step {step} move failed")
            break

        # 更新上一步位置和姿态
        prev_x, prev_y, prev_z = target_x, target_y, target_z
        prev_rx, prev_ry, prev_rz = target_rx, target_ry, target_rz

        time.sleep(3)  # 等待稳定

        # 从话题获取图片
        img = image_subscriber.get_latest_image(timeout_sec=3.0)
        if img is None:
            node.get_logger().error(f"Step {step} 获取图片失败")
            break

        # 保存图片
        file_path = os.path.join(save_dir, f"{step:03d}.jpg")
        cv2.imwrite(file_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        node.get_logger().info(f"Saved: {file_path}")

        # 记录实际位姿
        tool_msg = None
        tool_received = False
        
        def tool_callback(msg):
            nonlocal tool_msg, tool_received
            tool_msg = msg
            tool_received = True
        
        # 创建临时订阅
        tool_sub = node.create_subscription(JogPos, "/tool_pos", tool_callback, 10)
        
        start_time = time.time()
        timeout = 2.0
        while not tool_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(node, timeout_sec=0.1)
        
        # 清理订阅
        node.destroy_subscription(tool_sub)
        
        if tool_msg:
            current_pose = [tool_msg.x, tool_msg.y, tool_msg.z,
                            tool_msg.rx, tool_msg.ry, tool_msg.rz]
            write_floats_to_file(pose_path, current_pose)
            
            dist_from_center = math.sqrt((tool_msg.x - cx)**2 + (tool_msg.y - cy)**2)
            node.get_logger().info(f"  当前距中心距离: {dist_from_center:.4f} m (理论: {R:.4f} m)")
            node.get_logger().info(f"  当前姿态: rx={tool_msg.rx:.4f}, ry={tool_msg.ry:.4f}, rz={tool_msg.rz:.4f}")
        else:
            node.get_logger().error("获取/tool_pos失败，未写入位姿")

    node.get_logger().info("运动采集完成！")
    
    # 清理资源
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
