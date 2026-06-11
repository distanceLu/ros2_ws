import os
import sys
import ctypes
import math
import time
import random
import shutil
from pathlib import Path


def _prepend_env_path(var_name, value):
    value_str = str(value)
    current = os.environ.get(var_name, "")
    items = [item for item in current.split(os.pathsep) if item]
    if value_str in items:
        return
    os.environ[var_name] = value_str if not current else value_str + os.pathsep + current


def _bootstrap_local_python_paths():
    script_path = Path(__file__).resolve()
    workspace_root = script_path.parent.parent
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"

    candidate_site_packages = [
        workspace_root / "venv" / "lib" / pyver / "site-packages",
        workspace_root / "install" / "robot_control" / "lib" / pyver / "site-packages",
    ]

    install_root = workspace_root / "install"
    candidate_library_dirs = []
    if install_root.is_dir():
        for child in sorted(install_root.iterdir()):
            lib_dir = child / "lib"
            if lib_dir.is_dir():
                candidate_library_dirs.append(lib_dir)
            candidate = child / "lib" / pyver / "site-packages"
            if candidate.is_dir():
                candidate_site_packages.append(candidate)

    seen = set()
    for candidate in candidate_site_packages:
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        resolved_str = str(resolved)
        if resolved_str not in sys.path:
            sys.path.insert(0, resolved_str)
        _prepend_env_path("PYTHONPATH", resolved)
        _prepend_env_path("LD_LIBRARY_PATH", resolved)

    seen_libs = set()
    for lib_dir in candidate_library_dirs:
        resolved = lib_dir.resolve()
        if resolved in seen_libs:
            continue
        seen_libs.add(resolved)
        _prepend_env_path("LD_LIBRARY_PATH", resolved)
        _prepend_env_path("LIBRARY_PATH", resolved)


def _preload_local_rosidl_libraries():
    script_path = Path(__file__).resolve()
    workspace_root = script_path.parent.parent
    install_root = workspace_root / "install"
    if not install_root.is_dir():
        return

    for child in sorted(install_root.iterdir()):
        lib_dir = child / "lib"
        if not lib_dir.is_dir():
            continue
        for library_path in sorted(lib_dir.glob("lib*.so")):
            try:
                ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue


_bootstrap_local_python_paths()
_preload_local_rosidl_libraries()

import PyRVC as RVC
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from robot_control.srv import Move
from robot_control.msg import JogPos


# 随机移动参数范围
MAX_X_MOVE_MM = 100    # X轴最大移动 100mm
MAX_Y_MOVE_MM = 100    # Y轴最大移动 100mm
MAX_Z_MOVE_MM = 50     # Z轴最大移动 50mm
MAX_RX_MOVE_RAD = 0.2  # RX最大移动 0.2弧度
MAX_RY_MOVE_RAD = 0.2  # RY最大移动 0.2弧度
MAX_RZ_MOVE_RAD = 0.2  # RZ最大移动 0.2弧度


class RobotController(Node):
    """机器人控制节点，使用 /mov_tcp 服务"""
    
    def __init__(self):
        super().__init__('robot_calibration_node')
        
        # 创建服务客户端，使用回调组以支持并发调用
        self.callback_group = ReentrantCallbackGroup()
        self.move_tcp_client = self.create_client(
            Move, 
            '/mov_tcp',
            callback_group=self.callback_group
        )
        # 绝对运动服务（返回Home点使用）
        self.move_jog_client = self.create_client(
            Move,
            '/mov_jog',
            callback_group=self.callback_group
        )
        
        # 等待服务可用
        while not self.move_tcp_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('等待服务 /mov_tcp 可用...')
        self.get_logger().info('已连接到 /mov_tcp 服务')

        while not self.move_jog_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('等待服务 /mov_jog 可用...')
        # 创建位姿订阅者
        self.tool_pos_sub = self.create_subscription(
            JogPos,
            '/tool_pos',
            self.tool_pos_callback,
            10
        )
        self.current_tool_pos = None
        self.tool_pos_received = False
    
    def tool_pos_callback(self, msg):
        """位姿回调函数"""
        self.current_tool_pos = msg
        self.tool_pos_received = True
    
    def get_current_pose(self, timeout_sec=2.0):
        """获取当前机械臂位姿"""
        self.tool_pos_received = False  # 重置标志，等待新消息
        start_time = time.time()
        while not self.tool_pos_received and (time.time() - start_time) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if self.tool_pos_received:
            pose = [
                self.current_tool_pos.x,
                self.current_tool_pos.y,
                self.current_tool_pos.z,
                self.current_tool_pos.rx,
                self.current_tool_pos.ry,
                self.current_tool_pos.rz
            ]
            return pose
        else:
            self.get_logger().error(f'获取位姿超时 ({timeout_sec} 秒)')
            return None
    
    def move_relative(self, dx, dy, dz, drx, dry, drz, z_toward=1):
        """
        相对移动函数（增量移动）
        dx, dy, dz: 相对移动距离（米）
        drx, dry, drz: 相对旋转角度（弧度）
        z_toward: Z轴方向，1向上，-1向下
        """
        if not self.move_tcp_client.service_is_ready():
            self.get_logger().error('服务 /mov_tcp 不可用')
            return False
        
        request = Move.Request()
        request.a = float(dx)
        request.b = float(dy)
        request.c = float(dz * z_toward)
        request.d = float(drx)
        request.e = float(dry)
        request.f = float(drz)
        request.block = True
        request.name = ""
        
        self.get_logger().info(f'发送相对移动命令: dx={dx*1000:+.1f}mm, dy={dy*1000:+.1f}mm, dz={dz*1000:+.1f}mm')
        
        future = self.move_tcp_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        
        if future.result() is not None:
            self.get_logger().info('相对移动命令执行成功')
            return True
        else:
            self.get_logger().error(f'服务调用失败: {str(future.exception())}')
            return False
    
    def move_absolute(self,
                  target_x,
                  target_y,
                  target_z,
                  target_rx,
                  target_ry,
                  target_rz):
        """
        使用 /mov_jog 进行绝对位姿运动
        """

        if not self.move_jog_client.service_is_ready():
            self.get_logger().error('/mov_jog 服务不可用')
            return False

        request = Move.Request()

        request.a = float(target_x)
        request.b = float(target_y)
        request.c = float(target_z)

        request.d = float(target_rx)
        request.e = float(target_ry)
        request.f = float(target_rz)

        request.block = True
        request.name = ""

        self.get_logger().info(
            f'绝对运动到Home点: '
            f'x={target_x:.6f}, '
            f'y={target_y:.6f}, '
            f'z={target_z:.6f}, '
            f'rx={target_rx:.6f}, '
            f'ry={target_ry:.6f}, '
            f'rz={target_rz:.6f}'
        )

        future = self.move_jog_client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=20.0
        )

        if future.result() is not None:
            self.get_logger().info('绝对运动成功')
            return True

        self.get_logger().error(
            f'绝对运动失败: {future.exception()}'
        )

        return False


def get_random_multiplier():
    return random.uniform(0.8, 1.2)


def get_random_move():
    """
    生成随机移动变量（相对/增量移动）
    返回: [dx, dy, dz, drx, dry, drz] 单位：米（位置），弧度（姿态）
    """
    dx = random.uniform(-MAX_X_MOVE_MM, MAX_X_MOVE_MM) / 1000.0
    dy = random.uniform(-MAX_Y_MOVE_MM, MAX_Y_MOVE_MM) / 1000.0
    dz = random.uniform(-MAX_Z_MOVE_MM, MAX_Z_MOVE_MM) / 1000.0
    
    drx = random.uniform(-MAX_RX_MOVE_RAD, MAX_RX_MOVE_RAD) * get_random_multiplier()
    dry = random.uniform(-MAX_RY_MOVE_RAD, MAX_RY_MOVE_RAD) * get_random_multiplier()
    drz = random.uniform(-MAX_RZ_MOVE_RAD, MAX_RZ_MOVE_RAD) * get_random_multiplier()
    
    return [dx, dy, dz, drx, dry, drz]


def adjust_brightness(image, target_brightness=80):
    """调整灰度图像亮度到目标值"""
    if not isinstance(image, np.ndarray):
        raise TypeError("输入图像应为numpy数组格式")
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    current_brightness = np.mean(gray)
    
    if current_brightness >= target_brightness - 30:
        return gray
    
    ratio = target_brightness / current_brightness
    adjusted_image = cv2.convertScaleAbs(gray, alpha=ratio, beta=0)
    
    if current_brightness != np.mean(adjusted_image):
        print("   图片暗，已进行增亮处理")
    
    return adjusted_image


def detect_concentric_ellipses_minimal_opt(img_bgr):
    """检测同心圆标记物"""
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_circle = None
    max_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity < 0.5:
            continue
        
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        # 兼容不同版本的 NumPy
        try:
            box = np.int32(box)
        except AttributeError:
            box = np.int0(box)
        
        border_area = cv2.contourArea(box)
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            
            if border_area > max_area:
                max_area = border_area
                best_circle = {
                    "center": (cX, cY),
                    "outer_area": border_area,
                    "contour": contour,
                    "box": box.tolist() if hasattr(box, 'tolist') else box
                }
    
    return best_circle


def start_line_scan_guidance(x):
    """启动固定线扫模式作为视觉指引"""
    cap_opt = RVC.X2_CaptureOptions()
    cap_opt.capture_mode = RVC.CaptureMode_FixedLineScan
    cap_opt.line_scanner_exposure_time_us = 300
    cap_opt.line_scanner_min_distance = 400
    cap_opt.line_scanner_max_distance = 800
    cap_opt.correspond2d = False
    
    ret2 = x.StartFixedLineScan(cap_opt)
    if not ret2:
        print("启动固定线扫模式失败!")
        return False
    return True


def stop_line_scan_guidance(x):
    """停止固定线扫模式"""
    x.StopFixedLineScan()


def capture_in_ultra_mode(x):
    """在Ultra模式下捕获数据"""
    cap_opt = RVC.X2_CaptureOptions()
    cap_opt.capture_mode = RVC.CaptureMode_Ultra
    cap_opt.line_scanner_exposure_time_us = 300
    cap_opt.line_scanner_min_distance = 400
    cap_opt.line_scanner_max_distance = 1000
    
    ret = x.Capture(cap_opt)
    return ret


def write_floats_to_file(filename, float_list, mode='a'):
    """将浮点数列表写入文件，每行6个数字"""
    if len(float_list) % 6 != 0:
        raise ValueError("float_list的长度必须是6的倍数")
    
    with open(filename, mode) as f:
        for i in range(0, len(float_list), 6):
            line_numbers = float_list[i:i + 6]
            line = ' '.join(f"{num:.6f}" for num in line_numbers)
            f.write(line + '\n')


def prompt_input(prompt, default='q'):
    """在非交互环境下返回默认值，避免直接抛 EOFError。"""
    try:
        return input(prompt)
    except EOFError:
        print("\n检测到非交互输入，程序将退出。")
        return default


def main(args=None):
    node = None
    x = None
    rvc_initialized = False

    rclpy.init(args=args)

    try:
        node = RobotController()

        print("=" * 60)
        print("3D相机标定采集程序 - 随机移动采集模式")
        print("=" * 60)
        print("请保证焊头上下方各有至少10cm活动空间")
        print("")
        print("移动限制（每次随机移动范围）:")
        print(f"  X轴: ±{MAX_X_MOVE_MM}mm")
        print(f"  Y轴: ±{MAX_Y_MOVE_MM}mm")
        print(f"  Z轴: ±{MAX_Z_MOVE_MM}mm")
        print(f"  RX/RY/RZ: ±{MAX_RX_MOVE_RAD}rad (约±{MAX_RX_MOVE_RAD*180/math.pi:.0f}°)")
        print("=" * 60)

        RVC.SystemInit()
        rvc_initialized = True
        _ret, devices = RVC.SystemListDevices(RVC.SystemListDeviceTypeEnum.All)
        print("RVC X Camera devices number:", len(devices))

        if len(devices) == 0:
            print("Can not find any RVC X Camera!")
            return

        x = RVC.X2.Create(devices[0])
        if not x.IsValid():
            print("RVC X Camera is not valid!")
            return

        _ret1 = x.Open()
        if not x.IsOpen():
            print("RVC X Camera is occupied!")
            return

        calib_dir = 'data'
        if os.path.exists(calib_dir):
            shutil.rmtree(calib_dir)
        os.makedirs(calib_dir)

        start_line_scan_guidance(x)

        while True:
            a = prompt_input('\n请按键后回车 (n：开始采集数据 q：退出程序):\n', default='q')

            if a == 'n':
                stop_line_scan_guidance(x)

                print("\n请将同心圆标记物放到相机视野内...")
                prompt_input("准备好后按回车继续...", default='')

                print("\n正在获取初始位姿...")
                home_pose = node.get_current_pose(timeout_sec=5.0)
                if home_pose is None:
                    print("无法获取初始位姿")
                    break

                print(f"\n初始位姿（基准点）:")
                print(f"  x={home_pose[0]:.6f}, y={home_pose[1]:.6f}, z={home_pose[2]:.6f}")
                print(f"  rx={home_pose[3]:.6f}, ry={home_pose[4]:.6f}, rz={home_pose[5]:.6f}")

                pose_file = calib_dir + '/pose.txt'
                with open(pose_file, 'w') as f:
                    f.write("# x(m) y(m) z(m) rx(rad) ry(rad) rz(rad)\n")

                print("\n开始随机移动采集...")
                print("=" * 60)

                success_count = 0
                all_offsets = []

                for step in range(14):
                    print(f"\n第 {step + 1}/14 次采集")
                    print("-" * 40)

                    dx, dy, dz, drx, dry, drz = get_random_move()
                    all_offsets.append([dx, dy, dz, drx, dry, drz])

                    print(f"   随机移动量:")
                    print(f"     X: {dx*1000:+.1f}mm")
                    print(f"     Y: {dy*1000:+.1f}mm")
                    print(f"     Z: {dz*1000:+.1f}mm")
                    print(f"     RX: {drx:+.3f}rad ({drx*180/math.pi:+.1f}°)")
                    print(f"     RY: {dry:+.3f}rad ({dry*180/math.pi:+.1f}°)")
                    print(f"     RZ: {drz:+.3f}rad ({drz*180/math.pi:+.1f}°)")

                    print("   正在移动到随机位置...")
                    move_success = node.move_relative(dx, dy, dz, drx, dry, drz, z_toward=1)

                    if not move_success:
                        print(f"   ⚠ 第 {step + 1} 次移动失败，跳过本次采集")
                        time.sleep(2)
                        continue

                    print("   等待机械臂稳定...")
                    time.sleep(1.5)

                    print("   开始采集图像和点云...")
                    ret_cap = capture_in_ultra_mode(x)

                    if ret_cap:
                        img = np.array(x.GetImage(RVC.CameraID_Left))
                        resize_img = cv2.resize(img, (640, 480))
                        cv2.imshow('camera', resize_img)
                        cv2.waitKey(300)

                        img_adjusted = adjust_brightness(np.array(img, copy=False))
                        img_filename = f'{calib_dir}/{step + 1:02d}.png'
                        cv2.imwrite(img_filename, img_adjusted)
                        print(f"   图像已保存: {img_filename}")

                        ply_filename = f'{calib_dir}/{step + 1:02d}.ply'
                        if x.GetPointMap().Save(ply_filename, RVC.PointMapUnitEnum.Meter):
                            print(f"   点云已保存: {ply_filename}")
                        else:
                            print(f"   ⚠ 点云保存失败！")

                        print("   正在记录当前机械臂位姿...")
                        current_pose = node.get_current_pose(timeout_sec=3.0)
                        if current_pose:
                            write_floats_to_file(calib_dir + '/pose.txt', current_pose, mode='a')
                            print(f"   位姿已记录: x={current_pose[0]:.6f}, y={current_pose[1]:.6f}, z={current_pose[2]:.6f}")
                        else:
                            print(f"   ⚠ 无法获取当前位姿")

                        circle_info = detect_concentric_ellipses_minimal_opt(img)
                        if circle_info:
                            center = circle_info["center"]
                            area = circle_info["outer_area"]
                            print(f"   检测到标记物: 中心=({center[0]:.1f}, {center[1]:.1f}), 面积={area:.0f}")

                        success_count += 1
                        print(f"   ✓ 第 {step + 1} 次采集完成")
                    else:
                        print(f"   ✗ 第 {step + 1} 次采集失败！")

                    print("   正在返回初始位置...")
                    return_success = node.move_absolute(
                        home_pose[0], home_pose[1], home_pose[2],
                        home_pose[3], home_pose[4], home_pose[5]
                    )

                    if not return_success:
                        print(f"   ⚠ 返回初始位置失败！")
                        print("   尝试重新获取位姿并重试...")
                        time.sleep(2)
                        return_success = node.move_absolute(
                            home_pose[0], home_pose[1], home_pose[2],
                            home_pose[3], home_pose[4], home_pose[5]
                        )
                        if not return_success:
                            print(f"   ✗ 返回初始位置仍然失败，终止采集")
                            break

                    time.sleep(1.0)
                    print(f"   ✓ 已返回初始位置")
                    time.sleep(0.5)

                offset_file = f'{calib_dir}/offsets.txt'
                with open(offset_file, 'w') as f:
                    f.write("# dx(m) dy(m) dz(m) drx(rad) dry(rad) drz(rad)\n")
                    for offset in all_offsets:
                        f.write(' '.join(f"{val:.6f}" for val in offset) + '\n')
                print(f"\n所有偏移量已保存到: {offset_file}")

                print("\n" + "=" * 60)
                print(f"数据采集完成！成功采集 {success_count}/14 次")
                print(f"采集数据保存在: {calib_dir}/")
                print(f"   - 图像: {calib_dir}/01.png ~ {calib_dir}/14.png")
                print(f"   - 点云: {calib_dir}/01.ply ~ {calib_dir}/14.ply")
                print(f"   - 位姿: {calib_dir}/pose.txt")
                print(f"   - 偏移量: {calib_dir}/offsets.txt")
                print("=" * 60)

                cv2.destroyAllWindows()
                break

            if a == 'q':
                break
    finally:
        cv2.destroyAllWindows()

        if x is not None:
            try:
                if x.IsOpen():
                    x.Close()
            except Exception:
                pass
            try:
                RVC.X2.Destroy(x)
            except Exception:
                pass

        if rvc_initialized:
            try:
                RVC.SystemShutdown()
            except Exception:
                pass

        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        print("程序已退出")


if __name__ == "__main__":
    main()
