#!/usr/bin/env python3
import rospy
import HaplyHardwareAPI
import threading
from std_msgs.msg import UInt8
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

# 和手柄结合使用，手柄负责起弧收弧，以及开启/关闭定速巡缝
# 按键2切换虚拟焊道功能，以当前坐标建立上下左右4面力触反馈墙，并向外发送手柄位置

class HaplyInverse3Node:
    def __init__(self):
        rospy.init_node('haply_inverse3_node', anonymous = False)
        # 参数
        self.rate_hz = 1000 # 设备位置读取频率
        # self.handle_rate_hz = 50 # 设置操控笔读取频率

        # 设备句柄
        self.inverse3 = None
        self.com_stream = None

        self.mode_teleop = False
        self.mode_teleop_xy = False
        self.mode_free = False
        self.mode_plane = False

        # self.handle_stream = None
        # self.versegrip = None
        # self._pressed_buttons = []

        self.running = False
        # 检测设备
        connected_devices = HaplyHardwareAPI.detect_inverse3s()
        rospy.loginfo('\n[HAPLY] connected_devices: %s', connected_devices)
        if not connected_devices:
            rospy.logerr('[HAPLY] No Haply Inverse3 detected. '
                         'Check USB connection, permissions (dialout), and power.')
            return
        self.com_stream = HaplyHardwareAPI.SerialStream(connected_devices[0])
        self.inverse3 = HaplyHardwareAPI.Inverse3(self.com_stream)

        # 暖机
        rospy.sleep(1.5)

        response_to_wakeup = self.inverse3.device_wakeup_dict()
        
        # print the response to the wakeup command
        rospy.loginfo('\n[HAPLY] wakeup response:')
        for key, val in response_to_wakeup.items():
            rospy.loginfo('%s: %s', key, val)
        rospy.loginfo('[HAPLY] wakeup response end ===========\n')

        # 接收按键消息
        rospy.Subscriber("/haply_btn", UInt8, self.on_haply_btn, queue_size=10)
        # 接受手柄按键消息，操作笔按键不灵时的backup
        self._last_toggle_time = rospy.Time(0)
        self._debounce_sec = 0.5        
        rospy.Subscriber("/spacenav/joy", Joy, self.on_joy_btn, queue_size=10)


        # 发送位姿消息
        self.pub_twist = rospy.Publisher('/spacenav/twist', Twist, queue_size=50)

        # 遥操控制模拟焊道反馈
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self._x_ori = 0.0
        self._y_ori = 0.0
        self._z_ori = 0.0
        self.wall_z_min = -9999.0
        self.wall_z_max = 9999.0
        self.wall_y_min = -9999.0
        self.wall_y_max = 9999.0
        
        # 参数
        self.wall_space = 0.002
        self.force_max = 15.0
        self.wall_stiffness = 800.0    # 刚度        
        self.zeroforce_z_min = 1.4  # 各方向零区设置 2.5
        self.zeroforce_z_max = 0.5  # 1.3
        self.zeroforce_y_min = 1.6
        self.zeroforce_y_max = 1.6
        self.k_factor_x = 1.25
        self.k_factor_y = 1.0
        self.k_factor_z = 1.0
        self.wall_zeroforce_z_min = self.wall_z_min
        self.wall_zeroforce_z_max = self.wall_z_max
        self.wall_zeroforce_y_min = self.wall_y_min
        self.wall_zeroforce_y_max = self.wall_y_max

        # ---- 启动读取线程 ----
        self.running = True
        
        # Inverse3 线程
        self._reader_thread = threading.Thread(
            target=self._loop_read_position, daemon=True
        )
        self._reader_thread.start()


        rospy.on_shutdown(self._on_shutdown)


    def limitF(self, force, max_force):
        """
        将三维力向量按整体幅值限幅到不超过 max_force（单位：N），保留方向不变。
        - force: [fx, fy, fz]
        - max_force: 标量最大力
        """
        mag = math.sqrt(sum(f * f for f in force))
        if mag == 0.0 or mag <= max_force:
            return force
        scale = max_force / mag
        return [f * scale for f in force]

    def _end_effector_force(self, forces = [0, 0, 0]):
        forces = self.limitF(forces, self.force_max)
        position, velocity = self.inverse3.end_effector_force(forces)
        # 打印位置（单位：米）。为了不刷屏，节流到 ~20Hz 显示；采样仍按 self.rate_hz 进行
        # rospy.loginfo_throttle(
        #     0.05,  # 每 0.05s 打印一次（约 20Hz）
        #     "haply_position: [%.6f, %.6f, %.6f] forces: [%.6f, %.6f, %.6f]",
        #     self._x, self._y, self._z, forces[0], forces[1], forces[2]
        # )        
        return position, velocity

    # Inverse3 位置读取线程
    def _loop_read_position(self):
        """
        固定频率读取末端位置并打印到屏幕。
        说明：调用 inverse3.end_effector_force() 不带参数时，仅查询当前位置/速度。
        """
        rate = rospy.Rate(self.rate_hz)
        forces = [0, 0, 0]
        while not rospy.is_shutdown() and self.running:
            try:
                # position, velocity = self.inverse3.end_effector_force()
                position, velocity = self._end_effector_force(forces)
                
                self._x = position[0]
                self._y = position[1]
                self._z = position[2]
                tw = Twist()
                tw.linear.x = 0
                tw.linear.y = 0
                tw.linear.z = 0
                
                if self.mode_teleop == True:
                    forces = self.gen_force_wall()

                    # 发送当前位置消息到/spacenav/twist
                    tw.linear.x = self._x   # x通道直接发送

                    # y通道零区设置
                    if forces[1] > self.zeroforce_y_min:
                        tw.linear.y = self._y - self.wall_zeroforce_y_min
                    if forces[1] < -self.zeroforce_y_max:
                        tw.linear.y = self._y - self.wall_zeroforce_y_max
                    if forces[2] > self.zeroforce_z_min:
                        tw.linear.z = self._z - self.wall_zeroforce_z_min
                    if forces[2] < -self.zeroforce_z_max:
                        tw.linear.z = self._z - self.wall_zeroforce_z_max

                    # # 对于向上方向调整，改成离开接触面，就发送向上消息
                    # if self._z > self.wall_z_min:
                    #     tw.linear.z = self._z - self.wall_z_min

                    # # y方向直接发送
                    # tw.linear.y = self._y

                    # xy方向调换
                    tw.linear.y = self._x - self._x_ori
                    tw.linear.x = self._y

                    tw.linear.x = tw.linear.x * self.k_factor_x
                    tw.linear.y = tw.linear.y * self.k_factor_y
                    tw.linear.z = tw.linear.z * self.k_factor_z

                    self.pub_twist.publish(tw)

                if self.mode_teleop_xy == True:
                    forces = self.gen_force_wall_XY()

                    tw.linear.x = self._y - self._y_ori
                    tw.linear.y = self._x - self._x_ori
                    tw.linear.z = self._z - self._z_ori
                    tw.linear.x = tw.linear.x * self.k_factor_x
                    tw.linear.y = tw.linear.y * self.k_factor_y
                    tw.linear.z = tw.linear.z * self.k_factor_z

                    self.pub_twist.publish(tw)

                if self.mode_free == True:
                    forces = [0, 0, 0]
                    tw.linear.x = self._x
                    tw.linear.y = self._y
                    tw.linear.z = self._z
                    self.pub_twist.publish(tw)

                if not self.mode_teleop and not self.mode_free and not self.mode_teleop_xy:
                    forces = [0, 0, 0]

                if True:
                    rospy.loginfo_throttle(
                        3,  # 每 0.05s 打印一次（约 20Hz）
                        "haply_position: [%.6f, %.6f, %.6f] forces: [%.6f, %.6f, %.6f] twist_msg: [%.6f, %.6f, %.6f]",
                        self._x, self._y, self._z, forces[0], forces[1], forces[2],
                        tw.linear.x, tw.linear.y, tw.linear.z
                    )        

            except Exception as e:
                # 设备临时超时/串口被占用等，做节流告警
                rospy.logwarn_throttle(1.0, "[HAPLY] read failed: %s", str(e))
            rate.sleep()

    def gen_force_wall(self):
        forces = [0, 0, 0]
        if self._z < self.wall_z_min:
            forces[2] = (self.wall_z_min - self._z) * self.wall_stiffness
        if self._z > self.wall_z_max:
            forces[2] = (self.wall_z_max - self._z) * self.wall_stiffness
        # if self._y < self.wall_y_min:
        #     forces[1] = (self.wall_y_min - self._y) * self.wall_stiffness
        # if self._y > self.wall_y_max:
        #     forces[1] = (self.wall_y_max - self._y) * self.wall_stiffness
        return forces


    def gen_force_wall_XY(self):
        forces = [0, 0, 0]
        if self._x < self.wall_x_min:
            forces[0] = (self.wall_x_min - self._x) * self.wall_stiffness
        # if self._x > self.wall_x_max:
        #     forces[0] = (self.wall_x_max - self._x) * self.wall_stiffness
        rospy.logwarn_throttle(1.0, "[HAPLY] wall_XY forces: %.4f %.4f %.4f", forces[0], forces[1], forces[2])
        # return [0, 0, 0]
        return forces

    def mode_teleop_switch(self):
        self.mode_teleop_xy = False
        if not self.mode_teleop:
            self.setSeamWall()
        self.mode_teleop = not self.mode_teleop
        rospy.loginfo('[HAPLY] set mode teleop to %d', self.mode_teleop) 

    def mode_teleop_xy_switch(self):
        self.mode_teleop = False
        if not self.mode_teleop_xy:
            self.setSeamWallXY()
        self.mode_teleop_xy = not self.mode_teleop_xy
        rospy.loginfo('[HAPLY] set mode teleop xy to %d', self.mode_teleop_xy) 


    def on_haply_btn(self, msg:UInt8):
        btn = int(msg.data)
        # rospy.loginfo('[HAPLY] recv btn %d', btn)
        if btn == 2:
            self.mode_teleop_switch()
            # if not self.mode_teleop:
            #     self.setSeamWall()
            # self.mode_teleop = not self.mode_teleop
            # rospy.loginfo('[HAPLY] set mode teleop to %d', self.mode_teleop)

    def on_joy_btn(self, msg: Joy):
        # 手柄3号键按下切换模式
        # if len(msg.buttons) == 12 and msg.buttons[2] == 1:
        #     now = rospy.Time.now()
        #     if (now - self._last_toggle_time).to_sec() < self._debounce_sec:
        #         self._last_toggle_time = now
        #         return   # 接受持续数据认为只触发1次
        #     self._last_toggle_time = now
        #     if not self.mode_teleop:
        #         self.setSeamWall()
        #     self.mode_teleop = not self.mode_teleop
        #     rospy.loginfo('[HAPLY] set mode teleop to %d by joystick', self.mode_teleop)
        if len(msg.buttons) == 12:
            if msg.buttons[2] == 1 or msg.buttons[3] == 1:
                now = rospy.Time.now()
                if (now - self._last_toggle_time).to_sec() < self._debounce_sec:
                    self._last_toggle_time = now
                    return   # 接受持续数据认为只触发1次
                self._last_toggle_time = now
            if msg.buttons[2] == 1:
                self.mode_teleop_switch()
            if msg.buttons[3] == 1:
                self.mode_teleop_xy_switch()

    def setSeamWall(self):
        self._x_ori = self._x
        self._y_ori = self._y
        self._z_ori = self._z

        self.wall_z_min = self._z - self.wall_space / 2.0
        self.wall_z_max = self._z + self.wall_space / 2.0
        self.wall_y_min = self._y - self.wall_space / 2.0
        self.wall_y_max = self._y + self.wall_space / 2.0

        # 计算各方向零区边界位置
        self.wall_zeroforce_z_min = self.wall_z_min - self.zeroforce_z_min / self.wall_stiffness
        self.wall_zeroforce_z_max = self.wall_z_max + self.zeroforce_z_max / self.wall_stiffness
        self.wall_zeroforce_y_min = self.wall_y_min - self.zeroforce_y_min / self.wall_stiffness
        self.wall_zeroforce_y_max = self.wall_y_max + self.zeroforce_y_max / self.wall_stiffness

        self.k_factor_x = 1.25
        self.k_factor_y = 1.0
        self.k_factor_z = 1.0

        rospy.loginfo('[HAPLY] set seam channal to y(%.3f %.3f) z(%.3f %.3f)', 
            self.wall_y_min, self.wall_y_max, self.wall_z_min, self.wall_z_max)      
        

    def setSeamWallXY(self):
        self._x_ori = self._x
        self._y_ori = self._y
        self._z_ori = self._z

        self.wall_x_min = self._x - self.wall_space / 2.0
        self.wall_x_max = self._x + self.wall_space / 2.0

        self.k_factor_x = 1.0
        self.k_factor_y = 1.0
        self.k_factor_z = 0.7

        rospy.loginfo('[HAPLY] setSeamWallXY reset to (%.3f, %.3f, %.3f)', 
                      self._x_ori, self._y_ori, self._z_ori)


    def _on_shutdown(self):
        self.running = False
        try:
            if hasattr(self, '_reader_thread') and self._reader_thread.is_alive():
                self._reader_thread.join(timeout=1.0)
        except Exception:
            pass


if __name__ == '__main__':
    node = HaplyInverse3Node()
    rospy.spin()
