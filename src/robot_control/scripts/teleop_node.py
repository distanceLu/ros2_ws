#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Joy

# from my_package.srv import SpecialSpeedl, SpecialSpeedlResponse
# from my_package.srv import SpeedlStop, SpeedlStopResponse
from robot_control.srv import special_speedl, special_speedlResponse

from std_srvs.srv import Empty, EmptyResponse

class TeleopNode:
    def __init__(self):
        rospy.init_node('teleop_node', anonymous = False)

        # 订阅驱动节点发布的输入监听
        # 从 "spacenav/offset" 接收 3D鼠标 位移增量，作为速度增量控制机械臂
        # 当鼠标复位或一段时间内未监听到数据(设备断开)时，调用机械臂驱动的停止服务
        # 从 "spacenav/rot_offset" 接受 3D鼠标 旋转增量，作为机械臂旋转姿态
        # 从 "/spacenav/joy" 接收 3D鼠标 按键信息，输出日志
        # 订阅robot_driver的服务 speedl_s 和 speed_stop进行速度控制

        # 控制参数
        self.enable_ctl_speed = True    # 6dof控制使能
        self.enable_ctl_welding = False # 焊枪控制使能
        self.scale_lin = 1.0   # slow linear factor
        self.scale_ang = 0  # slow angular factor
        self.max_lin = 100      # m/s
        self.max_ang = 0.01     # rad/s
        self.timeout = 0.5         # seconds
        self.cmd_time = 10        # seconds (int32)

        # 订阅主题及服务
        # self.sub_twist = rospy.Subscriber('/spacenav/twist', Twist, self.cb_rcv_twist, queue_size=10)
        self.sub_offset = rospy.Subscriber('/spacenav/offset', Vector3, self.cb_rcv_offset, queue_size=10)
        #self.sub_rot_offset = rospy.Subscriber('/spacenav/rot_offset', Vector3, self.cb_rcv_rot_offset, queue_size=10)
        #self.sub_joy = rospy.Subscriber('/spacenav/joy', Joy, self.cb_rcv_joy, queue_size = 10)
        
        rospy.wait_for_service('speedl_s')  # SpecialSpeedl
        rospy.wait_for_service('speed_stop')
        self.srv_speedl = rospy.ServiceProxy('speedl_s', special_speedl)
       

        # 设置服务激活和关闭遥操作
        #self.srv_mode_activate = rospy.Service('teleop_activate', Empty, self.teleop_activate)
        #self.srv_mode_deactivate = rospy.Service('teleop_deactivate', Empty, self.teleop_deactivate)

        self.last_twist_time = rospy.Time.now()
        #self.timer = rospy.Timer(rospy.Duration(0.1), self.cb_timer)  # 检测是否设备断开
        self.run_mode = False

        self.rx = 0
        self.ry = 0
        self.rz = 0

        rospy.loginfo("[Teleop] init success.")

    def call_srv_speedl(self, vx, vy, vz, rx, ry, rz):

        self.srv_speedl(vx, vy, vz, rx, ry, rz, int(self.cmd_time))


    def cb_rcv_offset(self, msg):

        # Update timestamp
        self.last_twist_time = rospy.Time.now()
        # rospy.loginfo(f"[Teleop] rcv: offset.x={msg.x}, y={msg.y}, z={msg.z}")
        # Scale and clamp linear velocities
        vx = max(-self.max_lin, min(self.max_lin, msg.y * self.scale_lin))
        vy = max(-self.max_lin, min(self.max_lin, msg.x * self.scale_lin))
        vz = max(-self.max_lin, min(self.max_lin, msg.z * self.scale_lin))
        # # Scale and clamp angular velocities
        # rx = max(-self.max_ang, min(self.max_ang, msg.angular.x * self.scale_ang))
        # ry = max(-self.max_ang, min(self.max_ang, msg.angular.y * self.scale_ang))
        # rz = max(-self.max_ang, min(self.max_ang, msg.angular.z * self.scale_ang))
        # send speed cmd
        self.call_srv_speedl(vx, vy, 0, 0, 0, 0)
        # todo 旋转应该用绝对位置而不是增量

if __name__ == '__main__':
    node = TeleopNode()
    rospy.spin()


