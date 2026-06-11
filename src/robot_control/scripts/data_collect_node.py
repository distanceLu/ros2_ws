#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import io
import os
from typing import Dict, Any, List
import csv
import datetime
import yaml
from cv_bridge import CvBridge
# msg
from sensor_msgs.msg import Image as Image_msg
from sensor_msgs.msg import PointCloud2 as PointCloud2_msg

from robot_control.msg import joint_pos, jog_pos, freq_cfg ,spd_sts
from robot_control.srv import srv_rgbd,srv_rgbdRequest,srv_rgbdResponse
import message_filters
from megmeet_pcl.msg import point as pcl_point

from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Bool
from std_msgs.msg import Int32MultiArray

# NEW: 提供查询服务
from std_srvs.srv import Trigger, TriggerResponse

import sensor_msgs.point_cloud2 as pc2

from geometry_msgs.msg import Twist

from camera_sdk.srv import FetchImgSn, FetchImgSnRequest


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 获取当前日期，到天（格式：YYYY-MM-DD）
def get_current_date():
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    return current_date

# 获取从当前时间到指定日期0时0分0秒经过的微秒数
def get_elapsed_microseconds(target_date):
    dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    current_time = datetime.datetime.strptime(dt_ms, "%Y-%m-%d %H:%M:%S.%f")
    target_datetime = datetime.datetime.strptime(target_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    time_diff = current_time - target_datetime
    elapsed_microseconds = int(time_diff.total_seconds() * 1_000_000)
    return elapsed_microseconds

def get_current_daytime_str():
    return datetime.datetime.now().strftime('%H-%M-%S')


def save_ply_file(pc2msg, savepath):
    points = pc2.read_points(pc2msg,
                                field_names=("x", "y", "z"),
                                skip_nans=False)

    pts = [p for p in points]
    n_pts = len(pts)
    if n_pts == 0:
        rospy.logwarn("[pc2_to_ply] Received empty point cloud")
        return

    rospy.loginfo(f"[pc2_to_ply] Writing {n_pts} points to {savepath}")

    with open(savepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_pts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x} {y} {z}\n")


def _same_name(a, b):
    """与 C++ 版相同的名字比较逻辑，兼容前导 '/'"""
    if a == b:
        return True
    if a and a[0] != '/' and ("/" + a) == b:
        return True
    if b and b[0] != '/' and ("/" + b) == a:
        return True
    return False

class DataCollectNode:

    def __init__(self):

        rospy.init_node('data_collect_node', anonymous = True)
        self.save_dir_root = rospy.get_param('save_dir_root', '/mnt/data/')
        ensure_dir(self.save_dir_root)
        rospy.loginfo("[DataCollect] Saving data to %s.", self.save_dir_root)
        self.run_mode = False

        self.vis_mix_mode = True

        self.bridge = CvBridge()

        self.sub_image = rospy.Subscriber('/image_topic', Image_msg, self.cb_save_image, queue_size = 1)
        self.sub_image_extra = rospy.Subscriber('/image_topic1', Image_msg, self.cb_save_image_extra, queue_size = 1)
        self.sub_iamge_extra2 = rospy.Subscriber('/image_topic2', Image_msg, self.cb_save_image_extra2, queue_size = 1)
        self.sub_fix_ply = rospy.Subscriber('/fixed_scan', PointCloud2_msg, self.cb_save_pcd, queue_size = 1)
        self.sub_joint_state = rospy.Subscriber('/joint_pos', joint_pos, self.cb_save_joint_state, queue_size = 1)
        self.sub_tool_pose = rospy.Subscriber('/tool_pos', jog_pos, self.cb_save_tool_pose, queue_size = 1)
        self.sub_image_rgbd = rospy.Subscriber('/image_topic_3d', Image_msg, self.cb_save_image_rgbd, queue_size = 1)
        self.cmd_pose = rospy.Subscriber('/cmd_vel',jog_pos,self.save_speed_cmd,queue_size=1)
        self.cmd_freq = rospy.Subscriber('/cmd_freq', freq_cfg, self.save_freq_cmd, queue_size=1)



        self.sub_weld_signal = rospy.Subscriber(
            '/welding_start_signal', Bool, self.cb_save_weld_signal, queue_size = 1)
        self.sub_weld_state = rospy.Subscriber(
            '/welding_state', Int32MultiArray, self.cb_save_weld_state, queue_size=1)

        rospy.wait_for_service('fetch_img_sn')
        self.fetch_sn = rospy.ServiceProxy('fetch_img_sn', FetchImgSn)

        self.enable_save_rgbd_log = rospy.get_param('enable_save_rgbd_log', True)
        self.enable_save_melt_rgb_log = rospy.get_param('enable_save_melt_rgb_log', False)

        image_sub = message_filters.Subscriber("/sam_detect_image", Image_msg)
        point1_sub = message_filters.Subscriber("/head", pcl_point)
        point2_sub = message_filters.Subscriber("/bottom", pcl_point)
        self.sub_image_log = rospy.Subscriber('/sam_detect_image', Image_msg, self.cb_save_image_log, queue_size = 1)
        
        self._vis_spd_cur = 0.0
        self._vis_spd_stg = 0
        self.sub_spd_sts = rospy.Subscriber('/spd_sts_vis', spd_sts, self.cb_recv_spd_sts, queue_size = 1)


        # 设置服务(供逻辑节点调用采集低频数据)
        self.srv_mode_activate = rospy.Service('data_collect_activate', Empty, self.data_collect_activate)
        self.srv_mode_deactivate = rospy.Service('data_collect_deactivate', Empty, self.data_collect_deactivate)

        self.srv_save_rgbd = rospy.Service('save_rgb_depth', srv_rgbd, self.save_rgb_depth)

        self.latest_cam1_img = None
        self.latest_cam2_img = None

        self._joy_x = 0.0
        self._joy_y = 0.0
        self._joy_z = 0.0
        self.sub_joy_pose = rospy.Subscriber("/spacenav/twist", Twist, self._vis_joy_pos_cb, queue_size=10)

        # 节点状态初始为休眠模式
        self.skip_nans = True

        # NEW: 发布 run_mode 状态的 topic（latch，GUI 后启动也能立刻拿到最新状态）
        self.pub_run_mode = rospy.Publisher('/data_collect_run_mode', Bool, queue_size=1, latch=True)
        self.pub_run_mode.publish(Bool(data=self.run_mode))

        # NEW: 提供查询 run_mode 状态的 service（Trigger.success = run_mode）
        self.srv_get_run_mode = rospy.Service('data_collect_get_run_mode', Trigger, self.data_collect_get_run_mode)

        rospy.loginfo("[DataCollect] init success.")

    # NEW: 查询服务回调
    def data_collect_get_run_mode(self, req):
        return TriggerResponse(success=bool(self.run_mode),
                               message=("run_mode=1" if self.run_mode else "run_mode=0"))

    def data_collect_activate(self, req):
        if self.run_mode == True:
            return EmptyResponse()
        self.save_date = get_current_date()
        timestamp = get_current_daytime_str()
        root_dir = os.path.join(self.save_dir_root, self.save_date, str(timestamp))
        rospy.loginfo(f"[DataCollect] current save dir {root_dir}.")
        ensure_dir(root_dir)

        self.save_dir_camera_log = os.path.join(root_dir, 'camera_log')
        self.save_dir_camera_depth = os.path.join(root_dir, 'camera_depth')
        self.save_dir_camera_depth_log = os.path.join(root_dir, 'camera_depth_log')
        self.save_dir_camera_image_rgbd = os.path.join(root_dir, 'camera_rgbd')
        self.save_dir_point_cloud = os.path.join(root_dir, 'scan_point_cloud')
        self.save_dir_robot_state = os.path.join(root_dir, 'robot_state')
        self.save_dir_welding_state = os.path.join(root_dir, 'welding_state')
        self.save_dir_control_cmd = os.path.join(root_dir, 'control_cmd')
        self.save_dir_state_type = os.path.join(root_dir, 'state_type')

        sn_by_index = {}
        try:
            req = FetchImgSnRequest()
            resp = self.fetch_sn(req)
            if len(resp.index_list) == 0:
                rospy.loginfo("当前没有已初始化的相机。")
            else:
                rospy.loginfo("当前相机列表：")
                for idx, sn in zip(resp.index_list, resp.sn_list):
                    rospy.loginfo("  camera index=%d, sn=%s", idx, sn)
                    sn_by_index[idx] = sn

        except rospy.ServiceException as e:
            rospy.logerr("调用 fetch_img_sn 失败: %s", e)

        cam0_name = "camera"
        cam1_name = "camera_extra"
        cam2_name = "camera_extra_2"

        if 0 in sn_by_index:
            cam0_name = f"camera_{sn_by_index[0]}"
        if 1 in sn_by_index:
            cam1_name = f"camera_{sn_by_index[1]}"
        if 2 in sn_by_index:
            cam2_name = f"camera_{sn_by_index[2]}"

        self.save_dir_camera        = os.path.join(root_dir, cam0_name)
        self.save_dir_camera_extra  = os.path.join(root_dir, cam1_name)
        self.save_dir_camera_extra2 = os.path.join(root_dir, cam2_name)

        for folder in [self.save_dir_camera, self.save_dir_camera_extra, self.save_dir_camera_extra2,
                       self.save_dir_camera_depth, self.save_dir_camera_image_rgbd, self.save_dir_point_cloud,
                       self.save_dir_robot_state, self.save_dir_welding_state, self.save_dir_control_cmd, self.save_dir_state_type,self.save_dir_camera_depth_log,self.save_dir_camera_log]:
            ensure_dir(folder)

        rospy.loginfo("[DataCollect] Data collection activated. Saving to %s.", root_dir)
        self.run_mode = True

        # NEW: 发布状态
        self.pub_run_mode.publish(Bool(data=self.run_mode))

        # save calibration 
        try:
            calib_path = os.path.join(self.save_dir_robot_state, "calibration.csv")

            sn1 = sn_by_index.get(0, "unknown_sn1")
            sn2 = sn_by_index.get(1, "unknown_sn2")
            sn3 = sn_by_index.get(2, "unknown_sn3")

            yaml_path = rospy.get_param("~cfg", "")
            if not yaml_path:
                raise RuntimeError('ROS param "~cfg" is empty. Please set it in launch.')

            if not os.path.exists(yaml_path):
                raise FileNotFoundError(f"cfg yaml not found: {yaml_path}")

            with open(yaml_path, "r", encoding="utf-8") as yf:
                cfg = yaml.safe_load(yf) or {}

            tool = cfg.get("tool", {}) if isinstance(cfg, dict) else {}



            # T_falan_tcp read from yaml: T_falan * T_falan_tcp = T_tcp
            # 单位从mm和deg转成m和rad

            T_tcp_x  = float(tool.get("x",  0.0)) / 1000.0
            T_tcp_y  = float(tool.get("y",  0.0)) / 1000.0
            T_tcp_z  = float(tool.get("z",  0.0)) / 1000.0
            T_tcp_rx = float(tool.get("rx", 0.0)) * 3.1415926 / 180.0
            T_tcp_ry = float(tool.get("ry", 0.0)) * 3.1415926 / 180.0
            T_tcp_rz = float(tool.get("rz", 0.0)) * 3.1415926 / 180.0

            lines = [
                f"T_falan_tcp {T_tcp_x} {T_tcp_y} {T_tcp_z} {T_tcp_rx} {T_tcp_ry} {T_tcp_rz}",
                f"T_camera-{sn1}_falan 0 0 0 0 0 0",
                f"T_camera-{sn2}_falan 0 0 0 0 0 0",
                f"T_camera-{sn3}_falan 0 0 0 0 0 0",
            ]

            with open(calib_path, "w", encoding="utf-8", newline="\n") as f:
                for line in lines:
                    f.write(line + "\n")

            rospy.loginfo("[DataCollect] Saved calibration.csv to %s", calib_path)
        except Exception as e:
            rospy.logerr("[DataCollect] Failed to save calibration.csv: %s", e)


        return EmptyResponse()

    def data_collect_deactivate(self, req):
        rospy.loginfo("[DataCollect] Data collection deactivated.")
        self.run_mode = False

        # NEW: 发布状态
        self.pub_run_mode.publish(Bool(data=self.run_mode))

        return EmptyResponse()


    def _show_mixpanel(self):
        if self.latest_cam1_img is None or self.latest_cam2_img is None:
            return

        img1 = self.latest_cam1_img
        img2 = self.latest_cam2_img

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        src_x0, src_y0 = 424, 296
        roi_w, roi_h = 396, 396
        src_x1 = src_x0 + roi_w
        src_y1 = src_y0 + roi_h

        if src_x0 >= w2 or src_y0 >= h2:
            return
        src_x1 = min(src_x1, w2)
        src_y1 = min(src_y1, h2)

        roi = img2[src_y0:src_y1, src_x0:src_x1]

        dst_x0, dst_y0 = 430, 562
        dst_x1 = dst_x0 + roi.shape[1]
        dst_y1 = dst_y0 + roi.shape[0]

        dst_x1 = min(dst_x1, w1)
        dst_y1 = min(dst_y1, h1)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return

        dst_w = dst_x1 - dst_x0
        dst_h = dst_y1 - dst_y0
        roi_cropped = roi[:dst_h, :dst_w]

        mixed = img1.copy()
        mixed[dst_y0:dst_y1, dst_x0:dst_x1] = roi_cropped

        mixed_show = cv2.resize(mixed, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)

        y_val = 0
        radius = 0
        if self.vis_mix_mode:
            y_val = self._joy_y
            center = (578, 322)
            k_abs = 0.1
            k_radius = 30
            max_radius = 60

            abs_y = abs(y_val)
            radius = int(round(k_radius * (abs_y / k_abs)))
            if radius > max_radius:
                radius = max_radius

            if y_val < 0:
                color = (255, 255, 0)
            elif y_val > 0:
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)

            if radius == 0:
                radius = 1

            overlay = mixed_show.copy()
            cv2.circle(overlay, center, radius, color, thickness=-1)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, mixed_show, 1 - alpha, 0, mixed_show)

        # ===== NEW: 左上角叠加速度信息 self._vis_spd_cur(self._vis_spd_stg) =====
        spd_text = f"{self._vis_spd_cur:.3f}({int(self._vis_spd_stg)})"
        org = (10, 30)  # 左上角位置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # 先画黑色描边，增强可读性
        cv2.putText(mixed_show, spd_text, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # 再画正文（黄色）
        cv2.putText(mixed_show, spd_text, org, font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        # ================================================================

        cv2.imshow("megmeet_cam", mixed_show)
        cv2.waitKey(1)
        return mixed_show


    def cb_save_image(self, msg):
        if self.run_mode == True:
            timestamp = get_elapsed_microseconds(self.save_date)
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image_path = os.path.join(self.save_dir_camera, f'{timestamp}.jpg')
            timestamp = get_elapsed_microseconds(self.save_date)
            print("timestamp:",timestamp)
            cv2.imwrite(image_path, cv_img)
            timestamp = get_elapsed_microseconds(self.save_date)
            print("write suc timestamp:",timestamp)

            self.latest_cam2_img = cv_img
            mix_image = self._show_mixpanel()

    def cb_save_image_extra(self, msg):
        if self.run_mode == True:
            timestamp = get_elapsed_microseconds(self.save_date)
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image_path = os.path.join(self.save_dir_camera_extra, f'{timestamp}.jpg')
            cv2.imwrite(image_path, cv_img)

    def cb_save_image_extra2(self, msg):
        if self.run_mode == True:
            timestamp = get_elapsed_microseconds(self.save_date)
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image_path = os.path.join(self.save_dir_camera_extra2, f'{timestamp}.jpg')
            cv2.imwrite(image_path, cv_img)

            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.latest_cam1_img = cv_img

    def cb_save_image_rgbd(self, msg):
        if self.run_mode == True:
            timestamp = get_elapsed_microseconds(self.save_date)
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image_path = os.path.join(self.save_dir_camera_image_rgbd, f'{timestamp}.jpg')
            cv2.imwrite(image_path, cv_img)

    def cb_save_image_log(self, image_msg):
        if self.run_mode == True and self.enable_save_melt_rgb_log == True:
            timestamp = get_elapsed_microseconds(self.save_date)
            cv_img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            image_path = os.path.join(self.save_dir_camera_log, f'{timestamp}.jpg')
            cv2.imwrite(image_path, cv_img)

    def cb_recv_spd_sts(self, msg):
        if self.run_mode == True:
            self._vis_spd_cur = msg.spd_cur
            self._vis_spd_stg = msg.spd_stg


    def cb_save_pcd(self, msg):
        if self.run_mode == True:
            timestamp = get_elapsed_microseconds(self.save_date)
            pcd_path = os.path.join(self.save_dir_point_cloud, f'{timestamp}.bin')
            save_ply_file(msg, pcd_path)

    def cb_save_joint_state(self, msg):
        if self.run_mode == True:
            timestamp = get_elapsed_microseconds(self.save_date)
            joint_state_path = os.path.join(self.save_dir_robot_state, 'joint_state.csv')
            with open(joint_state_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp] + [msg.j1, msg.j2, msg.j3, msg.j4, msg.j5, msg.j6])

    def cb_save_tool_pose(self, msg):
        if self.run_mode == True:
            timestamp = get_elapsed_microseconds(self.save_date)
            tool_pose_path = os.path.join(self.save_dir_robot_state, 'tool_pose.csv')
            with open(tool_pose_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp] + [msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz])

    def save_rgb_depth(self, req):
        if self.run_mode == True:
            print('save rgbd image !!!!!')
            timestamp = get_elapsed_microseconds(self.save_date)
            ply_path = os.path.join(self.save_dir_camera_depth, f'{timestamp}.ply')
            save_ply_file(req.points, ply_path)
            cv_img = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
            image_path = os.path.join(self.save_dir_camera_depth, f'{timestamp}.jpg')
            cv2.imwrite(image_path, cv_img)
            scan_pose_path = os.path.join(self.save_dir_camera_depth, 'scan_pose.csv')
            with open(scan_pose_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp] + [req.pose.x, req.pose.y, req.pose.z, req.pose.rx, req.pose.ry, req.pose.rz])
        else:
            print('skip save data.')
        return srv_rgbdResponse()

    def save_speed_cmd(self, req):
        if self.run_mode == True:
            timestamp = get_elapsed_microseconds(self.save_date)
            speed_cmd_path = os.path.join(self.save_dir_control_cmd, 'control_speed.csv')
            with open(speed_cmd_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp] + [req.x, req.y, req.z, req.rx, req.ry, req.rz])

    def save_freq_cmd(self, req):
        if self.run_mode == True:
            timestamp = get_elapsed_microseconds(self.save_date)
            freq_cmd_path = os.path.join(self.save_dir_control_cmd, 'control_freq.csv')
            with open(freq_cmd_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp] + [req.vel, req.freq, req.amp])

    def cb_save_weld_signal(self, req):
        if self.run_mode == True:
            timestamp = get_elapsed_microseconds(self.save_date)
            weld_state_path = os.path.join(self.save_dir_welding_state, 'weld_signal.csv')
            with open(weld_state_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp] + [req])

    def cb_save_weld_state(self, req):
        if self.run_mode == True:
            timestamp = get_elapsed_microseconds(self.save_date)
            weld_state_path = os.path.join(self.save_dir_welding_state, 'weld_state.csv')
            if os.path.exists(weld_state_path):
                return
            with open(weld_state_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp] + [req.data[0], req.data[1]])

    def _vis_joy_pos_cb(self, msg):
        if not self.vis_mix_mode:
            return

        self._allowed_twist_publisher = "/haply_inverse3_node"
        pub = ''
        header = getattr(msg, '_connection_header', None)
        if header is not None:
            pub = header.get('callerid', '')
        if self._allowed_twist_publisher and \
           not _same_name(pub, self._allowed_twist_publisher):
            return
        self._joy_x = msg.linear.x
        self._joy_y = msg.linear.y
        self._joy_z = msg.linear.z

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    node = DataCollectNode()
    node.spin()
