#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil

import rospy
from robot_control.srv import Move


def read_points(points_file):
    points = []
    with open(points_file, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 6:
                raise ValueError(
                    f"第 {line_no} 行格式错误，期望 6 个逗号分隔值，实际 {len(parts)} 个: {line}"
                )

            try:
                pose = tuple(float(v) for v in parts)
            except ValueError as exc:
                raise ValueError(f"第 {line_no} 行存在非数字内容: {line}") from exc

            points.append(pose)

    if len(points) != 4:
        raise ValueError(f"4_points.txt 需要恰好 4 行有效点位，当前读取到 {len(points)} 行")

    return points


def move_to_target_with_workaround(move_proxy, target):
    target_x, target_y, target_z, target_rx, target_ry, target_rz = target

    # # 机械臂移动逻辑保持不变：先移开，再移回目标
    # print("步骤1: 先移开...")
    # move_proxy(-0.4500, target_y, target_z, target_rx, target_ry, target_rz, "", True)

    

    print("步骤2: 再移回目标...")
    move_proxy(target_x, target_y, target_z, target_rx, target_ry, target_rz, "", True)
    #rospy.sleep(500000)


def run_capture_script(script_path):
    print(f"执行脚本: {script_path}")
    result = subprocess.run([sys.executable, script_path], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"脚本执行失败，返回码: {result.returncode}")


def rename_pic_folder(current_dir, index):
    """重命名 pic 文件夹为 pic_index"""
    pic_path = os.path.join(current_dir, "pic")
    if os.path.exists(pic_path) and os.path.isdir(pic_path):
        new_pic_path = os.path.join(current_dir, f"pic_{index}")
        # 如果目标文件夹已存在，先删除或处理（这里选择覆盖删除）
        if os.path.exists(new_pic_path):
            shutil.rmtree(new_pic_path)
        shutil.move(pic_path, new_pic_path)
        print(f"已将 pic 文件夹重命名为: pic_{index}")
    else:
        print(f"警告: 未找到 pic 文件夹，跳过重命名")


rospy.init_node('move_absolute_workaround')
rospy.wait_for_service('/mov_jog')
move = rospy.ServiceProxy('/mov_jog', Move)

current_dir = os.path.dirname(os.path.abspath(__file__))
points_file = os.path.join(current_dir, "4_points.txt")
capture_script = os.path.join(current_dir, "test.py")

if not os.path.exists(points_file):
    raise FileNotFoundError(f"未找到点位文件: {points_file}")

if not os.path.exists(capture_script):
    raise FileNotFoundError(f"未找到脚本文件: {capture_script}")

points = read_points(points_file)

for idx, point in enumerate(points, start=1):
    print(f"\n=== 前往第 {idx} 个点 ===")
    move_to_target_with_workaround(move, point)
    print(f"第 {idx} 个点到位，开始执行拍照脚本...")
    run_capture_script(capture_script)
    print(f"第 {idx} 个点流程完成")
    
    # 重命名生成的 pic 文件夹
    rename_pic_folder(current_dir, idx)

print("\n全部4个点执行完成")
