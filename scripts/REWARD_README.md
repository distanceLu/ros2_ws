# 毛笔笔迹 Reward 使用说明

本目录提供两套 episode 级奖励函数，供真机强化学习使用。

| 文件 | 作用 | 输入 | 稳健性 |
|------|------|------|--------|
| `trajectory_reward.py` | 基于 TCP 轨迹评分（直度/长度/平滑度） | `robot_state/tool_pose.csv` | 高，不受图像干扰 |
| `ink_reward.py` | 基于纸面图像评分（墨迹形状/coverage/overflow） | before/after 纸面相机图 | 中，受机械臂遮挡影响 |
| `score_trajectories.py` | 批量验证脚本，同时跑两种 reward | `data_collect/YYYY-MM-DD/` | — |

## 快速验证

```bash
cd /home/shugen/yanjie/ros2_ws

# 单 session 轨迹评分
python3 scripts/trajectory_reward.py \
  --tool-pose data_collect/2026-06-25/15-50-43/robot_state/tool_pose.csv

# 单 session 图像评分
python3 scripts/ink_reward.py \
  --before data_collect/2026-06-25/15-50-43/camera_paper_aruco/57045399891.000001.jpg \
  --after  data_collect/2026-06-25/15-50-43/camera_paper_aruco/57061002341.000152.jpg \
  --out-dir /tmp/ink_reward_test

# 批量评分某天全部 session
python3 scripts/score_trajectories.py --data-root data_collect/2026-06-25

# 只跑轨迹评分（快）
python3 scripts/score_trajectories.py --data-root data_collect/2026-06-25 --skip-image

# 只评指定 session
python3 scripts/score_trajectories.py --data-root data_collect/2026-06-25 \
  --session 15-50-43 --session 17-50-16
```

## 输出

每个 session 目录下生成：
- `trajectory_reward.json` — 轨迹评分详情
- `image_reward.json` — 图像评分详情
- `reward_vis.png` — 图像评分可视化（绿=墨迹，红=主轴，黄=bbox）

日期目录下生成：
- `reward_summary.csv` — 全部 session 汇总

## 在 RL 训练中调用

### Python 直接调用

```python
import sys
sys.path.insert(0, "/home/shugen/yanjie/ros2_ws/scripts")

from trajectory_reward import score_trajectory_from_csv
from ink_reward import score_episode
import cv2

# 方式 1：轨迹 reward（推荐，稳健）
result = score_trajectory_from_csv("/path/to/session/robot_state/tool_pose.csv")
if result.valid:
    reward = result.reward   # [0, 1] 标量
else:
    reward = 0.0
    print("invalid:", result.reason)

# 方式 2：图像 reward
before = cv2.imread("/path/to/before.jpg")
after  = cv2.imread("/path/to/after.jpg")
result = score_episode(before, after, generate_vis=False)
if result.valid:
    reward = result.reward
else:
    reward = 0.0

# 方式 3：组合 reward（推荐）
traj_result = score_trajectory_from_csv(tool_pose_csv)
img_result  = score_episode(before, after, generate_vis=False)
if traj_result.valid and img_result.valid:
    reward = 0.5 * traj_result.reward + 0.5 * img_result.reward
elif traj_result.valid:
    reward = traj_result.reward
elif img_result.valid:
    reward = img_result.reward
else:
    reward = 0.0
```

### 在真机 RL episode 闭环中

```python
# episode 开始时
before_frame = capture_paper_camera_frame()   # 存一张
tool_pose_buffer = []                          # 开始记录 TCP

# episode 进行中
# 每个 step 把 /tool_pos 存入 tool_pose_buffer

# episode 结束时
after_frame = capture_paper_camera_frame()    # 存一张
# 写临时 tool_pose.csv 或直接传 numpy 数组
import numpy as np
ts = np.array([t for t, _ in tool_pose_buffer])
xyz = np.array([[p['x'], p['y'], p['z']] for _, p in tool_pose_buffer])
from trajectory_reward import score_trajectory
traj_res = score_trajectory(ts, xyz)
img_res  = score_episode(before_frame, after_frame, generate_vis=False)
reward = combine_reward(traj_res, img_res)
```

## 评分指标说明

### 轨迹 reward (`trajectory_reward.py`)

- **straightness** (权重 0.5)：侧向跨度 / 轨迹长度，越直越接近 1
- **length_score** (权重 0.3)：主方向跨度 / TARGET_LENGTH_MM（默认 100mm）
- **smoothness** (权重 0.2)：基于 jerk 的平滑度，1/(1+median_jerk/JERK_SCALE)
- score = 100 * (0.5*straightness + 0.3*length_score + 0.2*smoothness)
- reward = score / 100

可调参数（脚本顶部）：
- `TARGET_LENGTH_MM = 100.0` — 目标笔画长度
- `DRAW_Z_QUANTILE = 0.5` — z 低于该分位数视为画线阶段
- `JERK_SCALE = 5000.0` — smoothness 尺度

### 图像 reward (`ink_reward.py`)

两种模式，`auto` 默认优先 `ink_shape`：

**ink_shape 模式**（对市售练习纸稳健）：
- **aspect_score** (0.20)：h/w 接近理想竖线 aspect (3.5)
- **verticality** (0.15)：PCA 主轴与竖直方向夹角越小越高
- **straightness_ink** (0.25)：PCA 主轴残差中位数越小越高
- **length_score** (0.40)：主轴长度接近 TARGET_LENGTH_PX (220px)
- score = 100 * (0.20*aspect + 0.15*verticality + 0.25*straightness + 0.40*length)

**target_overlap 模式**（打印模板纸用，需 ArUco-free pill 检测可靠）：
- **coverage** (权重 70)：墨迹覆盖目标区域比例
- **shape_score** (权重 30)：Hu 矩形状相似度
- **overflow** (惩罚 80)：墨迹超出容忍区域比例
- score = clip(70*coverage + 30*shape - 80*overflow, 0, 100)

可调参数（脚本顶部）：见 `INK_*`、`PILL_*`、`TARGET_*`、`W_*` 系列。

## 已知局限与改进方向

1. **机械臂遮挡**：当前所有采集帧里机械臂夹具都在画面中，差分会捕获夹具移动假阳性。
   - 已通过 HSV 颜色过滤 + 竖直形态学开运算 + 形状过滤大幅抑制。
   - **根本解决**：episode 结束后让机械臂抬离纸面到画面外，再拍一张 clean after 帧。
2. **市售练习纸 pill 检测**：每个格子有 5×5 笔画网格，`target_overlap` 模式会误检。
   - 已让 `auto` 默认走 `ink_shape`，并加可靠性守卫（target 面积 < 15% ink_area 时弃用）。
   - 若改用程序生成模板纸（带 ArUco，如 `paper_template/`），可启用可靠的 target_overlap 模式。
3. **轨迹 reward 区分度**：演示数据质量高且相似，分数集中在 90~100。
   - RL agent 画的更差时分数会自然拉开。
   - 可调低 `TARGET_LENGTH_MM` 或调高 `JERK_SCALE` 让 smoothness 更敏感。
4. **target_id 来源**：当前 `ink_shape` 模式不依赖 target_id；若需要 per-target 评分，
   建议在 `training_session.py` 采集时把 `target_id` 写入 `session_meta.json`。

## 评分分布（2026-06-25 验证，71 个 session）

| reward 类型 | valid | min | median | max | std |
|-------------|-------|-----|--------|-----|-----|
| trajectory  | 62/71 | 89.8 | 96.4 | 99.9 | 2.26 |
| image       | 71/71 | 50.0 | 64.7 | 83.8 | 6.72 |

9 个轨迹 invalid 均为 `tool_pose.csv` 空（采集失败）。图像 reward 区分度更好。

## 不依赖 ROS 的纯 Python 验证

两个 reward 模块都是纯 Python + OpenCV，无需 source ROS 环境即可运行：

```bash
cd /home/shugen/yanjie/ros2_ws
python3 scripts/trajectory_reward.py --tool-pose <csv>
python3 scripts/ink_reward.py --before <jpg> --after <jpg>
```

只在真机闭环调用时才需要 ROS 环境来获取 `/tool_pos` 和纸面相机帧。
