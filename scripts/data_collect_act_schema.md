# data_collect 到 ACT/ALOHA HDF5 数据说明

本文档基于本地数据目录 `/home/shugen/yanjie/ros2_ws/data_collect` 的实际文件检查结果编写，用于后续编写 `constants.py`、预处理脚本、`Dataset/Dataloader` 和校验脚本。

重点结论：

- 当前数据是原始采集目录，不是 ACT/ALOHA 风格的 `episode_0.hdf5`、`episode_1.hdf5`。
- 当前每个 episode/session 只有 TCP 工具位姿 `timestamp,x,y,z,rx,ry,rz`，没有关节级 `qpos/qvel`，也没有明确的机器人控制 action。
- 若要接入当前 ACT loader，建议先转换为 HDF5，并把 `/observations/qpos` 和 `/action` 暂定为 6D TCP pose 序列；这不是原始 ACT 的双臂 14 维关节定义。
- 三个必须人工确认的关键点：`qpos/action` 是否允许使用 TCP pose、`action[t]` 的时间对齐规则、哪些 session 是成功轨迹。

## 1. 数据总览

```text
数据集名称：auto_welding_data_collect_2026_06_15
任务名称：welding_tool_pose_imitation_6d_tcp_pose（建议名，待确认）
数据来源：真实机器人 + 人工遥操作/交互控制采集（从采集脚本语义推断）
OSS bucket：gpu-ai
OSS endpoint：未在本地文件中发现，待确认
OSS 路径前缀：oss://gpu-ai/lcx/brush_pen/data_collect/
本地建议下载路径：/home/shugen/yanjie/ros2_ws/data_collect
总 episode 数：24
每个 episode 是否独立文件：当前不是；每个 episode 是一个独立目录
文件命名规则：YYYY-MM-DD/HH-MM-SS/
是否有 train/val/test 官方划分：未发现
采集日期或版本：2026-06-15
是否有失败 episode，失败如何标记：未发现成功/失败标注文件，需人工补充
```

建议的 `constants.py` 任务配置：

```text
task_name: welding_tool_pose_2026_06_15
dataset_dir: /home/shugen/yanjie/ros2_ws/data_collect_act_hdf5/welding_tool_pose_2026_06_15
num_episodes: 24
episode_len: 不固定；建议转换时以每个 HDF5 的实际 T 为准，训练端用 padding
camera_names:
  - pool
  - scan_2d
```

当前原始目录大小：

```text
/home/shugen/yanjie/ros2_ws/data_collect: 7.7G
/home/shugen/yanjie/ros2_ws/data_collect/2026-06-15: 7.7G
```

## 2. 当前原始文件结构

当前一个真实 session 的结构如下：

```text
/home/shugen/yanjie/ros2_ws/data_collect/2026-06-15/19-49-26/
  camera_pool/
    71366530114.000001.jpg
    ...
  camera_3d_2d/
    71366861003.000001.jpg
    ...
  robot_state/
    tool_pose.csv
  session_meta.json
```

`session_meta.json` 示例：

```json
{
  "created_at": "2026-06-15T19:49:26.527170",
  "pool_topic": "/pool_camera/image_raw",
  "scan_topic": "/scan/image_raw",
  "capture_2d_service": "/capture_2d",
  "capture_3d_2d_hz": 5.0,
  "save_every_n_pool": 1
}
```

`tool_pose.csv` 字段：

```text
timestamp,x,y,z,rx,ry,rz
```

注意：部分 CSV 的 header 不在第一行。比如 `19-49-26/tool_pose.csv` 第一行是数据，第二行才是 header。预处理脚本不能简单用第一行作为 header，应按行过滤：

- 跳过等于 `timestamp,x,y,z,rx,ry,rz` 的 header 行。
- 解析所有 7 列数字行。
- 对无法解析或列数不等于 7 的行报错或记录到校验报告。

## 3. 建议转换后的 HDF5 schema

为了兼容当前默认 ACT/ALOHA loader，建议每个 session 转成一个独立 HDF5：

```text
episode_0.hdf5
attrs:
  sim: False
  source_session: "2026-06-15/19-49-26"
  raw_qpos_semantics: "tcp_pose_xyz_rxryrz"
datasets:
  /observations/qpos: shape=(T, 6), dtype=float32
  /observations/qvel: shape=(T, 6), dtype=float32
  /observations/images/pool: shape=(T, H, W, 3), dtype=uint8
  /observations/images/scan_2d: shape=(T, H, W, 3), dtype=uint8
  /action: shape=(T, 6), dtype=float32
```

字段映射建议：

```text
/observations/qpos[:, 0:6] = [x, y, z, rx, ry, rz]
/observations/qvel = 对 qpos 按时间差分得到的 TCP 线/角速度，或全 0（需在文档和代码中固定）
/action = 下一步或当前步 TCP pose 目标，具体见第 5 节时间对齐
root.attrs['sim'] = False
```

图像建议：

```text
camera_pool -> /observations/images/pool
camera_3d_2d -> /observations/images/scan_2d
```

当前原始图像是 JPEG 压缩文件；转换到 HDF5 时，ACT loader 通常期望未压缩 `uint8` 数组。如果要在 HDF5 内继续压缩，可用 HDF5 chunk + gzip/lzf，但 loader 侧不应按 JPEG bytes 读取，除非同步改 loader。

## 4. 实际数据统计

所有原始 session 都位于：

```text
/home/shugen/yanjie/ros2_ws/data_collect/2026-06-15/
```

逐 session 统计：

```text
session   pool_jpg  scan_2d_jpg  tool_pose_csv_lines  csv_header_line
19-49-26       871          222                 4402                2
19-51-08       782          181                 4162                1
19-52-40       375           98                 2047                1
19-58-09       957          237                 5099                1
20-04-10       627          170                 3520                1
20-08-19       562          146                 2943                1
20-14-47       402          108                 2140                3
20-15-23       819          211                 4401                1
20-17-50       457          125                 2597                3
20-23-32       780          185                 4048                1
20-24-49       805          207                 4290                1
20-30-54       438          103                 2307                1
20-34-00       490          128                 2724                1
20-36-54       652          159                 3557                1
20-37-56       687          155                 3450                1
20-38-48       562          132                 2933                1
20-40-04       512          122                 2561                2
20-40-45       386           86                 1877                1
20-41-17       642          168                 3387                1
20-42-14       374           96                 2071                1
20-42-53       484          115                 2523                1
20-43-29       534          127                 2688                1
20-44-23       386           76                 1978                1
20-45-14       453          108                 2455                1
```

图像格式：

```text
camera_pool:
  文件格式：JPEG
  PIL mode：RGB
  原始分辨率：1280 x 1024
  建议 HDF5 shape：HWC，即 (T, 1024, 1280, 3)，或预处理 resize 后 (T, H, W, 3)

camera_3d_2d:
  文件格式：JPEG
  PIL mode：RGB
  原始分辨率：1440 x 1080
  建议 HDF5 shape：HWC，即 (T, 1080, 1440, 3)，或预处理 resize 后 (T, H, W, 3)
```

## 5. 机器人状态定义

当前数据没有关节状态，只有 TCP 工具位姿：

```text
qpos[0]: x，TCP 位置 x，单位疑似 meter，需机器人接口确认
qpos[1]: y，TCP 位置 y，单位疑似 meter，需机器人接口确认
qpos[2]: z，TCP 位置 z，单位疑似 meter，需机器人接口确认
qpos[3]: rx，TCP 姿态 rx，单位疑似 rad，需机器人接口确认
qpos[4]: ry，TCP 姿态 ry，单位疑似 rad，需机器人接口确认
qpos[5]: rz，TCP 姿态 rz，单位疑似 rad，需机器人接口确认
```

缺失且必须确认的信息：

```text
关节顺序：当前数据无关节
gripper：当前数据无 gripper
左右手顺序：不适用，当前不是双臂数据
是否包含 base / torso / head / mobile platform：当前数据未包含
qvel 是否和 qpos 同顺序：若预处理生成 qvel，应与 qpos 同顺序
qvel 单位：若由差分生成，单位为 qpos 单位 / second
```

当前默认 ACT 代码如果假设双臂 14 维：

```text
left arm 6 joints + left gripper + right arm 6 joints + right gripper
```

则不能直接用于这批原始数据；需要把模型维度改为 6，或另行采集/恢复关节状态。

## 6. Action 定义

当前原始数据没有单独 action 文件，也没有 `control_speed.csv`。因此 `/action` 只能由 `tool_pose.csv` 派生，推荐二选一：

方案 A：绝对 TCP pose action

```text
action[t] = qpos[t + 1]
action_dim = 6
含义：obs[t] 后希望到达的下一帧 TCP pose
最后一帧 action 可复制 qpos[-1] 或丢弃最后一个 timestep
```

方案 B：当前 TCP pose action

```text
action[t] = qpos[t]
action_dim = 6
含义：行为克隆为当前位置复现；实现简单，但监督信号弱于下一步目标
```

推荐方案 A，因为更符合 `obs[t] -> action[t]` 的因果关系。

当前 `utils.py` 中真实数据逻辑：

```python
action = root['/action'][max(0, start_ts - 1):]
```

对这批由 TCP pose 派生的 HDF5，建议不要使用 `start_ts - 1` 偏移，除非预处理时明确把 `action[t]` 存成“从 obs[t-1] 到 obs[t] 的动作”。若采用推荐方案 A，应使用：

```python
action = root['/action'][start_ts:]
```

需要在 loader 中按任务类型开关，而不是全局套用真实数据偏移。

## 7. 相机信息

```text
camera_names:
  - pool
  - scan_2d
```

`pool` 相机：

```text
原始目录：camera_pool/
ROS topic：/pool_camera/image_raw
分辨率：1280 x 1024
文件格式：JPEG
PIL mode：RGB
HDF5 建议 shape：HWC
fps：未显式记录；从文件名时间戳看约高于 scan_2d，需用时间戳统计确认
是否同步：未严格同步，需要预处理按时间戳最近邻对齐
是否有丢帧：未标注，需要校验脚本统计相邻时间戳 gap
是否需要 resize：建议 resize 到训练统一尺寸，例如 320x256 或 640x512
是否需要 crop/旋转/翻转：未发现元数据，需人工视觉确认
内参/外参：未发现
安装位置说明：熔池相机，具体安装位姿待补充
```

`scan_2d` 相机：

```text
原始目录：camera_3d_2d/
ROS topic：/scan/image_raw
服务：/capture_2d
分辨率：1440 x 1080
文件格式：JPEG
PIL mode：RGB
HDF5 建议 shape：HWC
fps：session_meta.json 中 capture_3d_2d_hz = 5.0
是否同步：未严格同步，需要预处理按时间戳最近邻对齐
是否有丢帧：未标注，需要校验脚本统计相邻时间戳 gap
是否需要 resize：建议 resize 到训练统一尺寸，例如 320x240 或 640x480
是否需要 crop/旋转/翻转：未发现元数据，需人工视觉确认
内参/外参：未发现
安装位置说明：3D 相机输出的无激光 2D 图，具体安装位姿待补充
```

## 8. 时间与同步逻辑

已知：

```text
scan_2d 采集频率：5 Hz（session_meta.json: capture_3d_2d_hz = 5.0）
pool 保存策略：save_every_n_pool = 1
控制频率：当前数据未记录；项目默认 DT = 0.02，即 50 Hz，但这批图像不是 50 Hz
timestamp 字段：tool_pose.csv 和图像文件名均含整数时间戳
timestamp 单位：未在元数据中说明；从数值和相邻差推断像微秒级单调时钟，需采集节点确认
```

建议预处理同步策略：

1. 以 `scan_2d` 帧作为主时间轴，因为它显式为 5 Hz，且帧数较少。
2. 对每个 `scan_2d` timestamp，找最近的 `pool` 图像。
3. 对同一 timestamp，找最近的 `tool_pose.csv` 行作为 qpos。
4. 记录每次匹配的时间差，超过阈值的 timestep 标记为 invalid。阈值建议先设为：

```text
scan_2d -> pool: <= 100 ms
scan_2d -> tool_pose: <= 50 ms
```

5. HDF5 的 `T` 取有效同步后的 timestep 数，不强行所有 episode 等长；训练时用 padding。

若必须按 50 Hz 训练，需要从 `tool_pose.csv` 构建 50 Hz 状态序列，并对相机帧做重复/最近邻保持。这会产生大量重复图像，不建议作为第一版。

## 9. 数据质量与过滤规则

当前没有成功/失败标注。建议先人工补充一个 CSV：

```text
episode_id,session,status,reason
0,2026-06-15/19-49-26,unknown,
1,2026-06-15/19-51-08,unknown,
...
```

建议过滤规则：

```text
只训练 status == success 的 session。
过滤同步后 T < 50 的 episode。
过滤任一相机缺失的 timestep。
过滤 qpos 中含 NaN/Inf 的 timestep。
过滤相邻 qpos 跳变超过阈值的 timestep 或整条 episode。
过滤图像无法解码、尺寸异常、黑屏/过曝比例异常的 timestep。
对 CSV header 不在第一行的情况做兼容，但记录 warning。
```

当前已知问题：

```text
部分 tool_pose.csv header 不在第一行。
没有失败标注。
没有 action/control_speed 文件。
没有 joint qpos/qvel。
没有相机内外参。
```

## 10. 预处理建议

建议新增脚本：

```text
scripts/convert_data_collect_to_act_hdf5.py
```

输入：

```text
/home/shugen/yanjie/ros2_ws/data_collect/2026-06-15/
```

输出：

```text
/home/shugen/yanjie/ros2_ws/data_collect_act_hdf5/welding_tool_pose_2026_06_15/
  episode_0.hdf5
  episode_1.hdf5
  ...
  metadata.json
  quality_report.csv
```

处理逻辑：

```text
1. 枚举 YYYY-MM-DD/HH-MM-SS session。
2. 读取 session_meta.json。
3. 读取 tool_pose.csv，过滤 header 行，解析 timestamp,x,y,z,rx,ry,rz。
4. 读取 camera_pool/*.jpg 和 camera_3d_2d/*.jpg，解析文件名前半段为 timestamp。
5. 以 scan_2d 为主时间轴，最近邻匹配 pool 和 tool_pose。
6. 可选 resize 两路图像到固定尺寸。
7. qpos = [x,y,z,rx,ry,rz]。
8. qvel = qpos 差分 / dt；第一帧可置 0。
9. action = qpos 向前平移一帧，即 action[t] = qpos[min(t+1,T-1)]。
10. 写入 ACT/ALOHA HDF5 schema。
11. 写 quality_report.csv，包含 T、匹配时间差、缺帧、异常图像、qpos 跳变统计。
```

是否需要压缩 HDF5：

```text
建议图像 dataset 使用 chunks，并视训练 IO 情况选择 lzf 或 gzip。
如果训练读取瓶颈明显，优先保存 resize 后未压缩 uint8，减少解压开销。
```

是否需要统一 episode 长度：

```text
不建议在 HDF5 中 padding。
建议 HDF5 保存真实 T，Dataset 在采样 action chunk 时 padding 并返回 is_pad。
```

## 11. 归一化策略

当前代码可统计：

```text
qpos_mean / qpos_std
action_mean / action_std
```

对 6D TCP pose 数据的建议：

```text
可以先用所有 success episode 统计 mean/std。
如果没有 success 标注，先用全部 episode 统计，但训练报告必须注明。
qpos/action 的 x,y,z 和 rx,ry,rz 单位不同，但按维度分别 mean/std 可以接受。
不存在 gripper 维度。
如果 qvel 全 0，不要对 qvel 做 std 归一化，或使用 eps 防止除 0。
固定常量维度需要在统计脚本里检测 std < 1e-6，并把 std 置为 1。
异常值建议在统计前先按质量报告过滤，不建议静默裁剪后混入训练。
```

## 12. 训练接口期望

建议第一版训练配置：

```text
policy_class: ACT 或 CNNMLP
输入相机：pool, scan_2d
是否使用多相机：是
是否使用 qvel：不建议第一版使用；如果 loader 必须返回，可保留但模型不输入
是否使用 language：否，当前数据无 language instruction
action chunk size 建议：10-30；scan_2d 为 5 Hz 时，对应约 2-6 秒
预测 horizon：同 action chunk size
batch size 预期：视 GPU 显存和图像 resize 后尺寸决定
```

ACT loader 返回接口可映射为：

```text
image_data:  (2, 3, H, W)
qpos_data:   (6,)
action_data: (chunk_size 或 episode_len, 6)
is_pad:      (chunk_size 或 episode_len,)
```

如果继续使用当前默认的双臂 14 维模型配置，需要先采集或恢复：

```text
/observations/qpos: 14 维关节 + gripper
/observations/qvel: 14 维速度
/action: 14 维目标关节或动作
```

当前这批数据无法凭已有文件恢复这些字段。

## 13. OSS 下载与权限

从现有 `scripts/readme.md` 可确认上传命令：

```bash
ossutil sync /home/shugen/yanjie/ros2_ws/data_collect oss://gpu-ai/lcx/brush_pen/data_collect/
```

已知：

```text
OSS bucket: gpu-ai
endpoint: 未记录，需 OSS 配置或账号确认
数据路径：oss://gpu-ai/lcx/brush_pen/data_collect/
是否公开：未知
需要什么 AK/SK 或 role：未知
目录大小：本地约 7.7G
单个 episode 大小：约 171M - 532M
总大小：约 7.7G
是否有校验文件 md5/sha256：未发现
是否有压缩包：未发现
解压后目录结构：无需解压，原始目录即 YYYY-MM-DD/HH-MM-SS/
```

建议下载命令模板：

```bash
ossutil sync oss://gpu-ai/lcx/brush_pen/data_collect/ /home/shugen/yanjie/ros2_ws/data_collect/
```

## 14. 最小样例

原始 session：

```text
2026-06-15/19-49-26
```

文件结构：

```text
camera_pool/       871 张 JPEG，1280 x 1024，RGB
camera_3d_2d/      222 张 JPEG，1440 x 1080，RGB
robot_state/       tool_pose.csv，4402 行，header 在第 2 行
session_meta.json  capture_3d_2d_hz = 5.0
```

`tool_pose.csv` 前几条有效数据：

```text
timestamp,x,y,z,rx,ry,rz
71366577685,0.1704510450,0.5821776390,0.3413496017,-0.8963723779,-0.4367825389,-2.9163248539
71366577196,0.1704464555,0.5821845531,0.3413592577,-0.8963850737,-0.4367870986,-2.9163320065
71366578109,0.1704473346,0.5821678042,0.3413341045,-0.8963393569,-0.4368024766,-2.9163389206
```

示例图像：

```text
pool:    /home/shugen/yanjie/ros2_ws/data_collect/2026-06-15/19-49-26/camera_pool/71366530114.000001.jpg
scan_2d: /home/shugen/yanjie/ros2_ws/data_collect/2026-06-15/19-49-26/camera_3d_2d/71366861003.000001.jpg
```

## 15. 待确认清单

在写最终转换脚本和 loader 配置前，至少需要确认：

```text
1. 这批数据是否允许用 TCP pose 6D 作为 qpos/action，而不是关节空间。
2. x/y/z 单位是否为 meter，rx/ry/rz 单位是否为 rad。
3. rx/ry/rz 的姿态表示是旋转向量、欧拉角，还是机器人控制器特定格式。
4. action[t] 是否应为 qpos[t+1]，以及 loader 是否应关闭 start_ts - 1 偏移。
5. 24 个 session 哪些是 success，哪些失败或无效。
6. 两路相机是否需要裁剪、旋转、翻转，以及训练目标分辨率。
7. 是否有未落盘的 control_speed、joint state 或 gripper 数据可补充。
```
