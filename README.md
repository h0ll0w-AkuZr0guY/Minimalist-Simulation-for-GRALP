# Minimalist Simulation for GRALP

这是一个为 github.com/hnsqdtt/GRALP 准备的极简 PyBullet 仿真环境，用 ONNXRuntime 版 PPO 导航策略（ppo_api/policy.onnx，经 `ppo_api.inference.PPOInference` 调用）做快速验证与可视化。

## 依赖
- Python 3.8+
- pip 包：pybullet、numpy、onnxruntime（CPU）；如需 GPU/TensorRT 请安装 onnxruntime-gpu，对应执行后端在 ppo_api/config.json 的 `execution_provider` 配置
- Windows 用户可直接运行 start.bat；其他平台运行 python task.py 即可

## 快速开始
1) 建议在仓库根目录创建虚拟环境：`python -m venv .venv && .\.venv\Scripts\activate`
2) 安装依赖：`pip install pybullet numpy onnxruntime`（有 CUDA/TensorRT 的机器可改用 `pip install onnxruntime-gpu`）
3) 运行模拟：`python task.py`（或双击 start.bat）
4) 相机操作：右键旋转视角，左键拖拽平移，滚轮缩放；`Ctrl+C` 退出

## 参数配置
- simulation_config.json：物理与场景参数。常用项：
  - TIMESTEP/GRAVITY；MAP_SIZE、WALL_HEIGHT/THICKNESS；STATIC_OBSTACLE_COUNT、DYNAMIC_OBSTACLE_COUNT、DYNAMIC_OBSTACLE_SPEED_RANGE
  - ROBOT_START_POS、ROBOT_RADIUS/HEIGHT/COLOR、OBSTACLE_INFLATION_EXTRA；相机 CAM_* 与鼠标灵敏度；DEBUG_MODE（true 时显示雷达射线）
- ppo_api/config.json：策略相关（供 ONNXRuntime 推理使用）。
  - vx_max/omega_max 与 dt 用于动作限幅及控制周期；
  - patch_meters 作为雷达量程，ray_max_gap 决定射线数：LIDAR_NUM_RAYS = ceil((2π·patch_meters)/ray_max_gap)
  - ckpt_filename 默认 policy.onnx，可改为自定义 ONNX 权重；execution_provider 控制 cpu/cuda/tensorrt 后端
- config_loader.py 会合并两份配置到 Config；LIDAR_RANGE/LIDAR_FOV/LIDAR_NUM_RAYS、CTRL_VX_MAX/CTRL_OMEGA_MAX 等在运行时由此派生，无需手改代码。

## 局部任务点选择机制
- 全局目标：`task._sample_goal` 在地图内随机采样，确保与障碍/墙体有 2.5×车体半径的安全距离。
- 视域构造：`robot.get_lidar_data` 采集 0° 对齐车头的 360° 雷达；`_build_los_points` 按 ROBOT_RADIUS+OBSTACLE_INFLATION_EXTRA 膨胀命中距离，并限制在 patch_meters 内，得到每条射线的可行线段及 PPO 使用的归一化距离。
- 任务折射：`_select_local_target` 将全局目标投影到这些可见线段上，选择距离目标最近的可行点；若完全遮挡则沿当前航向取前向可行距离。
- 控制推理：以局部任务点构造 sin_ref/cos_ref、task_dist，再连同上一帧/上上帧动作送入 `PPOInference.infer`；输出 vx/omega 在 `Robot.apply_control` 中按 CTRL_VX_MAX/CTRL_OMEGA_MAX 限幅后写入 PyBullet。
- 可视化：目标十字、局部连线、碰撞变色和运动轨迹均在 task.py 用 addUserDebugLine/Text 绘制；DEBUG_MODE 为真时额外显示雷达射线。

## 常用调整
- 增减静态/动态障碍或修改速度区间以压测策略。
- 调高 WALL_HEIGHT 或 MAP_SIZE 观察大场景下的视角效果。
- 修改 CAM_DIST/CAM_YAW/CAM_PITCH 设定初始视角，或调节 MOUSE_SENSITIVITY_* 改变交互手感。
