import json
import math
import os
from types import SimpleNamespace
from typing import Optional


# 加载PPO配置文件
def load_ppo_config(path: Optional[str] = None) -> dict:
    """加载PPO配置文件
    
    Args:
        path: PPO配置文件路径，如果未提供则使用默认路径
        
    Returns:
        dict: 解析后的PPO配置字典
    """
    base_dir = os.path.dirname(__file__)  # 获取当前文件所在目录
    # 构建PPO配置文件路径，如果未提供path则使用默认路径
    ppo_config_path = path or os.path.join(base_dir, "ppo_api", "config.json")
    # 打开并读取配置文件
    with open(ppo_config_path, "r", encoding="utf-8") as f:
        return json.load(f)  # 返回解析后的配置字典


# 加载和合并仿真配置
def load_simulation_config(
    sim_config_path: Optional[str] = None,
    ppo_config_path: Optional[str] = None,
) -> SimpleNamespace:
    """加载仿真配置并与PPO配置合并，生成最终的仿真配置
    
    Args:
        sim_config_path: 仿真配置文件路径，如果未提供则使用默认路径
        ppo_config_path: PPO配置文件路径，如果未提供则使用默认路径
        
    Returns:
        SimpleNamespace: 包含所有仿真配置的命名空间对象
    """
    base_dir = os.path.dirname(__file__)  # 获取当前文件所在目录
    # 构建仿真配置文件路径，如果未提供sim_config_path则使用默认路径
    config_path = sim_config_path or os.path.join(base_dir, "simulation_config.json")
    # 打开并读取仿真配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 解析后的仿真配置字典

    # 加载PPO配置
    ppo_cfg = load_ppo_config(ppo_config_path)
    # 从PPO配置中提取关键参数
    patch_meters = ppo_cfg.get("patch_meters")  # 补丁大小（米）
    ray_max_gap = ppo_cfg.get("ray_max_gap")  # 最大射线间隔
    vx_max = ppo_cfg.get("vx_max", 1.5)  # 最大线速度，默认1.5
    omega_max = ppo_cfg.get("omega_max", 2.0)  # 最大角速度，默认2.0

    # 检查必要参数是否存在
    if patch_meters is None or ray_max_gap is None:
        raise ValueError("ppo_api config.json must provide patch_meters and ray_max_gap")

    # 设置默认值和计算派生参数
    data.setdefault("LIDAR_FOV", 360)  # 激光雷达视场角，默认360度
    data["LIDAR_RANGE"] = float(patch_meters)  # 激光雷达范围（米）
    # 计算激光雷达射线数量：根据圆周长和最大射线间隔计算
    data["LIDAR_NUM_RAYS"] = int(math.ceil((2 * math.pi * patch_meters) / ray_max_gap))
    data["CTRL_VX_MAX"] = float(vx_max)  # 最大线速度控制限制
    data["CTRL_OMEGA_MAX"] = float(omega_max)  # 最大角速度控制限制
    data.setdefault("OBSTACLE_INFLATION_EXTRA", 0.1)  # 障碍物额外膨胀，默认0.1米

    # 处理动态障碍物速度范围
    speed_range = data.get("DYNAMIC_OBSTACLE_SPEED_RANGE", [0.5, 1.5])  # 默认速度范围
    # 验证速度范围格式
    if not isinstance(speed_range, (list, tuple)) or len(speed_range) != 2:
        raise ValueError("DYNAMIC_OBSTACLE_SPEED_RANGE must be a [min, max] list/tuple")
    # 转换为浮点数并验证范围
    speed_min, speed_max = float(speed_range[0]), float(speed_range[1])
    if speed_min > speed_max:
        raise ValueError("DYNAMIC_OBSTACLE_SPEED_RANGE min must be <= max")
    # 更新数据
    data["DYNAMIC_OBSTACLE_SPEED_RANGE"] = [speed_min, speed_max]

    # 将字典转换为命名空间对象，方便通过点符号访问
    return SimpleNamespace(**data)


# 初始化全局配置对象
Config = load_simulation_config()
