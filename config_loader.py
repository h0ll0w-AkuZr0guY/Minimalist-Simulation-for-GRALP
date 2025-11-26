import json
import math
import os
from types import SimpleNamespace
from typing import Optional


def load_ppo_config(path: Optional[str] = None) -> dict:
    """Load PPO config values."""
    base_dir = os.path.dirname(__file__)
    ppo_config_path = path or os.path.join(base_dir, "ppo_api", "config.json")
    with open(ppo_config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_simulation_config(
    sim_config_path: Optional[str] = None,
    ppo_config_path: Optional[str] = None,
) -> SimpleNamespace:
    """Merge simulation config with PPO config and derive lidar/control fields."""
    base_dir = os.path.dirname(__file__)
    config_path = sim_config_path or os.path.join(base_dir, "simulation_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ppo_cfg = load_ppo_config(ppo_config_path)
    patch_meters = ppo_cfg.get("patch_meters")
    ray_max_gap = ppo_cfg.get("ray_max_gap")
    vx_max = ppo_cfg.get("vx_max", 1.5)
    omega_max = ppo_cfg.get("omega_max", 2.0)

    if patch_meters is None or ray_max_gap is None:
        raise ValueError("ppo_api config.json must provide patch_meters and ray_max_gap")

    data.setdefault("LIDAR_FOV", 360)
    data["LIDAR_RANGE"] = float(patch_meters)
    data["LIDAR_NUM_RAYS"] = int(math.ceil((2 * math.pi * patch_meters) / ray_max_gap))
    data["CTRL_VX_MAX"] = float(vx_max)
    data["CTRL_OMEGA_MAX"] = float(omega_max)
    data.setdefault("OBSTACLE_INFLATION_EXTRA", 0.1)

    speed_range = data.get("DYNAMIC_OBSTACLE_SPEED_RANGE", [0.5, 1.5])
    if not isinstance(speed_range, (list, tuple)) or len(speed_range) != 2:
        raise ValueError("DYNAMIC_OBSTACLE_SPEED_RANGE must be a [min, max] list/tuple")
    speed_min, speed_max = float(speed_range[0]), float(speed_range[1])
    if speed_min > speed_max:
        raise ValueError("DYNAMIC_OBSTACLE_SPEED_RANGE min must be <= max")
    data["DYNAMIC_OBSTACLE_SPEED_RANGE"] = [speed_min, speed_max]

    return SimpleNamespace(**data)


Config = load_simulation_config()
