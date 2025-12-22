# -*- coding: utf-8 -*-
"""
项目主入口文件，负责管理整个仿真流程和PPO推理的集成
"""

import math
import random
import time
from typing import List, Sequence, Tuple

import numpy as np
import pybullet as p

from camera_ctrl import CameraController
from config_loader import Config
from ppo_api.inference import PPOInference
from robot import Robot
from scene import Scene


# 类型定义：2D向量和3D向量
Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


def _body_name(body_id: int) -> str:
    """
    获取物理引擎中物体的名称
    
    Args:
        body_id: 物体的唯一标识符
    
    Returns:
        物体名称，若获取失败则返回空字符串
    """
    try:
        info = p.getBodyInfo(body_id)
    except Exception:
        return ""
    if not info or len(info) < 2:
        return ""
    try:
        return info[1].decode("utf-8")
    except Exception:
        return ""


def _point_to_aabb_distance(pt: Vec3, aabb: Tuple[Vec3, Vec3]) -> float:
    """
    计算点到AABB包围盒的最短距离
    
    Args:
        pt: 三维点坐标
        aabb: AABB包围盒，格式为((xmin, ymin, zmin), (xmax, ymax, zmax))
    
    Returns:
        点到包围盒的最短距离
    """
    (xmin, ymin, zmin), (xmax, ymax, zmax) = aabb
    dx = max(xmin - pt[0], 0.0, pt[0] - xmax)
    dy = max(ymin - pt[1], 0.0, pt[1] - ymax)
    dz = max(zmin - pt[2], 0.0, pt[2] - zmax)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _collect_obstacle_aabbs(robot_id: int) -> List[Tuple[int, Tuple[Vec3, Vec3]]]:
    """
    收集场景中所有障碍物的AABB包围盒
    
    Args:
        robot_id: 机器人的物体ID，用于排除机器人自身
    
    Returns:
        障碍物ID和对应AABB包围盒的列表
    """
    aabbs: List[Tuple[int, Tuple[Vec3, Vec3]]] = []
    for bid in range(p.getNumBodies()):
        if bid == robot_id:  # 排除机器人自身
            continue
        name = _body_name(bid)
        if "plane" in name:  # 排除地面
            continue
        try:
            aabb = p.getAABB(bid)
        except Exception:
            continue
        if aabb is None or len(aabb) != 2:
            continue
        aabbs.append((bid, aabb))
    return aabbs


def _is_free(candidate_xy: Vec2, robot_id: int, clearance: float) -> bool:
    """
    检查候选位置是否为无障碍区域
    
    Args:
        candidate_xy: 候选位置的二维坐标
        robot_id: 机器人ID
        clearance: 安全距离阈值
    
    Returns:
        若位置安全返回True，否则返回False
    """
    aabbs = _collect_obstacle_aabbs(robot_id)
    pt3 = (candidate_xy[0], candidate_xy[1], 0.0)
    for _, aabb in aabbs:
        if _point_to_aabb_distance(pt3, aabb) < clearance:
            return False
    return True


def _sample_goal(robot_id: int, clearance: float) -> Vec3:
    """
    随机采样一个可行的目标位置
    
    Args:
        robot_id: 机器人ID
        clearance: 安全距离
    
    Returns:
        采样到的目标位置
    
    Raises:
        RuntimeError: 若无法采样到可行位置
    """
    half_map = Config.MAP_SIZE / 2.0 - clearance
    for _ in range(200):  # 尝试200次采样
        cand = (
            random.uniform(-half_map, half_map),
            random.uniform(-half_map, half_map),
            0.0,
        )
        if _is_free((cand[0], cand[1]), robot_id, clearance):
            return cand
    raise RuntimeError("Failed to sample a collision-free goal position.")


def _draw_goal_marker(goal: Vec3, existing_ids: Sequence[int]) -> List[int]:
    """
    在仿真中绘制目标位置标记
    
    Args:
        goal: 目标位置
        existing_ids: 已存在的调试项ID列表，将被移除
    
    Returns:
        新创建的调试项ID列表
    """
    # 移除旧标记
    for gid in existing_ids:
        try:
            p.removeUserDebugItem(gid)
        except Exception:
            pass
    
    # 设置标记颜色和大小
    color = [0.2, 1.0, 0.6]  # 浅绿色
    arm = 0.35  # 十字臂长度
    height = 0.05  # 标记高度
    center = [goal[0], goal[1], height]
    
    # 创建十字臂位置列表
    arms = [
        ([-arm, 0, 0], [arm, 0, 0]),  # 横向臂
        ([0, -arm, 0], [0, arm, 0]),  # 纵向臂
    ]
    
    ids: List[int] = []
    # 绘制十字标记
    for a_start, a_end in arms:
        start = [center[0] + a_start[0], center[1] + a_start[1], center[2] + a_start[2]]
        end = [center[0] + a_end[0], center[1] + a_end[1], center[2] + a_end[2]]
        ids.append(p.addUserDebugLine(start, end, color, lineWidth=3, lifeTime=0))
    
    # 绘制目标点到地面的连接线
    ids.append(
        p.addUserDebugLine(
            [goal[0], goal[1], 0.0],
            [goal[0], goal[1], 0.6],
            color,
            lineWidth=3,
            lifeTime=0,
        )
    )
    
    # 添加"GOAL"文本标记
    ids.append(
        p.addUserDebugText(
            "GOAL",
            [goal[0], goal[1], 0.7],
            textColorRGB=color,
            textSize=1.5,
            lifeTime=0,
        )
    )
    return ids


def _world_pose(body_id: int) -> Tuple[Vec3, float]:
    """
    获取物体在世界坐标系中的位置和偏航角
    
    Args:
        body_id: 物体ID
    
    Returns:
        (位置, 偏航角)元组
    """
    pos, orn = p.getBasePositionAndOrientation(body_id)
    _, _, yaw = p.getEulerFromQuaternion(orn)  # 仅返回偏航角
    return (float(pos[0]), float(pos[1]), float(pos[2])), float(yaw)


def _build_los_points(
    pos: Vec3, yaw: float, raw_ranges: np.ndarray, patch_limit: float
) -> Tuple[np.ndarray, List[Tuple[float, float, float, float, float]]]:
    """
    构建视线(LOS)点，用于障碍物检测和路径规划
    
    Args:
        pos: 机器人位置
        yaw: 机器人偏航角
        raw_ranges: LIDAR原始距离数据
        patch_limit: 最大作用范围
    
    Returns:
        (调整后的距离数据, LOS点列表)元组
    """
    # 计算LIDAR参数
    fov_rad = math.radians(Config.LIDAR_FOV)  # 转换为弧度
    angle_step = fov_rad / Config.LIDAR_NUM_RAYS  # 每条射线的角度增量
    raw_ranges = np.asarray(raw_ranges, dtype=np.float32)
    
    # 预计算每个LIDAR射线的方向向量
    dir_cache = []
    for idx in range(Config.LIDAR_NUM_RAYS):
        ang = yaw + idx * angle_step
        dir_cache.append((math.cos(ang), math.sin(ang)))

    def _dilate_ranges(inflate_radius: float) -> np.ndarray:
        """
        膨胀距离数据，用于障碍物膨胀
        
        Args:
            inflate_radius: 膨胀半径
        
        Returns:
            膨胀后的距离数据
        """
        dilated = np.minimum(raw_ranges, patch_limit).astype(np.float32)
        if inflate_radius <= 0.0:
            return dilated

        rx, ry, _ = pos
        radius_sq = inflate_radius * inflate_radius

        # 对每条LIDAR射线进行处理
        for i, hit_dist in enumerate(raw_ranges):
            hit_dist = float(hit_dist)
            if not math.isfinite(hit_dist) or hit_dist >= patch_limit - 1e-6:
                continue
            dir_ix, dir_iy = dir_cache[i]
            hx = rx + hit_dist * dir_ix
            hy = ry + hit_dist * dir_iy
            vec_x = hx - rx
            vec_y = hy - ry
            base_len_sq = vec_x * vec_x + vec_y * vec_y

            # 计算膨胀影响
            for j, (dir_jx, dir_jy) in enumerate(dir_cache):
                proj = vec_x * dir_jx + vec_y * dir_jy  # 沿射线j的有符号距离
                if proj < 0.0:
                    continue
                cross_sq = base_len_sq - proj * proj  # 射线j到撞击点的横向距离平方
                if cross_sq >= radius_sq:
                    continue
                offset = math.sqrt(max(0.0, radius_sq - cross_sq))
                inter = proj - offset  # 沿射线j与膨胀圆盘的第一个交点
                if inter < dilated[j]:
                    dilated[j] = inter
        return dilated

    # PPO使用机器人半径进行膨胀；LOS添加额外膨胀
    rx, ry, _ = pos
    ppo_dilated = _dilate_ranges(max(0.0, Config.ROBOT_RADIUS))
    los_dilated = (
        ppo_dilated
        if math.isclose(
            Config.ROBOT_RADIUS, Config.ROBOT_RADIUS + Config.OBSTACLE_INFLATION_EXTRA, rel_tol=0.0, abs_tol=1e-9
        )
        else _dilate_ranges(max(0.0, Config.ROBOT_RADIUS + Config.OBSTACLE_INFLATION_EXTRA))
    )

    # 构建LOS点列表
    los_points: List[Tuple[float, float, float, float, float]] = []
    adjusted = []
    for idx, dist in enumerate(los_dilated):
        dist = min(float(dist), patch_limit)
        dir_x, dir_y = dir_cache[idx]
        lx = rx + dist * dir_x
        ly = ry + dist * dir_y
        los_points.append((lx, ly, dist, dir_x, dir_y))
    for dist in ppo_dilated:
        adjusted.append(min(float(dist), patch_limit))
    return np.asarray(adjusted, dtype=np.float32), los_points


def _select_local_target(
    los_points: List[Tuple[float, float, float, float, float]], robot_pos: Vec3, robot_yaw: float, goal_xy: Vec2
) -> Tuple[float, float, float]:
    """
    选择局部目标点，用于机器人导航
    
    Args:
        los_points: LOS点列表
        robot_pos: 机器人位置
        robot_yaw: 机器人偏航角
        goal_xy: 全局目标位置
    
    Returns:
        局部目标点坐标
    """
    if not los_points:
        return (goal_xy[0], goal_xy[1], 0.0)
    
    gx, gy = goal_xy
    rx, ry, _ = robot_pos

    # 计算机器人当前朝向的单位向量
    heading_x = math.cos(robot_yaw)
    heading_y = math.sin(robot_yaw)

    # 检查是否有障碍物重叠
    overlap = None
    for _, _, dist, dir_x, dir_y in los_points:
        if dist < 0.0 and (overlap is None or dist > overlap[0]):
            overlap = (dist, dir_x, dir_y)

    # 若有重叠，选择最佳方向
    if overlap is not None:
        best_dir = None
        best_dir_dist = None
        best_dot = -float("inf")
        for _, _, dist, dir_x, dir_y in los_points:
            if dist > 0.0:
                # 计算方向与机器人朝向的点积，选择最接近前方的方向
                dot = dir_x * heading_x + dir_y * heading_y
                if dot > best_dot:
                    best_dot = dot
                    best_dir = (dir_x, dir_y)
                    best_dir_dist = dist
        if best_dir is None:
            best_dir = (heading_x, heading_y)
            best_dir_dist = abs(overlap[0])
        return (rx + best_dir_dist * best_dir[0], ry + best_dir_dist * best_dir[1], best_dir_dist)

    # 若无重叠，选择最接近全局目标的点
    best_point = None
    best_d2 = float("inf")  # 最小距离平方
    for _, _, dist, dir_x, dir_y in los_points:
        gd_x = gx - rx
        gd_y = gy - ry
        t = gd_x * dir_x + gd_y * dir_y  # 目标方向与射线方向的点积
        t = max(0.0, min(dist, t))  # 限制在有效范围内
        px = rx + t * dir_x
        py = ry + t * dir_y
        d2 = (px - gx) ** 2 + (py - gy) ** 2  # 计算距离平方
        if d2 < best_d2:
            best_d2 = d2
            best_point = (px, py, t)
    return best_point if best_point is not None else (goal_xy[0], goal_xy[1], 0.0)


def _is_plane_body(body_id: int) -> bool:
    """
    检查物体是否为地面
    
    Args:
        body_id: 物体ID
    
    Returns:
        若为地面返回True，否则返回False
    """
    try:
        name = p.getBodyInfo(body_id)[1].decode("utf-8")
    except Exception:
        return False
    return "plane" in name.lower()


def _direction_to_target(robot_pos: Vec3, robot_yaw: float, target: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    计算机器人到目标的方向向量（相对于机器人坐标系）
    
    Args:
        robot_pos: 机器人位置
        robot_yaw: 机器人偏航角
        target: 目标位置
    
    Returns:
        (sin_ref, cos_ref, dist)元组，其中sin_ref和cos_ref是目标方向的正弦和余弦值
    """
    dx = target[0] - robot_pos[0]
    dy = target[1] - robot_pos[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0, 1.0, 0.0  # 目标在机器人位置
    
    # 计算目标相对于机器人的角度，并转换到机器人坐标系
    ang = math.atan2(dy, dx) - robot_yaw
    cos_ref = math.cos(ang)
    sin_ref = math.sin(ang)
    return sin_ref, cos_ref, dist


def main() -> None:
    """
    主函数，运行整个仿真系统
    """
    # 初始化PyBullet仿真
    p.connect(p.GUI)  # 连接到GUI模式
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 禁用默认GUI
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)  # 禁用鼠标拾取

    # 初始化各个模块
    scene = Scene()  # 场景管理
    robot = Robot()  # 机器人模型
    cam_ctrl = CameraController()  # 相机控制
    ppo = PPOInference()  # PPO推理

    # 初始化目标位置
    clearance = Config.ROBOT_RADIUS * 2.5  # 目标点的安全距离
    goal = _sample_goal(robot.id, clearance)  # 采样目标点
    goal_dbg = _draw_goal_marker(goal, [])  # 绘制目标标记

    # 初始化控制变量
    prev_vx = 0.0  # 前一时刻线速度
    prev_omega = 0.0  # 前一时刻角速度
    prev_prev_vx = 0.0  # 前前一时刻线速度
    prev_prev_omega = 0.0  # 前前一时刻角速度
    hold_vx = 0.0  # 当前保持的线速度
    hold_omega = 0.0  # 当前保持的角速度
    control_interval = float(ppo.cfg.dt)  # 控制更新间隔
    next_control_time = 0.0  # 下一次控制更新时间

    # 调试可视化变量
    local_dbg: List[int] = []  # 局部目标调试项
    trail_dbg: List[int] = []  # 轨迹调试项
    trail_max_len = 2000  # 最大轨迹长度
    robot_is_green = False  # 机器人碰撞状态标记

    t = 0.0  # 仿真时间
    try:
        while True:
            # 按PPO的时间步长运行控制
            while t + 1e-9 >= next_control_time:
                # 获取机器人状态
                robot_pos, yaw = _world_pose(robot.id)
                raw_lidar = robot.get_lidar_data()
                ranges = np.asarray(raw_lidar, dtype=np.float32)
                
                # 构建LOS点
                adjusted_ranges, los_points = _build_los_points(robot_pos, yaw, ranges, ppo.cfg.patch_meters)

                # 选择局部目标
                local_target = _select_local_target(los_points, robot_pos, yaw, (goal[0], goal[1]))
                
                # 计算目标方向
                sin_ref, cos_ref, task_dist = _direction_to_target(robot_pos, yaw, local_target)
                task_dist = min(task_dist, ppo.cfg.patch_meters)

                # 使用PPO模型推理控制指令
                try:
                    action = ppo.infer(
                        rays_m=adjusted_ranges,
                        sin_ref=sin_ref,
                        cos_ref=cos_ref,
                        prev_vx=prev_vx,
                        prev_omega=prev_omega,
                        prev_prev_vx=prev_prev_vx,
                        prev_prev_omega=prev_prev_omega,
                        task_dist=task_dist,
                        deterministic=True,
                    )
                    # 更新历史控制指令
                    prev_prev_vx, prev_prev_omega = prev_vx, prev_omega
                    prev_vx = float(action[0])
                    prev_omega = float(action[1])
                    hold_vx, hold_omega = prev_vx, prev_omega
                except Exception as exc:
                    print(f"[warn] PPO inference failed: {exc}")
                    hold_vx, hold_omega = 0.0, 0.0  # 控制失败时停止机器人

                next_control_time += control_interval

                # 绘制局部目标线
                for did in local_dbg:
                    try:
                        p.removeUserDebugItem(did)
                    except Exception:
                        pass
                local_dbg = [
                    p.addUserDebugLine(
                        [robot_pos[0], robot_pos[1], robot_pos[2] + 0.05],
                        [local_target[0], local_target[1], robot_pos[2] + 0.05],
                        [1, 0.6, 0.1],  # 橙色
                        lifeTime=control_interval,
                    )
                ]

                # 检查是否到达目标
                goal_dist = math.hypot(goal[0] - robot_pos[0], goal[1] - robot_pos[1])
                if goal_dist <= max(0.5, Config.ROBOT_RADIUS * 1.5):
                    # 到达目标，生成新目标
                    goal = _sample_goal(robot.id, clearance)
                    goal_dbg = _draw_goal_marker(goal, goal_dbg)

            # 应用控制指令到机器人
            robot.apply_control(hold_vx, hold_omega)

            # 碰撞检测与高亮
            collided = False
            for cp in p.getContactPoints(bodyA=robot.id):
                other = cp[2]
                if other == robot.id or _is_plane_body(other):
                    continue
                collided = True
                break
            if collided != robot_is_green:
                # 碰撞时机器人变绿色，否则恢复原始颜色
                target_color = [0.1, 0.9, 0.1, 1.0] if collided else Config.ROBOT_COLOR
                p.changeVisualShape(robot.id, -1, rgbaColor=target_color)
                robot_is_green = collided

            # 绘制轨迹
            robot_pos, _ = _world_pose(robot.id)
            speed_mag = min(math.hypot(hold_vx, hold_omega), 1.0)  # 速度大小，归一化
            base_color = np.array([0.2, 0.8, 1.0], dtype=np.float32)  # 基础颜色
            color = (base_color * (0.3 + 0.7 * speed_mag)).tolist()  # 根据速度调整颜色亮度
            
            # 初始化轨迹起点
            if "last_trail_pos" not in locals():
                last_trail_pos = robot_pos
            else:
                # 添加轨迹段
                trail_dbg.append(
                    p.addUserDebugLine(
                        [last_trail_pos[0], last_trail_pos[1], last_trail_pos[2] + 0.02],
                        [robot_pos[0], robot_pos[1], robot_pos[2] + 0.02],
                        color,
                        lifeTime=0,
                        lineWidth=2,
                    )
                )
                last_trail_pos = robot_pos
                
                # 限制轨迹长度，避免内存溢出
                if len(trail_dbg) > trail_max_len:
                    try:
                        p.removeUserDebugItem(trail_dbg.pop(0))
                    except Exception:
                        pass

            # 更新场景、相机和仿真
            scene.update(t)
            cam_ctrl.update()
            p.stepSimulation()  # 步进仿真
            robot.update_heading_indicator()  # 更新机器人朝向指示器

            # 控制仿真速度
            time.sleep(Config.TIMESTEP)
            t += Config.TIMESTEP

    except KeyboardInterrupt:
        # 捕获键盘中断，优雅退出
        pass
    finally:
        p.disconnect()  # 断开PyBullet连接


if __name__ == "__main__":
    main()  # 运行主函数
