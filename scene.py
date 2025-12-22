import math
import random
import pybullet as p  # PyBullet物理引擎
import pybullet_data  # PyBullet内置模型数据
from config_loader import Config  # 配置加载器，获取全局配置


class DynamicObstacle:
    """动态障碍物类，实现沿x或y轴往复运动的障碍物"""
    
    def __init__(self, pos):
        """初始化动态障碍物
        
        Args:
            pos: 障碍物初始位置 [x, y, z]
        """
        self.start_pos = pos  # 初始位置，作为运动的参考点
        self.size = [0.3, 0.3, 0.5]  # 障碍物尺寸 [长, 宽, 高]
        
        # 创建碰撞形状（使用BOX几何体，halfExtents为尺寸的一半）
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s / 2 for s in self.size])
        # 创建视觉形状（蓝色，透明度1.0）
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s / 2 for s in self.size], rgbaColor=[0, 0, 1, 1])
        
        # 创建多体对象（质量为0，表示固定物体）
        self.id = p.createMultiBody(
            baseMass=0,  # 质量为0，不受重力影响
            baseCollisionShapeIndex=col,  # 碰撞形状索引
            baseVisualShapeIndex=vis,  # 视觉形状索引
            basePosition=pos,  # 初始位置
        )

        # 随机选择运动轴（0: x轴，1: y轴）
        self.axis = random.choice([0, 1])
        # 从配置获取动态障碍物速度范围
        speed_min, speed_max = Config.DYNAMIC_OBSTACLE_SPEED_RANGE
        self.speed = random.uniform(speed_min, speed_max)  # 随机速度
        self.range = 2.0  # 运动范围（偏离初始位置的最大距离）
        self.phase = random.uniform(0, 2 * math.pi)  # 初始相位，避免所有障碍物同步运动

    def update(self, time_t):
        """更新障碍物位置
        
        Args:
            time_t: 当前仿真时间（秒）
        """
        # 使用正弦函数计算偏移量，实现平滑往复运动
        offset = math.sin(time_t * self.speed + self.phase) * self.range
        new_pos = list(self.start_pos)  # 复制初始位置
        new_pos[self.axis] += offset  # 在选定轴上添加偏移
        # 重置障碍物位置和朝向（朝向始终保持不变）
        p.resetBasePositionAndOrientation(self.id, new_pos, [0, 0, 0, 1])


class Scene:
    """场景类，负责创建和管理仿真环境"""
    
    def __init__(self):
        """初始化场景，设置世界、墙壁、静态和动态障碍物"""
        self.dynamic_obstacles = []  # 动态障碍物列表
        self._setup_world()  # 设置物理世界
        self._create_walls()  # 创建边界墙壁
        self._spawn_static_obstacles()  # 生成静态障碍物
        self._spawn_dynamic_obstacles()  # 生成动态障碍物

    def _setup_world(self):
        """设置物理世界参数"""
        # 设置PyBullet数据路径，用于加载内置模型
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # 设置重力（从配置读取）
        p.setGravity(0, 0, Config.GRAVITY)
        # 加载地面模型（plane.urdf是PyBullet内置的平面模型）
        p.loadURDF("plane.urdf")

    def _create_walls(self):
        """创建场景边界墙壁"""
        half_map = Config.MAP_SIZE / 2  # 地图半宽
        h_ext = Config.WALL_HEIGHT / 2  # 墙壁半高
        t_ext = Config.WALL_THICKNESS / 2  # 墙壁半厚

        # 定义四面墙壁的信息：[位置xy, 半尺寸xy]
        walls_info = [
            ([0, half_map + t_ext], [half_map + 2 * t_ext, t_ext]),  # 右墙
            ([0, -half_map - t_ext], [half_map + 2 * t_ext, t_ext]),  # 左墙
            ([-half_map - t_ext, 0], [t_ext, half_map]),  # 前墙
            ([half_map + t_ext, 0], [t_ext, half_map]),  # 后墙
        ]

        # 遍历创建每面墙
        for pos_xy, half_size_xy in walls_info:
            pos = [pos_xy[0], pos_xy[1], h_ext]  # 墙壁位置（z轴为半高，使其底部对齐地面）
            size = [half_size_xy[0], half_size_xy[1], h_ext]  # 墙壁尺寸

            # 创建碰撞形状
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            # 创建视觉形状（灰色，透明度1.0）
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.3, 0.3, 0.3, 1])

            # 创建墙壁对象（质量为0，固定不动）
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)

    def _spawn_static_obstacles(self):
        """生成静态障碍物"""
        half_map = Config.MAP_SIZE / 2 - 1.0  # 地图半宽减去安全距离
        
        # 生成指定数量的静态障碍物
        for _ in range(Config.STATIC_OBSTACLE_COUNT):
            # 随机生成x和y坐标（避开中心区域）
            x = random.uniform(-half_map, half_map)
            y = random.uniform(-half_map, half_map)
            
            # 避免在中心1x1区域生成障碍物
            if abs(x) < 1.0 and abs(y) < 1.0:
                continue

            # 随机生成障碍物尺寸（0.2到0.8之间）
            size = [random.uniform(0.2, 0.8) for _ in range(3)]
            # 创建碰撞形状
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            # 创建视觉形状（灰色，透明度1.0）
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.5, 0.5, 0.5, 1])
            
            # 创建静态障碍物对象
            p.createMultiBody(
                baseMass=0,  # 质量为0，固定不动
                baseCollisionShapeIndex=col,  # 碰撞形状索引
                baseVisualShapeIndex=vis,  # 视觉形状索引
                basePosition=[x, y, size[2]],  # 位置（z轴为size[2]，使底部对齐地面）
            )

    def _spawn_dynamic_obstacles(self):
        """生成动态障碍物"""
        half_map = Config.MAP_SIZE / 2 - 1.5  # 地图半宽减去安全距离
        
        # 生成指定数量的动态障碍物
        for _ in range(Config.DYNAMIC_OBSTACLE_COUNT):
            # 随机生成x和y坐标（避开中心2x2区域）
            x = random.uniform(-half_map, half_map)
            y = random.uniform(-half_map, half_map)
            
            # 避免在中心2x2区域生成障碍物
            if abs(x) < 2.0 and abs(y) < 2.0:
                continue
            
            # 创建动态障碍物并添加到列表
            self.dynamic_obstacles.append(DynamicObstacle([x, y, 0.25]))

    def update(self, time_t):
        """更新场景中所有动态障碍物
        
        Args:
            time_t: 当前仿真时间（秒）
        """
        for obs in self.dynamic_obstacles:
            obs.update(time_t)  # 调用每个动态障碍物的update方法
