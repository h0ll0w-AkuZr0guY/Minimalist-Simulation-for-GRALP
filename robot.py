import math
import numpy as np
import pybullet as p  # PyBullet物理引擎库
from config_loader import Config  # 配置加载模块


class Robot:
    """机器人类
    
    描述：创建和控制机器人模型，包括机器人主体、速度控制、LIDAR数据获取和航向指示器绘制
    """
    
    def __init__(self):
        """初始化机器人
        
        创建机器人主体，初始化LIDAR射线ID和航向指示器ID列表
        """
        self.id = self._create_robot()  # 创建机器人主体，获取其ID
        self.lidar_ray_ids = []  # 用于存储LIDAR射线调试线的ID
        self.orientation_line_id = None  # 航向指示器主线的ID
        self.orientation_head_ids = [None, None]  # 航向指示器箭头头部的两个线段ID

    def _create_robot(self):
        """创建圆柱形机器人主体
        
        Returns:
            int: 机器人主体的PyBullet ID
        """
        # 创建碰撞形状（圆柱体）
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,  # 几何体类型：圆柱体
            radius=Config.ROBOT_RADIUS,  # 圆柱体半径
            height=Config.ROBOT_HEIGHT,  # 圆柱体高度
        )
        
        # 创建视觉形状（圆柱体）
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,  # 几何体类型：圆柱体
            radius=Config.ROBOT_RADIUS,  # 圆柱体半径
            length=Config.ROBOT_HEIGHT,  # 圆柱体长度（等同于高度）
            rgbaColor=Config.ROBOT_COLOR,  # 圆柱体颜色和透明度
        )

        # 创建多体对象（机器人主体）
        body_id = p.createMultiBody(
            baseMass=10,  # 机器人质量
            baseCollisionShapeIndex=collision_shape,  # 碰撞形状索引
            baseVisualShapeIndex=visual_shape,  # 视觉形状索引
            basePosition=Config.ROBOT_START_POS,  # 初始位置
        )
        return body_id

    def apply_control(self, linear_vel, angular_vel):
        """应用速度控制命令
        
        Args:
            linear_vel: 线速度命令（m/s）
            angular_vel: 角速度命令（rad/s）
        """
        # 获取配置中的速度限制
        vx_max = getattr(Config, "CTRL_VX_MAX", None)
        om_max = getattr(Config, "CTRL_OMEGA_MAX", None)
        
        # 应用线速度限制
        if vx_max is not None:
            linear_vel = max(-vx_max, min(vx_max, linear_vel))
        
        # 应用角速度限制
        if om_max is not None:
            angular_vel = max(-om_max, min(om_max, angular_vel))

        # 获取机器人当前位置和姿态
        _, orn = p.getBasePositionAndOrientation(self.id)
        # 将四元数转换为欧拉角，获取偏航角（yaw）
        _, _, yaw = p.getEulerFromQuaternion(orn)

        # 将线速度转换为世界坐标系下的速度分量
        vel_x = linear_vel * math.cos(yaw)
        vel_y = linear_vel * math.sin(yaw)

        # 应用速度控制
        p.resetBaseVelocity(
            self.id, 
            linearVelocity=[vel_x, vel_y, 0],  # 线速度分量（z轴为0）
            angularVelocity=[0, 0, angular_vel]  # 角速度分量（仅z轴有值）
        )

    def get_lidar_data(self):
        """获取LIDAR数据
        
        Returns:
            np.array: LIDAR距离数据（米），与机器人航向对齐
        """
        # 获取机器人当前位置和姿态
        pos, orn = p.getBasePositionAndOrientation(self.id)
        # 将四元数转换为欧拉角，获取偏航角（yaw）
        _, _, yaw = p.getEulerFromQuaternion(orn)

        # 计算LIDAR的安装位置（在机器人顶部）
        lidar_pos = [pos[0], pos[1], pos[2] + Config.LIDAR_HEIGHT]

        # 初始化射线起点和终点列表
        ray_froms = []
        ray_tos = []

        # 计算LIDAR射线参数
        fov_rad = np.deg2rad(Config.LIDAR_FOV)  # 视场角转换为弧度
        angle_step = fov_rad / Config.LIDAR_NUM_RAYS  # 相邻射线的角度间隔
        start_angle = yaw  # 起始角度为机器人当前航向

        # 生成所有LIDAR射线
        for i in range(Config.LIDAR_NUM_RAYS):
            angle = start_angle + i * angle_step  # 计算当前射线的角度
            # 计算射线终点坐标
            end_x = lidar_pos[0] + Config.LIDAR_RANGE * math.cos(angle)
            end_y = lidar_pos[1] + Config.LIDAR_RANGE * math.sin(angle)
            end_z = lidar_pos[2]
            # 添加到射线列表
            ray_froms.append(lidar_pos)
            ray_tos.append([end_x, end_y, end_z])

        # 执行批量射线检测
        results = p.rayTestBatch(ray_froms, ray_tos)

        # 处理检测结果
        ranges = []
        # 如果处于调试模式，清除之前的LIDAR调试线
        if Config.DEBUG_MODE:
            for dbg_id in self.lidar_ray_ids:
                p.removeUserDebugItem(dbg_id)
            self.lidar_ray_ids.clear()

        # 遍历处理每条射线的检测结果
        for i, res in enumerate(results):
            hit_fraction = res[2]  # 碰撞点占射线总长度的比例
            dist = hit_fraction * Config.LIDAR_RANGE  # 计算实际距离
            ranges.append(dist)  # 添加到距离列表

            # 如果处于调试模式，绘制LIDAR射线
            if Config.DEBUG_MODE:
                # 碰撞射线显示为红色，未碰撞显示为绿色
                color = [1, 0, 0] if hit_fraction < 1.0 else [0, 1, 0]
                # 计算碰撞点坐标
                hit_pos = [
                    ray_froms[i][0] + (ray_tos[i][0] - ray_froms[i][0]) * hit_fraction,
                    ray_froms[i][1] + (ray_tos[i][1] - ray_froms[i][1]) * hit_fraction,
                    ray_froms[i][2],
                ]
                # 添加调试线
                dbg_id = p.addUserDebugLine(ray_froms[i], hit_pos, color, lifeTime=0.1)
                self.lidar_ray_ids.append(dbg_id)

        return np.array(ranges)  # 返回距离数据数组

    def update_heading_indicator(self):
        """绘制航向指示器（如果启用）
        
        在机器人主体上绘制一个箭头，指示当前航向
        """
        # 如果未启用航向指示器，直接返回
        if not Config.SHOW_HEADING_INDICATOR:
            return

        # 获取机器人当前位置和姿态
        pos, orn = p.getBasePositionAndOrientation(self.id)
        # 将四元数转换为欧拉角，获取偏航角（yaw）
        _, _, yaw = p.getEulerFromQuaternion(orn)

        # 计算箭头起点和终点
        start = [pos[0], pos[1], pos[2]]  # 箭头起点（机器人中心）
        dir_x = math.cos(yaw)  # 航向方向的x分量
        dir_y = math.sin(yaw)  # 航向方向的y分量
        # 箭头终点（箭头尖端）
        tip = [
            start[0] + Config.HEADING_LINE_LENGTH * dir_x,
            start[1] + Config.HEADING_LINE_LENGTH * dir_y,
            start[2],
        ]

        # 获取配置中的颜色和宽度
        color = Config.HEADING_LINE_COLOR
        width = Config.HEADING_LINE_WIDTH
        
        # 绘制箭头主线
        self.orientation_line_id = p.addUserDebugLine(
            start,
            tip,
            color,
            lineWidth=width,
            lifeTime=0,  # 永久显示
            replaceItemUniqueId=self.orientation_line_id if self.orientation_line_id is not None else -1,  # 替换已存在的线
        )

        # 绘制箭头头部
        head_len = Config.HEADING_HEAD_LENGTH  # 箭头头部长度
        head_angle = Config.HEADING_HEAD_ANGLE  # 箭头头部角度
        # 绘制箭头头部的两个侧边
        for idx, sign in enumerate((-1, 1)):
            head_dir = yaw + sign * head_angle  # 计算侧边方向
            # 计算侧边终点
            head_end = [
                tip[0] - head_len * math.cos(head_dir),
                tip[1] - head_len * math.sin(head_dir),
                tip[2],
            ]
            # 获取要替换的线ID
            replace_id = self.orientation_head_ids[idx] if self.orientation_head_ids[idx] is not None else -1
            # 绘制箭头头部侧边
            self.orientation_head_ids[idx] = p.addUserDebugLine(
                tip,
                head_end,
                color,
                lineWidth=width,
                lifeTime=0,  # 永久显示
                replaceItemUniqueId=replace_id,  # 替换已存在的线
            )
