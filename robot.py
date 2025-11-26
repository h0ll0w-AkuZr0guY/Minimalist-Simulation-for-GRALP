import math
import numpy as np
import pybullet as p
from config_loader import Config


class Robot:
    def __init__(self):
        self.id = self._create_robot()
        self.lidar_ray_ids = []
        self.orientation_line_id = None
        self.orientation_head_ids = [None, None]

    def _create_robot(self):
        """Create cylindrical robot body."""
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=Config.ROBOT_RADIUS,
            height=Config.ROBOT_HEIGHT,
        )
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=Config.ROBOT_RADIUS,
            length=Config.ROBOT_HEIGHT,
            rgbaColor=Config.ROBOT_COLOR,
        )

        body_id = p.createMultiBody(
            baseMass=10,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=Config.ROBOT_START_POS,
        )
        return body_id

    def apply_control(self, linear_vel, angular_vel):
        """Apply velocity command with config limits."""
        vx_max = getattr(Config, "CTRL_VX_MAX", None)
        om_max = getattr(Config, "CTRL_OMEGA_MAX", None)
        if vx_max is not None:
            linear_vel = max(-vx_max, min(vx_max, linear_vel))
        if om_max is not None:
            angular_vel = max(-om_max, min(om_max, angular_vel))

        _, orn = p.getBasePositionAndOrientation(self.id)
        _, _, yaw = p.getEulerFromQuaternion(orn)

        vel_x = linear_vel * math.cos(yaw)
        vel_y = linear_vel * math.sin(yaw)

        p.resetBaseVelocity(self.id, linearVelocity=[vel_x, vel_y, 0], angularVelocity=[0, 0, angular_vel])

    def get_lidar_data(self):
        """Return lidar ranges (m) aligned with robot heading."""
        pos, orn = p.getBasePositionAndOrientation(self.id)
        _, _, yaw = p.getEulerFromQuaternion(orn)

        lidar_pos = [pos[0], pos[1], pos[2] + Config.LIDAR_HEIGHT]

        ray_froms = []
        ray_tos = []

        fov_rad = np.deg2rad(Config.LIDAR_FOV)
        angle_step = fov_rad / Config.LIDAR_NUM_RAYS
        start_angle = yaw

        for i in range(Config.LIDAR_NUM_RAYS):
            angle = start_angle + i * angle_step
            end_x = lidar_pos[0] + Config.LIDAR_RANGE * math.cos(angle)
            end_y = lidar_pos[1] + Config.LIDAR_RANGE * math.sin(angle)
            end_z = lidar_pos[2]
            ray_froms.append(lidar_pos)
            ray_tos.append([end_x, end_y, end_z])

        results = p.rayTestBatch(ray_froms, ray_tos)

        ranges = []
        if Config.DEBUG_MODE:
            for dbg_id in self.lidar_ray_ids:
                p.removeUserDebugItem(dbg_id)
            self.lidar_ray_ids.clear()

        for i, res in enumerate(results):
            hit_fraction = res[2]
            dist = hit_fraction * Config.LIDAR_RANGE
            ranges.append(dist)

            if Config.DEBUG_MODE:
                color = [1, 0, 0] if hit_fraction < 1.0 else [0, 1, 0]
                hit_pos = [
                    ray_froms[i][0] + (ray_tos[i][0] - ray_froms[i][0]) * hit_fraction,
                    ray_froms[i][1] + (ray_tos[i][1] - ray_froms[i][1]) * hit_fraction,
                    ray_froms[i][2],
                ]
                dbg_id = p.addUserDebugLine(ray_froms[i], hit_pos, color, lifeTime=0.1)
                self.lidar_ray_ids.append(dbg_id)

        return np.array(ranges)

    def update_heading_indicator(self):
        """Draw heading indicator if enabled."""
        if not Config.SHOW_HEADING_INDICATOR:
            return

        pos, orn = p.getBasePositionAndOrientation(self.id)
        _, _, yaw = p.getEulerFromQuaternion(orn)

        start = [pos[0], pos[1], pos[2]]
        dir_x = math.cos(yaw)
        dir_y = math.sin(yaw)
        tip = [
            start[0] + Config.HEADING_LINE_LENGTH * dir_x,
            start[1] + Config.HEADING_LINE_LENGTH * dir_y,
            start[2],
        ]

        color = Config.HEADING_LINE_COLOR
        width = Config.HEADING_LINE_WIDTH
        self.orientation_line_id = p.addUserDebugLine(
            start,
            tip,
            color,
            lineWidth=width,
            lifeTime=0,
            replaceItemUniqueId=self.orientation_line_id if self.orientation_line_id is not None else -1,
        )

        head_len = Config.HEADING_HEAD_LENGTH
        head_angle = Config.HEADING_HEAD_ANGLE
        for idx, sign in enumerate((-1, 1)):
            head_dir = yaw + sign * head_angle
            head_end = [
                tip[0] - head_len * math.cos(head_dir),
                tip[1] - head_len * math.sin(head_dir),
                tip[2],
            ]
            replace_id = self.orientation_head_ids[idx] if self.orientation_head_ids[idx] is not None else -1
            self.orientation_head_ids[idx] = p.addUserDebugLine(
                tip,
                head_end,
                color,
                lineWidth=width,
                lifeTime=0,
                replaceItemUniqueId=replace_id,
            )
