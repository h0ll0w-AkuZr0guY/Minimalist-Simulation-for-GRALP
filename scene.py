import math
import random
import pybullet as p
import pybullet_data
from config_loader import Config


class DynamicObstacle:
    def __init__(self, pos):
        self.start_pos = pos
        self.size = [0.3, 0.3, 0.5]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s / 2 for s in self.size])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s / 2 for s in self.size], rgbaColor=[0, 0, 1, 1])
        self.id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
        )

        self.axis = random.choice([0, 1])
        speed_min, speed_max = Config.DYNAMIC_OBSTACLE_SPEED_RANGE
        self.speed = random.uniform(speed_min, speed_max)
        self.range = 2.0
        self.phase = random.uniform(0, 2 * math.pi)

    def update(self, time_t):
        offset = math.sin(time_t * self.speed + self.phase) * self.range
        new_pos = list(self.start_pos)
        new_pos[self.axis] += offset
        p.resetBasePositionAndOrientation(self.id, new_pos, [0, 0, 0, 1])


class Scene:
    def __init__(self):
        self.dynamic_obstacles = []
        self._setup_world()
        self._create_walls()
        self._spawn_static_obstacles()
        self._spawn_dynamic_obstacles()

    def _setup_world(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, Config.GRAVITY)
        p.loadURDF("plane.urdf")

    def _create_walls(self):
        half_map = Config.MAP_SIZE / 2
        h_ext = Config.WALL_HEIGHT / 2
        t_ext = Config.WALL_THICKNESS / 2

        walls_info = [
            ([0, half_map + t_ext], [half_map + 2 * t_ext, t_ext]),
            ([0, -half_map - t_ext], [half_map + 2 * t_ext, t_ext]),
            ([-half_map - t_ext, 0], [t_ext, half_map]),
            ([half_map + t_ext, 0], [t_ext, half_map]),
        ]

        for pos_xy, half_size_xy in walls_info:
            pos = [pos_xy[0], pos_xy[1], h_ext]
            size = [half_size_xy[0], half_size_xy[1], h_ext]

            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.3, 0.3, 0.3, 1])

            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)

    def _spawn_static_obstacles(self):
        half_map = Config.MAP_SIZE / 2 - 1.0
        for _ in range(Config.STATIC_OBSTACLE_COUNT):
            x = random.uniform(-half_map, half_map)
            y = random.uniform(-half_map, half_map)
            if abs(x) < 1.0 and abs(y) < 1.0:
                continue

            size = [random.uniform(0.2, 0.8) for _ in range(3)]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.5, 0.5, 0.5, 1])
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, size[2]],
            )

    def _spawn_dynamic_obstacles(self):
        half_map = Config.MAP_SIZE / 2 - 1.5
        for _ in range(Config.DYNAMIC_OBSTACLE_COUNT):
            x = random.uniform(-half_map, half_map)
            y = random.uniform(-half_map, half_map)
            if abs(x) < 2.0 and abs(y) < 2.0:
                continue
            self.dynamic_obstacles.append(DynamicObstacle([x, y, 0.25]))

    def update(self, time_t):
        for obs in self.dynamic_obstacles:
            obs.update(time_t)
