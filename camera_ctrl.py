import math
import pybullet as p
from config_loader import Config


class CameraController:
    def __init__(self):
        self.distance = Config.CAM_DIST
        self.yaw = Config.CAM_YAW
        self.pitch = Config.CAM_PITCH
        self.target_pos = list(Config.CAM_TARGET)

        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.first_input = True

        self.left_button_down = False
        self.right_button_down = False

        self.update_camera()

    def update(self):
        events = p.getMouseEvents()

        if not events:
            self.sync_camera_dist()
            self.update_camera()
            return

        for e in events:
            event_type, curr_x, curr_y, button_index, button_state = e

            if self.first_input:
                self.last_mouse_x = curr_x
                self.last_mouse_y = curr_y
                self.first_input = False
                continue

            if event_type == 2:
                is_down = button_state in (1, 2, 3)

                if button_index == 0:
                    self.left_button_down = is_down
                    if is_down:
                        self.last_mouse_x = curr_x
                        self.last_mouse_y = curr_y

                if button_index == 2:
                    self.right_button_down = is_down
                    if is_down:
                        self.last_mouse_x = curr_x
                        self.last_mouse_y = curr_y

            elif event_type == 1:
                dx = curr_x - self.last_mouse_x
                dy = curr_y - self.last_mouse_y

                if self.right_button_down:
                    self.yaw -= dx * Config.MOUSE_SENSITIVITY_ROTATE
                    self.pitch -= dy * Config.MOUSE_SENSITIVITY_ROTATE
                    self.pitch = max(-89, min(-1, self.pitch))

                if self.left_button_down:
                    pan_sens = Config.MOUSE_SENSITIVITY_PAN * self.distance
                    yaw_rad = math.radians(self.yaw)
                    dx_fix = -dx  # keep drag direction intuitive

                    d_world_x = (dx_fix * math.cos(yaw_rad) - dy * math.sin(yaw_rad)) * pan_sens
                    d_world_y = (dx_fix * math.sin(yaw_rad) + dy * math.cos(yaw_rad)) * pan_sens

                    self.target_pos[0] += d_world_x
                    self.target_pos[1] += d_world_y

                self.last_mouse_x = curr_x
                self.last_mouse_y = curr_y

        self.sync_camera_dist()
        self.update_camera()

    def sync_camera_dist(self):
        cam_info = p.getDebugVisualizerCamera()
        if cam_info:
            self.distance = max(0.1, cam_info[10])

    def update_camera(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=self.distance,
            cameraYaw=self.yaw,
            cameraPitch=self.pitch,
            cameraTargetPosition=self.target_pos,
        )
