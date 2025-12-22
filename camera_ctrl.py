import math
import pybullet as p
from config_loader import Config


class CameraController:
    """相机控制器类，用于处理PyBullet调试可视化器中的相机交互
    
    功能：
    - 右键旋转相机视角
    - 左键平移相机目标点
    - 滚轮缩放相机距离
    - 自动同步相机距离变化
    """
    
    def __init__(self):
        """初始化相机控制器
        
        从配置文件中加载初始相机参数，并设置初始状态
        """
        # 从配置加载初始相机参数
        self.distance = Config.CAM_DIST       # 相机到目标点的初始距离
        self.yaw = Config.CAM_YAW             # 相机初始偏航角（绕z轴旋转角度）
        self.pitch = Config.CAM_PITCH         # 相机初始俯仰角（上下旋转角度）
        self.target_pos = list(Config.CAM_TARGET)  # 相机目标点初始位置

        # 鼠标状态跟踪
        self.last_mouse_x = 0                 # 上一次鼠标X坐标
        self.last_mouse_y = 0                 # 上一次鼠标Y坐标
        self.first_input = True               # 标记是否为第一次鼠标输入

        # 鼠标按钮状态
        self.left_button_down = False         # 左键是否按下（用于平移）
        self.right_button_down = False        # 右键是否按下（用于旋转）

        # 应用初始相机设置
        self.update_camera()

    def update(self):
        """更新相机状态，处理鼠标事件
        
        这是相机控制的主循环方法，需要在仿真主循环中持续调用
        处理所有鼠标事件：
        - 右键拖动：旋转相机视角
        - 左键拖动：平移相机目标点
        - 滚轮：缩放相机距离
        """
        # 获取当前鼠标事件
        events = p.getMouseEvents()

        # 如果没有事件，仅同步相机距离并更新相机
        if not events:
            self.sync_camera_dist()  # 同步相机距离（处理滚轮缩放）
            self.update_camera()      # 更新相机视图
            return

        # 处理所有鼠标事件
        for e in events:
            # 解析事件参数
            event_type, curr_x, curr_y, button_index, button_state = e

            # 处理第一次鼠标输入，初始化鼠标位置
            if self.first_input:
                self.last_mouse_x = curr_x
                self.last_mouse_y = curr_y
                self.first_input = False
                continue

            # 处理按钮按下/释放事件
            if event_type == 2:  # 按钮状态变化事件
                is_down = button_state in (1, 2, 3)  # 1=按下，2=保持按下，3=释放

                # 处理左键（平移）
                if button_index == 0:
                    self.left_button_down = is_down
                    if is_down:
                        self.last_mouse_x = curr_x
                        self.last_mouse_y = curr_y

                # 处理右键（旋转）
                if button_index == 2:
                    self.right_button_down = is_down
                    if is_down:
                        self.last_mouse_x = curr_x
                        self.last_mouse_y = curr_y

            # 处理鼠标移动事件
            elif event_type == 1:  # 鼠标移动事件
                # 计算鼠标位移
                dx = curr_x - self.last_mouse_x
                dy = curr_y - self.last_mouse_y

                # 右键旋转相机
                if self.right_button_down:
                    # 根据鼠标位移更新相机偏航角和俯仰角
                    self.yaw -= dx * Config.MOUSE_SENSITIVITY_ROTATE
                    # 限制俯仰角范围在-89度到-1度之间
                    self.pitch = max(-89, min(-1, self.pitch))

                # 左键平移相机目标点
                if self.left_button_down:
                    # 计算平移灵敏度，与相机距离成正比
                    pan_sens = Config.MOUSE_SENSITIVITY_PAN * self.distance
                    # 将当前偏航角转换为弧度
                    yaw_rad = math.radians(self.yaw)
                    # 修正鼠标X方向位移，保持拖拽方向直观
                    dx_fix = -dx
                    
                    # 计算世界坐标系下的平移量
                    d_world_x = (dx_fix * math.cos(yaw_rad) - dy * math.sin(yaw_rad)) * pan_sens
                    d_world_y = (dx_fix * math.sin(yaw_rad) + dy * math.cos(yaw_rad)) * pan_sens

                    # 更新相机目标点位置
                    self.target_pos[0] += d_world_x
                    self.target_pos[1] += d_world_y

                # 更新鼠标位置
                self.last_mouse_x = curr_x
                self.last_mouse_y = curr_y

        # 同步相机距离（处理滚轮缩放）
        self.sync_camera_dist()
        # 更新相机视图
        self.update_camera()

    def sync_camera_dist(self):
        """同步相机距离，处理滚轮缩放事件
        
        从PyBullet获取当前相机距离，并限制最小值
        """
        # 获取当前相机信息
        cam_info = p.getDebugVisualizerCamera()
        if cam_info:
            # 从相机信息中获取当前距离，并限制最小值为0.1
            self.distance = max(0.1, cam_info[10])

    def update_camera(self):
        """应用相机参数到PyBullet调试可视化器
        
        使用当前相机参数更新PyBullet的调试可视化相机
        """
        p.resetDebugVisualizerCamera(
            cameraDistance=self.distance,       # 相机到目标点的距离
            cameraYaw=self.yaw,                 # 相机偏航角（绕z轴旋转角度）
            cameraPitch=self.pitch,             # 相机俯仰角（上下旋转角度）
            cameraTargetPosition=self.target_pos,  # 相机目标点位置
        )
