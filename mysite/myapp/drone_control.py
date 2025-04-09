import json

import cv2
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from djitellopy import Tello
import time

# -------------------------------------------
# 指令映射表
CTRL_MAP = {
    "takeoff": '起飞',  # 示例：按下 "t" 键起飞
    "land": '降落',  # 示例：按下 "l" 键降落
    "up": '上升',  # 示例：按下 "l" 键降落
    "down": '下降',  # 示例：按下 "l" 键降落
    "forward": '前进',  # 示例：按下 "w" 键前进
    "back": '后退',  # 示例：按下 "s" 键后退
    "left": '左移',  # 示例：按下 "a" 键左移
    "right": '右移',  # 示例：按下 "d" 键右移
    "rotate_left": '向左转',  # 示例：按下 "d" 键右移
    "rotate_right": '向右转',  # 示例：按下 "d" 键右移

    "battery": '查询电量'
}


# -------------------------------------------


class DroneController:
    def __init__(self, tello: Tello):
        self.tello = tello

        # RC
        self.lr = 0
        self.fb = 0
        self.ud = 0
        self.yv = 0
        self.speed = 50
        self.delay = 2.5

    def key_ctrl(self, key):
        """
        根据输入控制无人机动作
        :param key: 无人机动作对应的id
        """
        self.lr = self.fb = self.ud = self.yv = 0

        print(f"key: {key}")

        if key == "takeoff":
            self.tello.takeoff()
            # time.sleep(5)
            # self.tello.land()
        elif key == "land":
            self.tello.land()

        if key == "left":
            self.lr = -self.speed
        elif key == "right":
            self.lr = self.speed

        if key == "forward":
            self.fb = self.speed
        elif key == "back":
            self.fb = -self.speed

        if key == "up":
            self.ud = self.speed
        elif key == "down":
            self.ud = -self.speed

        if key == "rotate_left":
            self.yv = self.speed
        elif key == "rotate_right":
            self.yv = -self.speed

        print(f"lr: {self.lr}, fb: {self.fb}, ud: {self.ud}, yv: {self.yv}")
        self.tello.send_rc_control(self.lr, self.fb, self.ud, self.yv)
        time.sleep(self.delay)
        self.tello.send_rc_control(0, 0, 0, 0)

        return {'status': 1, 'message': f'{CTRL_MAP[key]}'}

    def set_delay(self, delay):
        self.delay = delay


# drone = Tello()
# drone.connect()
# drone_controller = DroneController(myDrone)

@csrf_exempt
def key_input(request):
    """
    控制面板-键盘控制 对应的按键click后返回request_key
    """
    try:
        data = json.loads(request.body)
        request_key = data.get('request_key')
        print(f"request_key: {request_key}")
        if not request_key:
            return JsonResponse({'status': 0, 'message': "无效按键"})

        response = drone_controller.key_ctrl(request_key)

        #
        if response:
            return JsonResponse(response)
        else:
            action_ch = CTRL_MAP.get(request_key, "未知命令")
            return JsonResponse({'status': 1, 'message': action_ch})
    except Exception as e:
        print(f"Error: {e}")
        return JsonResponse({'status': 0, 'message': f"Error: {e}"})


@csrf_exempt
def set_delay(request):
    """
    修改飞机单次移动时间
    """
    try:
        data = json.loads(request.body)
        delay = data.get('delay')
        print(f"delay: {delay}")
        if not delay:
            return JsonResponse({'status': 0, 'message': "设置无效"})

        drone_controller.set_delay(delay)
        return JsonResponse({'status': 1, 'message': f"设置成功"})

    except Exception as e:
        print(f"Error: {e}")
        return JsonResponse({'status': 0, 'message': f"Error: {e}"})
