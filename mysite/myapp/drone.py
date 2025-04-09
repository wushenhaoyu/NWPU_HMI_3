import json
import re
import os
import cv2
import time
import torch
import logging

from django.views.decorators.csrf import csrf_exempt

from . import login, wifi
from ultralytics import YOLO
from djitellopy import Tello
# from myapp.models import Face
from insightface.app import FaceAnalysis
from django.http import JsonResponse, StreamingHttpResponse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',  # 定义日志格式
    datefmt='%Y-%m-%d %H:%M:%S'  # 定义时间格式
)
# 禁用 pywifi 的日志记录
logging.getLogger('pywifi').setLevel(logging.CRITICAL)

TELLO_SSID = "TELLO-FDDA9E"

LAND = 0
TAKEOFF = 1
HOVER = 2
FOWARD = 3
BACK = 4
LEFT = 5
RIGHT = 6

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

global drone_video, drone_controller


class DroneVideo:
    def __init__(self, tello: Tello):
        # 人脸识别、手势识别
        self.frame = None
        self.tello = tello
        print("实例化")

    def get_frame_info(self):
        """
        返回无人机摄像头画面
        """
        try:
            if not self.tello.stream_on:
                self.tello.streamon()
            self.frame = self.tello.get_frame_read().frame

            ret, jpeg = cv2.imencode('.jpg', self.frame)
            return jpeg.tobytes()
            # return jpeg.tobytes(), face_count, face_exists

        except Exception as e:
            logging.error(f"Error getting frame: {e}")  # 增加日志记录
            return None
            # return None, 0, False


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

    def set_speed(self, speed):
        self.speed = speed

    def get_current_state(self):
        """{
            'pitch': 0,          # 俯仰角
            'roll': 0,           # 横滚角
            'yaw': 0,            # 航向角
            'vgx': 0,            # 水平速度(X轴)
            'vgy': 0,            # 水平速度(Y轴)
            'vgz': 0,            # 垂直速度(Z轴)
            'templ': 0,          # 温度(低)
            'temph': 0,          # 温度(高)
            'tof': 0,            # ToF(飞行时间传感器)距离
            'h': 0,              # 当前高度
            'bat': 100,          # 电池电量
            'baro': 0.0,         # 气压计高度
            'time': 0,           # 飞行时间
            'agx': 0.0,          # 加速度(X轴)
            'agy': 0.0,          # 加速度(Y轴)
            'agz': 0.0,          # 加速度(Z轴)
            'wifi': 0            # Wi-Fi 信号强度
        }"""
        try:
            state = self.tello.get_current_state()
            return state
        except Exception as e:
            pass


def get_wifi_state(request):
    try:
        response = wifi.wifi_connect(TELLO_SSID)
        if response['status'] == 1:
            drone_video = DroneVideo(Tello())
            drone_controller = DroneController(Tello())
            logging.info("无人机已成功连接并初始化")

        return JsonResponse(response)
    except Exception as e:
        logging.error(f"Error getting wifi state: {e}")  # 增加日志记录
        return JsonResponse({'status': 0, 'message': str(e)}, status=500)


def gen(camera):
    while True:
        # frame, face_count, face_exists = camera.get_frame_info()
        frame = camera.get_frame_info()

        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed(request):
    return StreamingHttpResponse(gen(drone_video), content_type='multipart/x-mixed-replace; boundary=frame')


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


def get_current_state(request):
    try:
        state = drone_controller.get_current_state()
        response = {
            'status': 1,  # 表示成功获取状态
            'tello_state': state
        }
        return JsonResponse(response)
    except Exception as e:
        logging.error(f"Error getting current state: {e}")  # 增加日志记录
        return JsonResponse({'status': 0, 'message': str(e)}, status=500)
