# import json
# import re
# import os
# import cv2
# import time
# import torch
# import logging
#
# from django.views.decorators.csrf import csrf_exempt
#
# from . import login, wifi
# from ultralytics import YOLO
# from djitellopy import Tello
# # from myapp.models import Face
# from insightface.app import FaceAnalysis
# from django.http import JsonResponse, StreamingHttpResponse
#
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',  # 定义日志格式
#     datefmt='%Y-%m-%d %H:%M:%S'  # 定义时间格式
# )
# # 禁用 pywifi 的日志记录
# logging.getLogger('pywifi').setLevel(logging.CRITICAL)
#
# TELLO_SSID = "TELLO-FDDA9E"
#
# LAND = 0
# TAKEOFF = 1
# HOVER = 2
# FOWARD = 3
# BACK = 4
# LEFT = 5
# RIGHT = 6
#
# # -------------------------------------------
# # 指令映射表
# CTRL_MAP = {
#     "takeoff":  '起飞',
#     "land":     '降落',
#     "up":       '上升',
#     "down":     '下降',
#     "forward":  '前进',
#     "back":     '后退',
#     "left":     '左移',
#     "right":    '右移',
#     "rotate_left":  '向左转',
#     "rotate_right": '向右转',
#
#     "battery": '查询电量'
# }
# # -------------------------------------------
#
# global drone_video, drone_controller
#
#
# class DroneVideo:
#     def __init__(self, tello: Tello):
#         # 人脸识别、手势识别
#         self.frame = None
#         self.tello = tello
#         print("实例化")
#
#     def get_frame_info(self):
#         """
#         返回无人机摄像头画面
#         """
#         try:
#             if not self.tello.stream_on:
#                 self.tello.streamon()
#             self.frame = self.tello.get_frame_read().frame
#
#             ret, jpeg = cv2.imencode('.jpg', self.frame)
#             return jpeg.tobytes()
#             # return jpeg.tobytes(), face_count, face_exists
#
#         except Exception as e:
#             logging.error(f"Error getting frame: {e}")  # 增加日志记录
#             return None
#             # return None, 0, False
#
#
# class DroneController:
#     def __init__(self, tello: Tello):
#         self.tello = tello
#         # RC
#         self.lr = 0
#         self.fb = 0
#         self.ud = 0
#         self.yv = 0
#         self.speed = 50
#         self.delay = 2.5
#
#     def key_ctrl(self, key):
#         """
#         根据输入控制无人机动作
#         :param key: 无人机动作对应的id
#         """
#         try:
#             self.lr = self.fb = self.ud = self.yv = 0
#
#             print(f"key: {key}")
#
#             if key == "takeoff":
#                 self.tello.takeoff()
#                 # time.sleep(5)
#                 # self.tello.land()
#             elif key == "land":
#                 self.tello.land()
#
#             if key == "left":
#                 self.lr = -self.speed
#             elif key == "right":
#                 self.lr = self.speed
#
#             if key == "forward":
#                 self.fb = self.speed
#             elif key == "back":
#                 self.fb = -self.speed
#
#             if key == "up":
#                 self.ud = self.speed
#             elif key == "down":
#                 self.ud = -self.speed
#
#             if key == "rotate_left":
#                 self.yv = self.speed
#             elif key == "rotate_right":
#                 self.yv = -self.speed
#
#             print(f"lr: {self.lr}, fb: {self.fb}, ud: {self.ud}, yv: {self.yv}")
#             self.tello.send_rc_control(self.lr, self.fb, self.ud, self.yv)
#             time.sleep(self.delay)
#             self.tello.send_rc_control(0, 0, 0, 0)
#
#             return {'status': 1, 'message': f'{CTRL_MAP[key]}'}
#
#         except Exception as e:
#                 logging.error(f"Error sending RC control: {e}")  # 增加日志记录
#                 return {'status': 0, 'message': f"Error: {e}"}
#
#     def set_delay(self, delay):
#         self.delay = delay
#
#     def set_speed(self, speed):
#         self.speed = speed
#
#     def get_current_state(self):
#         """{
#             'pitch': 0,          # 俯仰角
#             'roll': 0,           # 横滚角
#             'yaw': 0,            # 航向角
#             'vgx': 0,            # 水平速度(X轴)
#             'vgy': 0,            # 水平速度(Y轴)
#             'vgz': 0,            # 垂直速度(Z轴)
#             'templ': 0,          # 温度(低)
#             'temph': 0,          # 温度(高)
#             'tof': 0,            # ToF(飞行时间传感器)距离
#             'h': 0,              # 当前高度
#             'bat': 100,          # 电池电量
#             'baro': 0.0,         # 气压计高度
#             'time': 0,           # 飞行时间
#             'agx': 0.0,          # 加速度(X轴)
#             'agy': 0.0,          # 加速度(Y轴)
#             'agz': 0.0,          # 加速度(Z轴)
#             'wifi': 0            # Wi-Fi 信号强度
#         }"""
#         try:
#             state = self.tello.get_current_state()
#             return state
#         except Exception as e:
#             pass
#
#
# def initialize_drone():
#     """
#     初始化无人机对象
#     """
#     try:
#         tello = Tello()
#         tello.connect()
#         logging.info("无人机已成功连接并初始化")
#         return tello
#     except Exception as e:
#         logging.error(f"Error initializing drone: {e}")
#         return None
#
#
# def gen(camera):
#     """
#     视频流生成器
#     """
#     while True:
#         frame = camera.get_frame_info()
#         if frame is not None:
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#
#
# @csrf_exempt
# def key_input(request, drone_controller):
#     """
#     控制面板-键盘控制 对应的按键click后返回request_key
#     """
#     try:
#         data = json.loads(request.body)
#         request_key = data.get('request_key')
#         if not request_key:
#             return JsonResponse({'status': 0, 'message': "无效按键"})
#
#         response = drone_controller.key_ctrl(request_key)
#         return JsonResponse(response) if response else JsonResponse({'status': 0, 'message': "未知命令"})
#     except Exception as e:
#         logging.error(f"Error processing key input: {e}")
#         return JsonResponse({'status': 0, 'message': str(e)})
#
#
# @csrf_exempt
# def set_delay(request, drone_controller):
#     """
#     修改飞机单次移动时间
#     """
#     try:
#         data = json.loads(request.body)
#         delay = data.get('delay')
#         if not delay:
#             return JsonResponse({'status': 0, 'message': "设置无效"})
#
#         drone_controller.set_delay(delay)
#         return JsonResponse({'status': 1, 'message': "设置成功"})
#     except Exception as e:
#         logging.error(f"Error setting delay: {e}")
#         return JsonResponse({'status': 0, 'message': str(e)})
#
#
# def get_wifi_state(request):
#     """
#     获取Wi-Fi连接状态
#     """
#     try:
#         response = wifi.wifi_connect(TELLO_SSID)
#         if response['status'] == 1:
#             # wifi连接成功后，初始化无人机对象
#             tello = initialize_drone()
#             if tello:
#                 drone_video = DroneVideo(tello)
#                 drone_controller = DroneController(tello)
#                 print('1111111111')
#                 return JsonResponse({'status': 1, 'message': "无人机初始化成功", 'drone': {'video': drone_video, 'controller': drone_controller}})
#         return JsonResponse(response)
#     except Exception as e:
#         logging.error(f"Error getting wifi state: {e}")
#         return JsonResponse({'status': 0, 'message': str(e)}, status=500)
#
#
# def video_feed(request, drone_video):
#     """
#     返回视频流
#     """
#     return StreamingHttpResponse(gen(drone_video), content_type='multipart/x-mixed-replace; boundary=frame')
#
#
# def get_current_state(request, drone_controller):
#     """
#     获取无人机当前状态
#     """
#     try:
#         state = drone_controller.get_current_state()
#         return JsonResponse({'status': 1, 'tello_state': state})
#     except Exception as e:
#         logging.error(f"Error getting current state: {e}")
#         return JsonResponse({'status': 0, 'message': str(e)}, status=500)

import json
import cv2
import time
import logging
from django.views.decorators.csrf import csrf_exempt
from djitellopy import Tello
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
import threading

from myapp import wifi

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger('pywifi').setLevel(logging.CRITICAL)

# 全局无人机实例
global_drone = None
TELLO_SSID = "TELLO-FDDA9E"

# 控制指令映射
CTRL_MAP = {
    "takeoff": '起飞',
    "land": '降落',
    "up": '上升',
    "down": '下降',
    "forward": '前进',
    "back": '后退',
    "left": '左移',
    "right": '右移',
    "rotate_left": '向左转',
    "rotate_right": '向右转',
    "battery": '查询电量'
}


class Drone:
    def __init__(self):
        self.tello = None
        self._is_connected = False
        self.frame = None
        self.lock = threading.Lock()
        self.isOpenDroneCamera = False

        # 控制参数
        self.lr = 0
        self.fb = 0
        self.ud = 0
        self.yv = 0
        self.channel_rod = 50   # 设置遥控器的 4 个通道杆量
        self.delay = 2.5
        self.speed = 10     # 无人机速度 10~100cm/s，限制在20cm以内

    def connect(self):
        """连接无人机"""
        with self.lock:
            try:
                self.tello = Tello()
                self.tello.connect()
                # self.tello.streamon()
                self._is_connected = True
                # self.isOpenDroneCamera = True
                logging.info("无人机连接成功")
                return True
            except Exception as e:
                logging.error(f"连接失败: {e}")
                self._is_connected = False
                # self.isOpenDroneCamera = False
                return False

    def is_connected(self):
        """返回连接状态"""
        with self.lock:
            return self._is_connected

    def get_frame(self):
        """获取视频帧"""
        with self.lock:
            if not self._is_connected or not self.isOpenDroneCamera:
                return None

            try:
                frame = self.tello.get_frame_read().frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes() if ret else None
            except Exception as e:
                logging.error(f"获取视频帧失败: {e}")
                return None

    def control(self, command):
        """执行控制命令"""
        with self.lock:
            if not self._is_connected:
                return {'status': 0, 'message': '无人机未连接'}

            try:
                # 重置控制参数
                self.lr = self.fb = self.ud = self.yv = 0

                # 处理特殊命令
                if command == "takeoff":
                    self.tello.takeoff()
                    return {'status': 1, 'message': CTRL_MAP[command]}
                elif command == "land":
                    self.tello.land()
                    return {'status': 1, 'message': CTRL_MAP[command]}

                # 处理移动命令
                move_commands = {
                    "left": (-self.channel_rod, 0, 0, 0),
                    "right": (self.channel_rod, 0, 0, 0),
                    "forward": (0, self.channel_rod, 0, 0),
                    "back": (0, -self.channel_rod, 0, 0),
                    "up": (0, 0, self.channel_rod, 0),
                    "down": (0, 0, -self.channel_rod, 0),
                    "rotate_left": (0, 0, 0, self.channel_rod),
                    "rotate_right": (0, 0, 0, -self.channel_rod)
                }

                if command in move_commands:
                    self.lr, self.fb, self.ud, self.yv = move_commands[command]
                    self.tello.send_rc_control(self.lr, self.fb, self.ud, self.yv)
                    time.sleep(self.delay)
                    self.tello.send_rc_control(0, 0, 0, 0)  # 停止
                    return {'status': 1, 'message': CTRL_MAP[command]}

                return {'status': 0, 'message': '未知命令'}
            except Exception as e:
                logging.error(f"控制指令失败: {e}")
                return {'status': 0, 'message': str(e)}

    def update_speed(self, speed):
        """设置移动速度"""
        with self.lock:
            self.speed = max(10, min(20, speed))  # 限制在10-100之间
            self.tello.set_speed(self.speed)

    def get_battery(self):
        """获取电池电量"""
        with self.lock:
            return self.tello.get_battery() if self._is_connected else 0

    def get_current_state(self):
        """获取无人机当前状态"""
        with self.lock:
            try:
                state = self.tello.get_current_state()
                return state
            except Exception as e:
                logging.error(f"获取当前状态失败: {e}")
                return {}

    def disconnect(self):
        """断开连接"""
        with self.lock:
            if self._is_connected:
                self.tello.streamoff()
                self.tello.end()
                self._is_connected = False
                logging.info("无人机断开连接")


# ================ Django 视图函数 ================
@csrf_exempt
def connect_drone(request):
    """连接无人机视图"""
    global global_drone

    # 如果已有实例则先断开
    if global_drone and global_drone.is_connected():
        global_drone.disconnect()
        logging.info("已断开之前的无人机连接")

    # 创建新实例
    global_drone = Drone()

    # 尝试连接 Wi-Fi
    wifi_response = wifi.wifi_connect(TELLO_SSID)
    if wifi_response['status'] != 1:
        logging.error(f"Wi-Fi 连接失败: {wifi_response['message']}")
        return JsonResponse({'status': 0, 'message': f"Wi-Fi 连接失败: {wifi_response['message']}"})

    # 尝试连接无人机
    if global_drone.connect():
        battery_level = global_drone.get_battery()
        logging.info(f"无人机连接成功，电池电量: {battery_level}%")
        return JsonResponse({'status': 1, 'message': f'连接成功,电池电量:{battery_level}'})
    else:
        logging.error("无人机连接失败")
        return JsonResponse({'status': 0, 'message': '无人机连接失败'})


@csrf_exempt
def control(request):
    """控制指令视图"""
    if not global_drone or not global_drone.is_connected():
        print('111')
        return JsonResponse({'status': 0, 'message': '无人机未连接'})

    try:
        data = json.loads(request.body)
        command = data.get('command')
        print(f'command: {command}')
        if not command:
            return JsonResponse({'status': 0, 'message': '无效指令'})

        return JsonResponse(global_drone.control(command))
    except Exception as e:
        logging.error(f"处理控制指令时出错: {e}")
        return JsonResponse({'status': 0, 'message': str(e)})


def video_stream(request):
    """视频流视图"""
    if not global_drone or not global_drone.is_connected():
        return JsonResponse({'status': 0, 'message': '无人机未连接'})

    def generate():
        while True:
            frame = global_drone.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                time.sleep(0.1)

    return StreamingHttpResponse(generate(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_current_state(request):
    """获取无人机当前状态"""
    if not global_drone or not global_drone.is_connected():
        return JsonResponse({'status': 0, 'message': '无人机未连接'})

    try:
        state = global_drone.get_current_state()
        return JsonResponse({'status': 1, 'tello_state': state})
    except Exception as e:
        logging.error(f"获取当前状态时出错: {e}")
        return JsonResponse({'status': 0, 'message': str(e)})


@csrf_exempt
def update_speed(request):
    """设置无人机速度"""
    if not global_drone or not global_drone.is_connected():
        return JsonResponse({'status': 0, 'message': '无人机未连接'})

    try:
        data = json.loads(request.body)
        speed = data.get('speed')
        if not speed:
            return JsonResponse({'status': 0, 'message': '设置无效'})

        global_drone.update_speed(speed)
        return JsonResponse({'status': 1, 'message': f"设置成功，速度: {speed} cm/s"})
    except Exception as e:
        logging.error(f"设置速度时出错: {e}")
        return JsonResponse({'status': 0, 'message': str(e)})


def turn_drone_camera(request):
    try:
        # 检查全局无人机实例是否存在且已连接
        if global_drone is None or not global_drone.is_connected():
            logging.error("无人机未连接")
            return JsonResponse({'status': 0, 'message': '无人机未连接'}, status=400)

        with global_drone.lock:
            if global_drone.isOpenDroneCamera:
                global_drone.tello.streamoff()
                global_drone.isOpenDroneCamera = False
                return JsonResponse({'status': 0, 'message': '关闭无人机摄像头成功'})
            else:
                global_drone.tello.streamon()
                global_drone.isOpenDroneCamera = True
                return JsonResponse({'status': 1, 'message': '打开无人机摄像头成功'})

    except Exception as e:
        logging.error(f"Error turning camera: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
