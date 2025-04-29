import cv2
import json
import time
import queue
import logging
import threading
import numpy as np

from djitellopy import Tello

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse

from myapp import wifi
from .face_analysis import face_analysis_instance
from .commands import COMMANDS_MAP
from .face_track import FaceTracker

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


class Drone:
    def __init__(self):
        self.tello = None
        self._is_connected = False
        self.frame = None
        self.lock = threading.Lock()
        self.isOpenDroneCamera = False

        self.faceDetect = face_analysis_instance
        self.tracker = None

        self.initialBarometer = 0

        # 控制参数
        self.lr = 0
        self.fb = 0
        self.ud = 0
        self.yv = 0
        self.channel_rod = 50  # UI界面的速度其实是设置遥控器的 4 个通道杆量
        self.delay = 2.5

        self.hError = 0
        self.vError = 0
        self.isTracking = False

        self.command_queue = queue.Queue()
        self.command_thread = None

        self.visualization_enabled = False
        self.visualizer_thread = None

    def start_visualization(self):
        self.tracker.visualization_enabled = True
        self.tracker.visualizer.start()

    def stop_visualization(self):
        self.tracker.visualization_enabled = False
        self.tracker.visualizer.stop()

    def connect(self):
        """连接无人机"""
        with self.lock:
            try:
                self.tello = Tello()
                self.tello.connect()
                self._is_connected = True
                self.initialBarometer = self.tello.get_barometer()
                self.tracker = FaceTracker(self.tello, self.faceDetect)
                logging.info("无人机连接成功")
                return True
            except Exception as e:
                logging.error(f"连接失败: {e}")
                self._is_connected = False
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
                # print(frame.shape)
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.isTracking:
                    self.frame, face_info = self.tracker.find_face(self.frame)
                    self.tracker.track(face_info)

                ret, jpeg = cv2.imencode('.jpg', self.frame)
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
                if command == "stop" or command == "":
                    return
                if command == "takeoff":
                    self.tello.takeoff()
                    return {'status': 1, 'message': COMMANDS_MAP[command]}
                elif command == "land":
                    self.tello.land()
                    return {'status': 1, 'message': COMMANDS_MAP[command]}

                # 处理移动命令
                move_commands = {
                    "left": (-self.channel_rod, 0, 0, 0),
                    "right": (self.channel_rod, 0, 0, 0),
                    "forward": (0, self.channel_rod, 0, 0),
                    "backward": (0, -self.channel_rod, 0, 0),
                    "up": (0, 0, self.channel_rod, 0),
                    "down": (0, 0, -self.channel_rod, 0),
                    "rotate_left": (0, 0, 0, -self.channel_rod),
                    "rotate_right": (0, 0, 0, self.channel_rod)
                }

                if command in move_commands:
                    self.lr, self.fb, self.ud, self.yv = move_commands[command]

                    threading.Thread(target=self._execute_command).start()

                    self.command_queue.put(move_commands[command])  # 将命令放入队列
                    return {'status': 1, 'message': COMMANDS_MAP[command]}

                return {'status': 0, 'message': '未知命令'}
            except Exception as e:
                logging.error(f"控制指令失败: {e}")
                return {'status': 0, 'message': str(e)}

    def process_commands(self):
        while self._is_connected:
            try:
                command = self.command_queue.get(timeout=1)  # 从队列中获取命令
                self.lr, self.fb, self.ud, self.yv = command
                self._execute_command()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"处理命令时出错: {e}")

    def _execute_command(self):
        """执行控制命令的线程函数"""
        try:
            # print('command1')
            self.tello.send_rc_control(self.lr, self.fb, self.ud, self.yv)
            time.sleep(self.delay)
            # print('command2')
            self.tello.send_rc_control(0, 0, 0, 0)  # 停止
        except Exception as e:
            logging.error(f"执行控制指令时出错: {e}")

    def update_speed(self, speed):
        with self.lock:
            self.channel_rod = max(30, min(100, speed))

    def get_battery(self):
        with self.lock:
            return self.tello.get_battery() if self._is_connected else 0

    def get_current_state(self):
        with self.lock:
            try:
                state = self.tello.get_current_state()
                state['wifi'] = wifi.get_wifi_signal_strength()
                # print(state['wifi'])
                # 原始的 state['baro'] 单位是 m，计算后转化为 cm
                state['baro'] = state['baro']*100 - self.initialBarometer
                return state
            except Exception as e:
                logging.error(f"获取当前状态失败: {e}")
                return {}

    def disconnect(self):
        with self.lock:
            if self._is_connected:
                self.tello.streamoff()
                self.tello.end()
                self._is_connected = False
                self.isTracking = False
                logging.info("无人机断开连接")

    def is_stream(self):
        with self.lock:
            return self.tello.stream_on


def get_drone():
    global global_drone
    return global_drone


def is_drone_connected():
    drone = get_drone()
    if drone is None:
        return False
    try:
        return drone.is_connected()
    except AttributeError:
        return False


def is_stream_on():
    drone = get_drone()
    if drone is None:
        return False
    try:
        return drone.is_stream()
    except AttributeError:
        return False


def control_drone(command):
    drone = get_drone()
    if drone is None:
        return {'status': 0, 'message': '无人机未初始化'}
    try:
        return drone.control(command)
    except Exception as e:
        logging.error(f"控制无人机时出错: {e}")
        return {'status': 0, 'message': str(e)}


@csrf_exempt
def connect_drone(request):
    global global_drone
    # # 如果已有实例则先断开
    # if global_drone and global_drone.is_connected():
    #     global_drone.disconnect()
    #     logging.info("已断开之前的无人机连接")

    # 创建新实例
    global_drone = Drone()

    # 尝试连接 Wi-Fi
    wifi_response = wifi.wifi_connect(TELLO_SSID)
    if wifi_response['status'] != 1:
        logging.error(f"Wi-Fi 连接失败: {wifi_response['message']}")
        return JsonResponse({'status': 0, 'message': f"Wi-Fi 连接失败: {wifi_response['message']}"})

    # 尝试连接无人机
    drone = get_drone()
    if drone.connect():
        battery_level = drone.get_battery()
        logging.info(f"无人机连接成功，电池电量: {battery_level}%")
        return JsonResponse({'status': 1, 'message': f'连接成功,电池电量:{battery_level}'})
    else:
        logging.error("无人机连接失败")
        return JsonResponse({'status': 0, 'message': '无人机连接失败'})


@csrf_exempt
def disconnect_drone(request):
    # try:
    drone = get_drone()
    # if drone:
    #     return JsonResponse({'status': 0, 'message': '未连接无人机'})

    if drone and drone.is_connected():
        drone.disconnect()
        logging.info("无人机已断开连接")
        return JsonResponse({'status': 1, 'message': '无人机已断开连接'})
    else:
        logging.error("无人机未连接")
        return JsonResponse({'status': 0, 'message': '无人机未连接'})


@csrf_exempt
def control(request):
    drone = get_drone()
    if drone is None or not drone.is_connected():
        # print('111')
        logging.error("无人机未连接")
        return JsonResponse({'status': 0, 'message': '无人机未连接'})

    try:
        data = json.loads(request.body)
        command = data.get('command')
        print(f'command: {command}')
        if not command:
            return JsonResponse({'status': 0, 'message': '无效指令'})

        return JsonResponse(drone.control(command))
    except Exception as e:
        logging.error(f"处理控制指令时出错: {e}")
        return JsonResponse({'status': 0, 'message': str(e)})


def video_stream(request):
    drone = get_drone()
    if drone is None or not drone.is_connected():
        logging.error("无人机未连接")
        return JsonResponse({'status': 0, 'message': '无人机未连接'})

    def generate():
        while True:
            frame = drone.get_frame()

            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                time.sleep(0.1)

    return StreamingHttpResponse(generate(), content_type='multipart/x-mixed-replace; boundary=frame')


def get_current_state(request):
    drone = get_drone()
    if drone is None or not drone.is_connected():
        logging.error("无人机未连接")
        return JsonResponse({'status': 0, 'message': '无人机未连接'})

    try:
        state = drone.get_current_state()

        return JsonResponse({'status': 1, 'tello_state': state})
    except Exception as e:
        logging.error(f"获取当前状态时出错: {e}")
        return JsonResponse({'status': 0, 'message': str(e)})


@csrf_exempt
def update_speed(request):
    drone = get_drone()
    if not drone or not drone.is_connected():
        logging.error("无人机未连接")
        return JsonResponse({'status': 0, 'message': '无人机未连接'})

    try:
        data = json.loads(request.body)
        speed = data.get('speed')
        if not speed:
            return JsonResponse({'status': 0, 'message': '设置无效'})

        drone.update_speed(speed)
        return JsonResponse({'status': 1, 'message': f"设置成功，速度: {speed} cm/s"})
    except Exception as e:
        logging.error(f"设置速度时出错: {e}")
        return JsonResponse({'status': 0, 'message': str(e)})


def turn_drone_camera(request):
    drone = get_drone()
    try:
        # 检查全局无人机实例是否存在且已连接
        if drone is None or not drone.is_connected():
            logging.error("无人机未连接")
            return JsonResponse({'status': 0, 'message': '无人机未连接'})

        with drone.lock:
            if drone.isOpenDroneCamera:
                drone.tello.streamoff()
                drone.isOpenDroneCamera = False
                return JsonResponse({'status': 0, 'message': '关闭无人机摄像头'})
            else:
                drone.tello.streamon()
                drone.isOpenDroneCamera = True
                return JsonResponse({'status': 1, 'message': '打开无人机摄像头'})

    except Exception as e:
        logging.error(f"Error turning camera: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def turn_face_track(request):
    drone = get_drone()
    try:
        if drone is None or not drone.is_connected():
            logging.error("无人机未连接")
            return JsonResponse({'status': 0, 'message': '无人机未连接'})

        if not is_stream_on():
            logging.error("无人机摄像头未开启")
            return JsonResponse({'status': 0, 'message': '无人机摄像头未开启，无法开启人脸跟随'})

        with drone.lock:
            drone.isTracking = not drone.isTracking
            if drone.isTracking:
                logging.info("开启人脸跟随")
                return JsonResponse({'status': 1, 'message': '打开人脸跟随'})
            else:
                logging.info("停止人脸跟随")
                drone.visualization_enabled = False
                drone.stop_visualization()
                return JsonResponse({'status': 0, 'message': '停止人脸跟随'})

    except Exception as e:
        logging.error(f"Error turning face tracking: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
def toggle_visualization(request):
    drone = get_drone()
    try:
        if drone is None or not drone.is_connected():
            logging.error("无人机未连接")
            return JsonResponse({'status': 0, 'message': '无人机未连接'})

        if drone.visualization_enabled:
            drone.visualization_enabled = False

            drone.stop_visualization()

            return JsonResponse({'status': 0, 'message': '关闭PID可视化'})
        else:
            drone.visualization_enabled = True
            drone.start_visualization()

            return JsonResponse({'status': 1, 'message': '打开PID可视化'})

    except Exception as e:
        logging.error(f"Error toggling visualization: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)