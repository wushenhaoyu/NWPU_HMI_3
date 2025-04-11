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
    "stop": '停止',
    "up": '上升',
    "down": '下降',
    "forward": '前进',
    "backward": '后退',
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
        self.channel_rod = 50  # 设置遥控器的 4 个通道杆量
        self.delay = 2.5
        self.speed = 10  # 无人机速度 10~100cm/s，限制在20cm以内

    def connect(self):
        """连接无人机"""
        with self.lock:
            try:
                self.tello = Tello()
                self.tello.connect()
                self._is_connected = True
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
                if command == "stop" or command == "":
                    return
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
                    "backward": (0, -self.channel_rod, 0, 0),
                    "up": (0, 0, self.channel_rod, 0),
                    "down": (0, 0, -self.channel_rod, 0),
                    "rotate_left": (0, 0, 0, self.channel_rod),
                    "rotate_right": (0, 0, 0, -self.channel_rod)
                }

                if command in move_commands:
                    start_time = time.time()
                    self.lr, self.fb, self.ud, self.yv = move_commands[command]
                    threading.Thread(target=self._execute_command).start()
                    # while time.time() - start_time < self.delay:
                    #     self.tello.send_rc_control(self.lr, self.fb, self.ud, self.yv)

                    # self.tello.send_rc_control(self.lr, self.fb, self.ud, self.yv)
                    # time.sleep(self.delay)
                    self.tello.send_rc_control(0, 0, 0, 0)  # 停止
                    return {'status': 1, 'message': CTRL_MAP[command]}

                return {'status': 0, 'message': '未知命令'}
            except Exception as e:
                logging.error(f"控制指令失败: {e}")
                return {'status': 0, 'message': str(e)}

    def _execute_command(self):
        """执行控制命令的线程函数"""
        try:
            self.tello.send_rc_control(self.lr, self.fb, self.ud, self.yv)
            time.sleep(self.delay)
            self.tello.send_rc_control(0, 0, 0, 0)  # 停止
        except Exception as e:
            logging.error(f"执行控制指令时出错: {e}")

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

    def is_stream(self):
        with self.lock:
            return self.tello.stream_on


def get_drone():
    """获取全局无人机实例
    Returns:
        Drone: 全局无人机实例，如果未初始化则为None
    """
    global global_drone
    return global_drone


def is_drone_connected():
    """检查无人机是否已连接
    Returns:
        bool: 如果无人机已连接返回True，否则返回False
    """
    drone = get_drone()
    if drone is None:
        return False
    try:
        return drone.is_connected()
    except AttributeError:
        return False


def is_stream_on():
    """检查无人机视频流是否已开启
    Returns:
        bool: 如果无人机已开启视频流返回True，否则返回False
    """
    drone = get_drone()
    if drone is None:
        return False
    try:
        return drone.is_stream()
    except AttributeError:
        return False


def control_drone(command):
    """控制无人机执行命令
    Args:
        command (str): 控制命令
    Returns:
        dict: 包含操作状态和消息的字典
    """
    drone = get_drone()
    if drone is None:
        return {'status': 0, 'message': '无人机未初始化'}
    try:
        return drone.control(command)
    except Exception as e:
        logging.error(f"控制无人机时出错: {e}")
        return {'status': 0, 'message': str(e)}


# ================ Django 视图函数 ================
@csrf_exempt
def connect_drone(request):
    """连接无人机视图"""
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
    # if global_drone.connect():
    #     battery_level = global_drone.get_battery()
    #     logging.info(f"无人机连接成功，电池电量: {battery_level}%")
    #     return JsonResponse({'status': 1, 'message': f'连接成功,电池电量:{battery_level}'})
    # else:
    #     logging.error("无人机连接失败")
    #     return JsonResponse({'status': 0, 'message': '无人机连接失败'})


@csrf_exempt
def disconnect_drone(request):
    """断开无人机连接视图"""
    # try:
    drone = get_drone()
    if drone and drone.is_connected():
        drone.disconnect()
        logging.info("无人机已断开连接")
        return JsonResponse({'status': 1, 'message': '无人机已断开连接'})
    else:
        logging.error("无人机未连接")
        return JsonResponse({'status': 0, 'message': '无人机未连接'})


@csrf_exempt
def control(request):
    """控制指令视图"""
    drone = get_drone()
    if not drone or not drone.is_connected():
        # print('111')
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
    """视频流视图"""
    drone = get_drone()
    if not drone or not drone.is_connected():
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
    # if not global_drone or not global_drone.is_connected():
    #     return JsonResponse({'status': 0, 'message': '无人机未连接'})

    # def generate():
    #     while True:
    #         frame = global_drone.get_frame()
    #         if frame:
    #             yield (b'--frame\r\n'
    #                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    #         else:
    #             time.sleep(0.1)

    # return StreamingHttpResponse(generate(), content_type='multipart/x-mixed-replace; boundary=frame')


def get_current_state(request):
    """获取无人机当前状态"""
    drone = get_drone()
    if not drone or not drone.is_connected():
        return JsonResponse({'status': 0, 'message': '无人机未连接'})

    try:
        state = drone.get_current_state()
        return JsonResponse({'status': 1, 'tello_state': state})
    except Exception as e:
        logging.error(f"获取当前状态时出错: {e}")
        return JsonResponse({'status': 0, 'message': str(e)})


@csrf_exempt
def update_speed(request):
    """设置无人机速度"""
    drone = get_drone()
    if not drone or not drone.is_connected():
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
