import re
import os
import cv2
import time
import torch
import logging
from . import login,wifi
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

# 调整ROI参数(右侧位置)
ROI_WIDTH_RATIO = 1.2
ROI_HEIGHT_RATIO = 1.5
ROI_OFFSET_X_RATIO = 1.2  # 正数表示右侧偏移
ROI_OFFSET_Y_RATIO = 0.8

TELLO_SSID = "TELLO-FDDA9E"

LAND = 0
TAKEOFF = 1
HOVER = 2
FOWARD = 3
BACK = 4
LEFT = 5
RIGHT = 6


class Drone:
    def __init__(self):
        # 人脸识别、手势识别
        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition', 'landmark_2d_106'],
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
        self.model = YOLO(os.path.join(os.path.dirname(__file__), 'best.pt'))

        self.isFaceRecognize = True
        self.name = ""
        # self.embedding = []

        self.tello = Tello()
        # 连接无人机wifi
        wifi.wifi_connect(TELLO_SSID)
        self.tello.connect()
        # 默认开启视频流
        self.tello.streamon()
        self.frame = self.tello.get_frame_read().frame
        logging.info(f"drone battery: {self.tello.get_battery()}")

        self.action = LAND #控制飞行状态
        self.action_cache1 = None
        self.action_cache2 = None

        self.isDroneIdle = False  # 是否处于空闲状态

        self.move_speed = 10  # 移动速度 cm/s
        self.move_distance = 10 #单次移动距离 cm

    def change_action(self, action):
        """加窗改变无人机状态"""
        self.action_cache2 = self.action_cache1
        self.action_cache1 = self.action
        self.action = action

        if self.action_cache1 == self.action_cache2:
            if self.action_cache1 == self.action:
                self.send_action_command(self.action)

    def send_action_command(self, command):
        self.check_drone_idle()
        if self.isDroneIdle:
            pass #空闲就发送指令
        else:
            return #不空闲就不发送指令


        
    
    def check_drone_idle(self):
        """检查无人机是否处于空闲状态"""
        pass
    def get_frame_info(self):
        """只有一个人时(绿)，绘制对应的手势框(蓝)，
        多人时会画脸的框，并标红
        Returns:
            1.经过识别并标注的人脸frame
            2.人脸数量
            3.人脸是否在数据库中存在(只有一个人时)
        """
        try:
            frame = self.tello.get_frame_read().frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame = cv2.flip(frame, 1)

            ih, iw, _ = self.frame.shape
            if self.isFaceRecognize:    # 默认开启
                face_results = []  # 存储人脸识别结果
                try:
                    faces = self.app.get(self.frame)
                except Exception as e:
                    logging.error(f"app.get()检测人脸时出错： {e}")
                    faces = []

                if not faces:
                    # logging.info("没有检测到人脸")
                    face_results.append({"name": "", "bbox": None})
                else:
                    # 获取图像中所有在数据库中存在的人脸
                    for index, face in enumerate(faces):
                        try:
                            recognized_face = login.compare_face_with_database(face)
                            if recognized_face is not None:
                                face_results.append({
                                    "name": recognized_face.name,
                                    "bbox": face['bbox'],
                                })

                        except Exception as e:
                            logging.error(f"识别人脸 {index} 时出错： {e}")
                            # face_results.append({"name": "", "bbox": None})

                    # 如果数据库人脸识别有结果
                    if len(face_results) > 0:
                        # 识别的第一个人，脸绿手蓝
                        # logging.info(f"识别到的人脸为：{face_results[0]['name']}")
                        # 以识别出的第一个人为准
                        name = face_results[0]["name"]
                        bbox = face_results[0]["bbox"]
                        x1, y1 = int(bbox[0]), int(bbox[1])
                        x2, y2 = int(bbox[2]), int(bbox[3])
                        w = x2 - x1
                        h = y2 - y1
                        # 调整ROI到人脸右侧
                        roi_x = x2 + int(w * ROI_OFFSET_X_RATIO)  # 右侧偏移使用加法
                        roi_y = y2 + int(h * ROI_OFFSET_Y_RATIO)
                        roi_w = int(w * ROI_WIDTH_RATIO)
                        roi_h = int(h * ROI_HEIGHT_RATIO)

                        # ROI边界约束
                        roi_x = max(0, min(roi_x, iw - roi_w))
                        roi_y = max(0, min(roi_y, ih - roi_h))
                        # 脸
                        cv2.putText(self.frame, name, (int(bbox[0]), int(bbox[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.rectangle(self.frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      (0, 255, 0), 2)
                        # 手
                        cv2.rectangle(self.frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h),
                                      (255, 0, 0), 2)

                        # 其他人统一标红
                        for face in face_results[1:]:
                            name = face["name"]
                            bbox = face['bbox']
                            cv2.putText(self.frame, name, (int(bbox[0]), int(bbox[1]) - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            cv2.rectangle(self.frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                          (0, 0, 255), 2)

                    # 数据库中没有存入当前图片中的所有人脸
                    else:
                        # 绘制陌生人边界框
                        for face in faces:
                            bbox = face['bbox']
                            cv2.rectangle(self.frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                          (0, 0, 255), 2)
                            cv2.putText(self.frame, 'Stranger', (int(bbox[0]), int(bbox[1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        # logging.info("当前画面中的人均未在数据库中,请先存入人脸")

            # cv2.imshow('frame', self.frame)
            ret, jpeg = cv2.imencode('.jpg', self.frame)
            return jpeg.tobytes()
            # return jpeg.tobytes(), face_count, face_exists

        except Exception as e:
            logging.error(f"Error getting frame: {e}")  # 增加日志记录
            return None
            # return None, 0, False

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
    
    def set_speed(self, speed):
        """设置无人机速度"""
        try:
            if(speed < 10): #限幅
                speed = 10
            elif(speed >30):
                speed = 30
            self.move_speed = speed
            self.tello.set_speed(speed)
        except Exception as e:
            logging.error(f"Error setting speed: {e}")

    
    def set_move_distance(self, distance):
        """设置无人机移动距离,后续移动要调用这个"""
        try:
            if(distance < 10): #限幅
                distance = 10
            elif(distance >30):
                distance = 30
            self.move_distance = distance
        except Exception as e:
            logging.error(f"Error setting move distance: {e}")

    

 
        

    
        
        




# drone = Drone()
# drone.get_frame_info()

def gen(camera):
    while True:
        # frame, face_count, face_exists = camera.get_frame_info()
        frame = camera.get_frame_info()
        # frame, face_count, face_exists = camera.get_frame_info()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def video_feed(request):
    return StreamingHttpResponse(gen(drone), content_type='multipart/x-mixed-replace; boundary=frame')


# def turn_camera(request):
#     try:
#         if drone.isOpenDroneCamera:
#             # drone.cap.release()
#
#             drone.isOpenDroneCamera = False
#             return JsonResponse({'status': 0, 'message': 'Drone Camera turned successfully'})
#         elif not drone.isOpenDroneCamera:
#             # drone.cap = cv2.VideoCapture(0)
#
#             drone.isOpenDroneCamera = True
#             return JsonResponse({'status': 1, 'message': 'Drone Camera turned successfully'})
#     except Exception as e:
#         logging.error(f"Error turning camera: {e}")  # 增加日志记录
#         return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def get_current_state(request):
    try:
        state = drone.get_current_state()
        response = {
            'status': 1,  # 表示成功获取状态
            'tello_state': state
        }
        return JsonResponse(response)
    except Exception as e:
        logging.error(f"Error getting current state: {e}")  # 增加日志记录
        return JsonResponse({'status': 0 , 'message': str(e)}, status=500)

