import json
import os
import threading

from django.http import JsonResponse, StreamingHttpResponse
import cv2
import insightface
from insightface.app import FaceAnalysis
import torch
import numpy as np
from scipy.spatial.distance import cosine
from django.views.decorators.csrf import csrf_exempt

from myapp.drone import global_drone
from myapp.models import Face
import re
import time
import logging
from myapp.hand_process import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',  # 定义日志格式
    datefmt='%Y-%m-%d %H:%M:%S'  # 定义时间格式
)


def compare_face_with_database(face, threshold=0.4):
    """从数据库中比较人脸
    :param face: 人脸信息字典，包含 'bbox' 人脸的边界框坐标，
                 'embedding' 人脸的特征向量, 'kps' 人脸的关键点
    :param threshold: 特征向量之间距离的阈值
    :return: 返回识别出的最相似的人脸对象，如果没有找到匹配的，则返回 None
    """
    input_embedding = face['embedding']
    all_faces = Face.objects.all()
    min_distance = float('inf')
    recognized_face = None

    for face_entry in all_faces:
        db_embedding = face_entry.get_feature_vector()
        distance = cosine(input_embedding, db_embedding)
        if distance < min_distance:
            min_distance = distance
            recognized_face = face_entry

    if recognized_face is not None and min_distance < threshold:
        # logging.info(f"Recognized face: {recognized_face.name}, Distance: {min_distance}")
        return recognized_face
    else:
        # logging.info(f"No match found! (Minimum distance: {min_distance}, Threshold: {threshold})")
        return None


class FaceLogin:
    """
    应用首页，使用PC摄像头
    应用功能包括：
    ···人脸登录
    ···人脸注册
    ···查看已注册、录入人脸
    """

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition', 'landmark_2d_106'],
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))

        # self.model = YOLO(os.path.join(os.path.dirname(__file__), 'best.pt'))
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7,
                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.isOpenPcCamera = False  # 默认开启摄像头
        self.isFaceRecognize = True  # True：开摄像头同时人脸检测
        self.isOpenAlign = False  # 对齐
        # self.firstRecognizedPeople = None  # 是否已经识别过人脸，用于存在多人脸的识别情况

        self.isHandRecognize = False
        # self.isHandPoint = False

        self.isStorageFace = False
        self.name = ""
        self.embedding = []

        self.frame = None
        self.label = ""  # 手势名称标签

        # 新增标志位，指示是否应该继续处理帧
        # 解决开摄像头、关闭后，get_frame_info继续运行再次打开摄像头
        self.isProcessingFrame = False

        self.lock = threading.Lock()

    def __del__(self):
        self.cap.release()

    def align_face(self, face, frame):
        left_eye = face['kps'][0]
        right_eye = face['kps'][1]

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.arctan2(dy, dx) * 180 / np.pi

        center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 1.0 是缩放因子
        rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        bbox = face['bbox']
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]

        target_width = 112
        target_height = 112
        scale_factor = min(target_width / face_width, target_height / face_height)
        resized_face = cv2.resize(rotated_frame, None, fx=scale_factor, fy=scale_factor)

        return resized_face

    def storage_face(self, frame, face, name):
        try:
            embedding = face['embedding']

            # 创建 Face 对象并存储信息到数据库
            face_entry = Face(name=name)
            face_entry.set_feature_vector(embedding)
            face_entry.address = time.time()
            face_entry.save()

            return face_entry
        except Exception as e:
            logging.error(f"存储人脸信息时出错： {e}")
            return None

    def get_frame(self):
        """获取视频帧"""
        if not self.isProcessingFrame:
            return None
        try:
            with self.lock:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("无法从PC摄像头中获取画面")
                    return None
                self.frame = cv2.flip(frame, 1)
                return self.frame
        except Exception as e:
            logging.error(f"获取视频帧时发生错误: \n{e}")
            return None

    def recognize_faces(self, frame):
        """人脸识别"""
        face_count = 0
        face_exists = False
        try:
            with self.lock:
                # 如果人脸登录
                if self.isFaceRecognize:
                    face_results = []  # 存储人脸识别结果
                    try:
                        faces = self.app.get(frame)
                    except Exception as e:
                        logging.error(f"app.get()检测人脸时出错： {e}")
                        faces = []

                    face_count = len(faces)
                    if face_count == 1:
                        recognized_face = compare_face_with_database(faces[0])
                        if recognized_face is not None:
                            face_exists = True
                        # 如果需要存储人脸且只检测到一张人脸
                        if self.isStorageFace:
                            # 人脸对齐
                            aligned_frame = self.align_face(faces[0], frame)
                            aligned_face = self.app.get(aligned_frame)
                            self.storage_face(aligned_frame, aligned_face[0], self.name)
                            self.isStorageFace = False

                    if not faces:
                        face_results.append({"name": "", "bbox": None})
                    else:
                        # 获取图像中所有在数据库中存在的人脸
                        for index, face in enumerate(faces):
                            try:
                                recognized_face = compare_face_with_database(face)
                                if recognized_face is not None:
                                    face_results.append({
                                        "name": recognized_face.name,
                                        "bbox": face['bbox'],
                                    })

                            except Exception as e:
                                logging.error(f"识别人脸 {index} 时出错： {e}")
                                face_results.append({"name": "", "bbox": None})

                        # 如果数据库人脸识别有结果
                        if len(face_results) > 0:
                            name = face_results[0]["name"]
                            bbox = face_results[0]["bbox"]
                            cv2.putText(frame, name, (int(bbox[0]), int(bbox[1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                          (0, 255, 0), 2)

                            if self.isHandRecognize:
                                frame, self.label = hand_recognize(frame, bbox)
                                self.gesture_control()

                            # 其他人统一标红
                            for face in face_results[1:]:
                                name = face["name"]
                                bbox = face['bbox']
                                cv2.putText(frame, name, (int(bbox[0]), int(bbox[1]) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                              (0, 0, 255), 2)
                        # 数据库中没有存入当前图片中的所有人脸
                        else:
                            for face in faces:
                                bbox = face['bbox']
                                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                              (0, 0, 255), 2)
                                cv2.putText(frame, 'Stranger', (int(bbox[0]), int(bbox[1]) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            return frame, face_count, face_exists

        except Exception as e:
            logging.error(f"人脸识别时发生错误: \n{e}")
            return frame, 0, False

    def get_frame_info(self):
        """获取视频帧并进行人脸识别"""
        frame = self.get_frame()
        if frame is None:
            return None, 0, False
        frame, face_count, face_exists = self.recognize_faces(frame)
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes(), face_count, face_exists
        else:
            return None, 0, False


    def gesture_control(self):
        """手势控制无人机"""
        try:
            from myapp.drone import control_drone
            if self.label:  # 只有当有有效手势标签时才发送命令
                control_drone(self.label)
        except Exception as e:
            logging.error(f"手势控制无人机时发生错误: \n{e}")


camera = FaceLogin()


def gen(camera):
    while True:
        if not camera.isProcessingFrame:
            time.sleep(0.1)  # 等待一段时间后再次检查
            continue
        frame, face_count, face_exists = camera.get_frame_info()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            time.sleep(0.1)


def video_feed(request):
    return StreamingHttpResponse(gen(camera), content_type='multipart/x-mixed-replace; boundary=frame')


def turn_pc_camera(request):
    try:
        if camera.isOpenPcCamera:
            camera.cap.release()
            camera.isOpenPcCamera = False
            camera.isProcessingFrame = False
            return JsonResponse({'status': 0, 'message': '关闭电脑摄像头'})
        else:
            camera.cap = cv2.VideoCapture(0)
            camera.isOpenPcCamera = True
            camera.isProcessingFrame = True
            return JsonResponse({'status': 1, 'message': '打开电脑摄像头'})
    except Exception as e:
        logging.error(f"Error turning camera: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


def turn_hand(request):
    try:
        # 使用辅助函数检查无人机是否已连接
        from myapp.drone import is_drone_connected,is_stream_on

        if not is_drone_connected():
            logging.error("无人机未连接或未正确初始化")
            camera.isHandRecognize = False
            return JsonResponse({'status': 0, 'message': '无人机未连接，无法开启手势检测'})

        if not is_stream_on():
            logging.error("无人机摄像头未开启")
            camera.isHandRecognize = False
            return JsonResponse({'status': 0, 'message': '无人机摄像头未开启，无法开启手势检测'})

        if camera.isHandRecognize:
            camera.isHandRecognize = False
            return JsonResponse({'status': 0, 'message': '关闭手势检测'})
        else:
            camera.isHandRecognize = True
            return JsonResponse({'status': 1, 'message': '打开手势检测'})
    except Exception as e:
        logging.error(f"手势检测操作出错: {e}")
        camera.isHandRecognize = False  # 确保出错时关闭手势检测
        return JsonResponse({'status': 'error', 'message': str(e)})

# def turn_hand(request):
#     try:
#         # 重新导入，确保获取最新的global_drone实例
#         from myapp.drone import global_drone
#         # print(global_drone)
#         # print(global_drone.is_connected())

#         # 检查global_drone是否为None
#         if global_drone is None:
#             logging.error("global_drone 为 None，无人机可能未正确初始化")
#             camera.isHandRecognize = False
#             return JsonResponse({'status': 0, 'message': '无人机未连接，无法开启手势检测'})

#         # 安全地调用is_connected方法
#         try:
#             is_connected = global_drone.is_connected()
#             if not is_connected:
#                 camera.isHandRecognize = False
#                 return JsonResponse({'status': 0, 'message': '无人机未连接，无法开启手势检测'})
#         except AttributeError:
#             logging.error("global_drone 没有 is_connected 方法")
#             camera.isHandRecognize = False
#             return JsonResponse({'status': 0, 'message': '无人机对象异常，无法开启手势检测'})

#         if camera.isHandRecognize:
#             camera.isHandRecognize = False
#             return JsonResponse({'status': 0, 'message': '关闭手势检测'})
#         elif not camera.isHandRecognize:
#             camera.isHandRecognize = True
#             return JsonResponse({'status': 1, 'message': '打开手势检测'})
#     except Exception as e:
#         logging.error(f"error :{e}")
#         return JsonResponse({'status': 'error', 'message': str(e)})


@csrf_exempt
def storage_face(request):
    try:
        data = json.loads(request.body)
        name = data.get('name')
        logging.info(f"收到的前端输入name：{name}")

        # 验证人名是否只包含英文和下划线
        if not re.match(r'^[a-zA-Z_]+$', name):
            response = JsonResponse({'status': 0, 'message': '人名只能包含英文和下划线'})
            response["Access-Control-Allow-Origin"] = "http://localhost:9080"
            return response

        # 检查数据库中是否已存在相同名字的人脸
        if Face.objects.filter(name=name).exists():
            response = JsonResponse({'status': 0, 'message': '该名字已存在，请选择不同的名字'})
            response["Access-Control-Allow-Origin"] = "http://localhost:9080"
            return response

        camera.name = name
        camera.isStorageFace = True
        response = JsonResponse({'status': 1, 'message': '人脸录入成功'})
        response["Access-Control-Allow-Origin"] = "http://localhost:9080"
        return response
    except Exception as e:
        response = JsonResponse({'status': 0, 'message': str(e)})
        response["Access-Control-Allow-Origin"] = "http://localhost:9080"
        return response


@csrf_exempt
def get_frame_info(request):
    try:
        frame, face_count, face_exists = camera.get_frame_info()
        if frame is not None:
            response = JsonResponse({
                'status': 1,
                'face_count': face_count,
                'face_exists': face_exists
            })
            response["Access-Control-Allow-Origin"] = "http://localhost:9080"
            return response
        else:
            response = JsonResponse({'status': 0, 'message': '无法获取画面'})
            response["Access-Control-Allow-Origin"] = "http://localhost:9080"
            return response
    except Exception as e:
        logging.error(f"获取人脸信息时出错: {e}")
        response = JsonResponse({'status': 0, 'message': str(e)})
        response["Access-Control-Allow-Origin"] = "http://localhost:9080"
        return response
