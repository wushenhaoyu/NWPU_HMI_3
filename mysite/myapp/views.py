import json
import os
from django.http import JsonResponse, StreamingHttpResponse
import cv2
import insightface
from insightface.app import FaceAnalysis
import torch
import numpy as np
from scipy.spatial.distance import cosine
from django.views.decorators.csrf import csrf_exempt
from myapp.models import Face
from ultralytics import YOLO
from myapp.live import eye_aspect_ratio, mouth_aspect_ratio, nose_jaw_distance
from djitellopy import Tello
import time
import logging
import re

# 调整ROI参数（右侧位置）
ROI_WIDTH_RATIO = 1.2
ROI_HEIGHT_RATIO = 1.5
ROI_OFFSET_X_RATIO = 1.2  # 正数表示右侧偏移
ROI_OFFSET_Y_RATIO = 0.8

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',  # 定义日志格式
    datefmt='%Y-%m-%d %H:%M:%S'  # 定义时间格式
)


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
        self.model = YOLO(os.path.join(os.path.dirname(__file__), 'best.pt'))

        self.isOpenPcCamera = True  # 默认开启摄像头
        self.isFaceRecognize = False  # 人脸检测
        self.isOpenAlign = False  # 对齐
        self.firstRecognizedPeople = None  # 是否已经识别过人脸，用于存在多人脸的识别情况

        self.isStorageFace = False
        self.name = ""
        self.embedding = []

        # self.isHandRecognize = False
        # self.isHandPoint = False

        self.frame = None

    def __del__(self):
        self.cap.release()

    def align_face(self, face, frame):
        left_eye = face['kps'][0]
        right_eye = face['kps'][1]

        eye_dist = np.linalg.norm(np.array(right_eye) - np.array(left_eye))
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.arctan2(dy, dx) * 180 / np.pi

        center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 1.0 是缩放因子
        rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        rotated_left_eye = np.dot(M[:, :2], np.array(left_eye).T) + M[:, 2]
        rotated_right_eye = np.dot(M[:, :2], np.array(right_eye).T) + M[:, 2]

        bbox = face['bbox']
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]

        target_width = 112
        target_height = 112
        scale_factor = min(target_width / face_width, target_height / face_height)
        resized_face = cv2.resize(rotated_frame, None, fx=scale_factor, fy=scale_factor)

        return resized_face

    def storage_face(self, frame, face, name):
        if not os.path.exists('database'):
            os.mkdir('database')
        bbox = face['bbox']
        x1, y1, x2, y2 = bbox[:]
        embedding = face['embedding']

        # 裁剪出人脸图像
        face_image = frame[int(y1):int(y2), int(x1):int(x2)]

        # 生成一个唯一的图片名称

        # 创建 Face 对象并存储信息到数据库
        face_entry = Face(name=name, address="1")
        face_entry.set_feature_vector(embedding)
        face_entry.save()

        image_filename = f"database/{face_entry.id}/{name}_{str(np.random.randint(1000))}.jpg"
        face_entry.address = image_filename
        face_entry.save()
        # 将人脸图像保存到文件
        cv2.imwrite(image_filename, face_image)

        logging.info(f"Stored face info for {name} at {image_filename}")
        return face_entry

    def compare_face_with_database(self, face, threshold=0.4):
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

    def get_frame_info(self):
        """打开人脸识别开关，进行人脸登录
        只允许画面存在一个人，多人时不识别
        Returns:
            1.经过识别并标注的人脸frame
            2.人脸数量
            3.人脸是否在数据库中存在（只有一个人时）
        """
        try:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("无法从PC摄像头中获取画面，尝试重新打开摄像头")
                self.cap.release()
                self.cap = cv2.VideoCapture(0)
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("重新打开摄像头后仍然无法获取画面")
                    return None
            self.frame = frame

            face_count = 0
            face_exists = False

            # 如果人脸登录
            if self.isFaceRecognize:
                face_results = []  # 存储人脸识别结果
                try:
                    faces = self.app.get(self.frame)
                except Exception as e:
                    logging.error(f"app.get()检测人脸时出错： {e}")
                    faces = []

                face_count = len(faces)
                if face_count == 1:
                    recognized_face = self.compare_face_with_database(faces[0])
                    if recognized_face is not None:
                        face_exists = True
                    # 如果需要存储人脸且只检测到一张人脸
                    if self.isStorageFace:
                        self.storage_face(self.frame, faces[0], self.name)
                        self.isStorageFace = False

                if not faces:
                    # logging.info("没有检测到人脸")
                    face_results.append({"name": "", "bbox": None})
                else:
                    # 获取图像中所有在数据库中存在的人脸
                    for index, face in enumerate(faces):
                        try:
                            recognized_face = self.compare_face_with_database(face)
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
                        self.firstRecognizedPeople = face_results[0]
                        # logging.info(f"识别到的人脸为：{self.firstRecognizedPeople['name']}")
                        # 以识别出的第一个人为准
                        name = self.firstRecognizedPeople["name"]
                        bbox = self.firstRecognizedPeople["bbox"]
                        cv2.putText(self.frame, name, (int(bbox[0]), int(bbox[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.rectangle(self.frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      (0, 255, 0), 2)
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

            ret, jpeg = cv2.imencode('.jpg', self.frame)
            return jpeg.tobytes(), face_count, face_exists

            # else:
            #     ret, jpeg = cv2.imencode('.jpg', self.frame)
            #     return jpeg.tobytes()
        except Exception as e:
            logging.warning(f"从PC摄像头中获取画面时发生错误: \n{e}")
            return None, 0, False


def gen(camera):
    while True:
        frame, face_count, face_exists = camera.get_frame_info()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


camera = FaceLogin()


def video_feed(request):
    return StreamingHttpResponse(gen(camera), content_type='multipart/x-mixed-replace; boundary=frame')


# def turn_camera(request):
#     try:
#         if camera.isOpenPcCamera:
#             camera.video.release()
#             camera.isOpenPcCamera = False
#             return JsonResponse({'status': 0, 'message': 'Camera turned successfully'})
#         elif not camera.isOpenPcCamera:
#             camera.video = cv2.VideoCapture(0)
#             camera.isOpenPcCamera = True
#             return JsonResponse({'status': 1, 'message': 'Camera turned successfully'})
#     except Exception as e:
#         return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def turn_face(request):
    try:
        if camera.isFaceRecognize:
            camera.isFaceRecognize = False
            return JsonResponse({'status': 0, 'message': 'Face detection turned off'})
        elif not camera.isFaceRecognize:
            camera.isFaceRecognize = True
            return JsonResponse({'status': 1, 'message': 'Face detection turned on'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def turn_point(request):
    try:
        if camera.isOpenPoint:
            camera.isOpenPoint = False
            return JsonResponse({'status': 0, 'message': 'Face point turned off'})
        elif not camera.isOpenPoint:
            camera.isOpenPoint = True
            return JsonResponse({'status': 1, 'message': 'Face point turned on'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def turn_align(request):
    try:
        if camera.isOpenAlign:
            camera.isOpenAlign = False
            return JsonResponse({'status': 0, 'message': 'Face align turned off'})
        elif not camera.isOpenAlign:
            camera.isOpenAlign = True
            return JsonResponse({'status': 1, 'message': 'Face align turned on'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def turn_hand(request):
    try:
        if camera.isHandRecognize:
            camera.isHandRecognize = False
            return JsonResponse({'status': 0, 'message': 'Hand detection turned off'})
        elif not camera.isHandRecognize:
            camera.isHandRecognize = True
            return JsonResponse({'status': 1, 'message': 'Hand detection turned on'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def turn_hand_point(request):
    try:
        if camera.isHandPoint:
            camera.isHandPoint = False
            return JsonResponse({'status': 0, 'message': 'Hand point turned off'})
        elif not camera.isHandPoint:
            camera.isHandPoint = True
            return JsonResponse({'status': 1, 'message': 'Hand point turned on'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


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
        response = JsonResponse({'status': 1, 'message': 'Storage face turned on'})
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
                'status': 'success',
                'face_count': face_count,
                'face_exists': face_exists
            })
            response["Access-Control-Allow-Origin"] = "http://localhost:9080"
            return response
        else:
            response = JsonResponse({'status': 'error', 'message': '无法获取画面'}, status=500)
            response["Access-Control-Allow-Origin"] = "http://localhost:9080"
            return response
    except Exception as e:
        logging.error(f"获取人脸信息时出错: {e}")
        response = JsonResponse({'status': 'error', 'message': str(e)}, status=500)
        response["Access-Control-Allow-Origin"] = "http://localhost:9080"
        return response
