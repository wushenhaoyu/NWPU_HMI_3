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
import re
import time
import logging
from ultralytics import YOLO
import mediapipe as mp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',  # 定义日志格式
    datefmt='%Y-%m-%d %H:%M:%S'  # 定义时间格式
)
# 调整ROI参数（右侧位置）
ROI_WIDTH_RATIO = 1.8
ROI_HEIGHT_RATIO = 1.5
ROI_OFFSET_X_RATIO = 1.2  # 正数表示右侧偏移
ROI_OFFSET_Y_RATIO = 0

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


def get_hand_roi(frame, bbox):
    iw, ih = frame.shape[:2]
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

    return roi_x, roi_y, roi_w, roi_h



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
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7,
                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.isOpenPcCamera = False  # 默认开启摄像头
        self.isFaceRecognize = True  # True：开摄像头同时人脸检测
        self.isOpenAlign = False  # 对齐
        self.firstRecognizedPeople = None  # 是否已经识别过人脸，用于存在多人脸的识别情况

        self.isHandRecognize = False
        self.isHandPoint = False

        self.isStorageFace = False
        self.name = ""
        self.embedding = []

        self.frame = None
        self.label = "" # 手势名称标签

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
                    recognized_face = compare_face_with_database(faces[0])
                    if recognized_face is not None:
                        face_exists = True
                    # 如果需要存储人脸且只检测到一张人脸
                    if self.isStorageFace:
                        # 人脸对齐
                        aligned_frame = self.align_face(faces[0], self.frame)
                        aligned_face = self.app.get(aligned_frame)
                        self.storage_face(aligned_frame, aligned_face[0], self.name)
                        self.isStorageFace = False

                if not faces:
                    # logging.info("没有检测到人脸")
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
                        # self.firstRecognizedPeople = face_results[0]
                        # logging.info(f"识别到的人脸为：{self.firstRecognizedPeople['name']}")
                        # 以识别出的第一个人为准
                        name = face_results[0]["name"]
                        bbox = face_results[0]["bbox"]
                        cv2.putText(self.frame, name, (int(bbox[0]), int(bbox[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.rectangle(self.frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      (0, 255, 0), 2)
                        if self.isHandRecognize:
                            roi_x, roi_y, roi_w, roi_h = get_hand_roi(self.frame, bbox)
                            # 手
                            cv2.rectangle(self.frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h),
                                          (255, 0, 0), 2)

                            detect_frame = self.frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                            detect_hands = self.model(detect_frame, stream=True)
                            for detection in detect_hands:
                                for box in detection.boxes:
                                    # 获取边界框坐标
                                    # x1, y1, x2, y2 = map(int, box.xyxy[0])  # 转换为整数
                                    conf = box.conf[0]  # 置信度
                                    cls = int(box.cls[0])  # 类别索引
                                    label = self.model.names[cls]
                                    ges_info = f"{label} {conf:.2f}"  # 获取类别名称和置信度

                                    # 绘制边界框和类别
                                    # cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(self.frame, ges_info, (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (0, 255, 0), 2)
                        if self.isHandPoint:
                            roi_x, roi_y, roi_w, roi_h = get_hand_roi(self.frame, bbox)
                            roi_area = self.frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                            # if roi_area.size != 0:
                            # roi_rgb = cv2.cvtColor(roi_area, cv2.COLOR_BGR2RGB)
                            hand_results = self.hands.process(roi_area)

                            if hand_results.multi_hand_landmarks:
                            #     for hand_landmarks in hand_results.multi_hand_landmarks:
                            #         self.mp_drawing.draw_landmarks(
                            #             self.frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            #             self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                            #             self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                            #         )
                                for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                                    # 获取手部属性
                                    handedness = hand_results.multi_handedness[idx]
                                    hand_label = handedness.classification[0].label

                                    # 仅处理右手检测结果
                                    # if hand_label == ("Right" if ROI_OFFSET_X_RATIO > 0 else "Left"):
                                    # 转换坐标到原始图像
                                    adjusted_landmarks = [(roi_x + int(lm.x * roi_w),
                                                           roi_y + int(lm.y * roi_h))
                                                          for lm in hand_landmarks.landmark]

                                    # 绘制右手关键点（使用自定义颜色）
                                    for point in adjusted_landmarks:
                                        cv2.circle(self.frame, point, 4, (0, 0, 255), -1)

                                    # 绘制骨骼连接线（蓝色）
                                    connections = self.mp_hands.HAND_CONNECTIONS
                                    for connection in connections:
                                        start = connection[0]
                                        end = connection[1]
                                        cv2.line(self.frame,
                                                 adjusted_landmarks[start],
                                                 adjusted_landmarks[end],
                                                 (255, 255, 255), 2)

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

            ret, jpeg = cv2.imencode('.jpg', self.frame)
            return jpeg.tobytes(), face_count, face_exists

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


def turn_camera(request):
    try:
        if camera.isOpenPcCamera:
            camera.cap.release()
            camera.isOpenPcCamera = False
            return JsonResponse({'status': 0, 'message': 'Camera turned successfully'})
        elif not camera.isOpenPcCamera:
            camera.cap = cv2.VideoCapture(0)
            camera.isOpenPcCamera = True
            return JsonResponse({'status': 1, 'message': 'Camera turned successfully'})
    except Exception as e:
        logging.error(f"Error turning camera: {e}")  # 增加日志记录
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

def turn_off_camera(request):
    try:
        if camera.isOpenPcCamera:
            camera.cap.release()
            camera.isOpenPcCamera = False
            return JsonResponse({'status': 1, 'message': 'Camera turned off successfully'})
        else:
            return JsonResponse({'status': 0, 'message': 'Camera is already off'})
    except Exception as e:
        logging.error(f"Error turning off camera: {e}")  # 增加日志记录
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
                'status': 1,
                'face_count': face_count,
                'face_exists': face_exists
            })
            response["Access-Control-Allow-Origin"] = "http://localhost:9080"
            return response
        else:
            response = JsonResponse({'status': 0, 'message': '无法获取画面'}, status=500)
            response["Access-Control-Allow-Origin"] = "http://localhost:9080"
            return response
    except Exception as e:
        logging.error(f"获取人脸信息时出错: {e}")
        response = JsonResponse({'status': 0, 'message': str(e)}, status=500)
        response["Access-Control-Allow-Origin"] = "http://localhost:9080"
        return response
