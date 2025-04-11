from collections import deque
import mediapipe as mp
import numpy as np
import cv2
import logging

# 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',  # 定义日志格式
#     datefmt='%Y-%m-%d %H:%M:%S'  # 定义时间格式
# )

# 调整ROI参数（右侧位置）
ROI_WIDTH_RATIO = 1.5
ROI_HEIGHT_RATIO = 1.5
ROI_OFFSET_X_RATIO = 0.5  # 正数表示右侧偏移
ROI_OFFSET_Y_RATIO = 0.2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 手势检测配置需要接收返回的左右手信息
hands_detection = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # 需要检测双手以准确判断左右
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 手势缓冲队列
gesture_buffer = deque(maxlen=5)  # 保存最近5帧的手势结果


def get_gesture(image, lms_list):
    """
    :param image: 视频图像
    :param lms_list: 手指关节点

    :return: 手势名称
    """
    lms_list = np.array(lms_list, dtype=np.int32)
    # 构造凸包
    hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
    hull = cv2.convexHull(lms_list[hull_index, :])
    # 绘制凸包轮廓
    cv2.polylines(image, [hull], True, (0, 255, 0), 2)

    # 指尖（凸包外部）索引
    pip_index = [4, 8, 12, 16, 20]
    # 伸直的手指
    up_fingers = []
    for i in pip_index:
        pt = (int(lms_list[i][0]), int(lms_list[i][1]))
        # 判断指尖是否在凸包外
        # 返回值：正数=内部，0=边界，负数=外部
        dist = cv2.pointPolygonTest(hull, pt, True)
        if dist < 0:
            up_fingers.append(i)

    if len(up_fingers) == 0:
        return "takeoff"
    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 20:
        return "land"
    elif len(up_fingers) == 5:
        return "stop"
    elif len(up_fingers) == 1 and up_fingers[0] == 4:
        return "left"
    elif len(up_fingers) == 1 and up_fingers[0] == 20:
        return "right"
    elif len(up_fingers) == 1 and up_fingers[0] == 8:
        return "up"
    elif len(up_fingers) == 2 and up_fingers[0] == 8 and up_fingers[1] == 12:
        return "down"
    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 8:
        return "forward"
    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 12:
        return "backward"
    elif len(up_fingers) == 3 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16:
        return "rotate_left"
    elif len(up_fingers) == 4 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16 and up_fingers[3] == 20:
        return "rotate_right"
    else:
        return ""


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
    # roi_x = max(0, min(roi_x, iw - roi_w))
    # roi_y = max(0, min(roi_y, ih - roi_h))
    roi_x = max(0, min(roi_x, iw))
    roi_y = max(0, min(roi_y, ih))

    return roi_x, roi_y, roi_w, roi_h


def hand_recognize(frame, bbox):
    """
    :param frame 要处理的画面
    :param bbox 人脸的bbox

    :return 绘制了手部ROI框和手势名称的画面
    """
    final_gesture = ""
    try:
        roi_x, roi_y, roi_w, roi_h = get_hand_roi(frame, bbox)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0),2)
        hand_roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # cv2.imshow("hand_roi", hand_roi)
        if hand_roi.size == 0:
            print("hand_roi is empty")
            return frame, ""

        hand_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
        hand_results = hands_detection.process(hand_roi)
        if not hand_results.multi_hand_landmarks:
            # logging.warning("No hand landmarks detected")
            return frame, ""

        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            # 转换坐标到原始图像
            adjusted_landmarks = [(roi_x + int(lm.x * roi_w),
                                   roi_y + int(lm.y * roi_h))
                                  for lm in hand_landmarks.landmark]

            # 绘制右手关键点（使用自定义颜色）
            for point in adjusted_landmarks:
                cv2.circle(frame, point, 4, (0, 0, 255), -1)

            # 绘制骨骼连接线（蓝色）
            connections = mp_hands.HAND_CONNECTIONS
            for connection in connections:
                start = connection[0]
                end = connection[1]
                cv2.line(frame,
                         adjusted_landmarks[start],
                         adjusted_landmarks[end],
                         (255, 255, 255), 2)

            str_gesture = get_gesture(frame, adjusted_landmarks)
            gesture_buffer.append(str_gesture)

            # 检查缓冲队列中的手势是否一致
            if len(gesture_buffer) == gesture_buffer.maxlen and len(set(gesture_buffer)) == 1:
                final_gesture = gesture_buffer[0]
                cv2.putText(frame, final_gesture,
                            (roi_x + 20, roi_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                            color=(0, 0, 255), thickness=2)

        return frame, final_gesture

    except Exception as e:
        logging.warning(f"hand_recognize error: {e}")
        return frame, ""








