import time
import cv2
import numpy as np
from collections import deque
import threading
import queue


class PIDController:
    def __init__(self, Kp, Kd, Ki, speed_limit=40):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.speed_limit = speed_limit

        self.integral = 0
        self.prev_error = 0

    def reset(self):
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        self.integral += error * dt
        self.integral = np.clip(self.integral, -100, 100)  # 防积分饱和
        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        output = self.Kp * error + self.Kd * derivative + self.Ki * self.integral
        output = int(np.clip(output, -self.speed_limit, self.speed_limit))
        self.prev_error = error
        return output


class FaceTracker:
    def __init__(self, tello, face_detector):
        self.tello = tello
        self.faceDetect = face_detector
        self.frame = None

        self.VIDEO_WIDTH = 960
        self.VIDEO_HEIGHT = 720
        self.FBRANGE = [30000, 50000]  # 控制前后移动

        # PID 控制器初始化
        self.pid_h = PIDController(Kp=0.25, Kd=0.02, Ki=0.005)
        self.pid_v = PIDController(Kp=0.25, Kd=0.02, Ki=0.005)

        self.last_time = time.time()

        self.visualizer = PIDVisualizer(self.pid_h.speed_limit)
        self.visualization_enabled = False  # 新增可视化开关

    def __del__(self):
        cv2.destroyAllWindows()

    def find_face(self, frame):
        faces = self.faceDetect.get(frame)
        if not faces:
            return frame, [[0, 0], 0]

        max_face = max(faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))
        x1, y1, x2, y2 = map(int, max_face['bbox'])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
        cv2.line(frame, (cx, cy), (self.VIDEO_WIDTH // 2, self.VIDEO_HEIGHT // 2), (0, 0, 255), 2)

        return frame, [[cx, cy], area]

    def track(self, face_info):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        (x, y), area = face_info
        speed_fb = 0

        error_h = x - self.VIDEO_WIDTH // 2
        error_v = y - self.VIDEO_HEIGHT // 2

        if x == 0 or y == 0:
            self.pid_h.reset()
            self.pid_v.reset()
            # self.tello.send_rc_control(0, 0, 0, 0)
            return

        speed_h = self.pid_h.compute(error_h, dt)
        speed_v = self.pid_v.compute(error_v, dt)

        if self.FBRANGE[0] < area < self.FBRANGE[1]:
            speed_fb = 0
        elif area > self.FBRANGE[1]:
            speed_fb = -20
        elif area < self.FBRANGE[0] and area != 0:
            speed_fb = 20

        print(f"speed_fb: {speed_fb}, speed_v: {-speed_v}, speed_h: {speed_h}, area: {area}")
        # self.tello.send_rc_control(0, speed_fb, -speed_v, speed_h)

        if self.visualization_enabled:
            self.visualizer.plot_queue.put((error_h, error_v, speed_h, -speed_v, speed_fb))


class PIDVisualizer:
    def __init__(self, speed_limit, max_points=50):
        self.max_points = max_points
        self.speed_limit = speed_limit
        
        self.errors_h = deque(maxlen=max_points)
        self.errors_v = deque(maxlen=max_points)
        self.speed_h = deque(maxlen=max_points)
        self.speed_v = deque(maxlen=max_points)
        self.speed_fb = deque(maxlen=max_points)

        self.plot_queue = queue.Queue()
        self.is_running = False
        self.thread = None
        self.window_name = "PID Visualization"

        # 图表参数
        self.canvas_size = (800, 600)
        self.bg_color = (255, 255, 255)
        self.line_color = (0, 0, 0)
        self.error_h_color = (0, 255, 0)        # 绿
        self.error_v_color = (255, 255, 0)      # 天蓝
        self.speed_h_color = (0, 128, 255)      # 橙
        self.speed_v_color = (0, 0, 255)      # 黄
        self.speed_fb_color = (255, 0, 255)     # 粉

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._plot_loop)
            self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
            cv2.destroyWindow(self.window_name)
            self.thread = None

    def _plot_loop(self):
        while self.is_running:
            try:
                data = self.plot_queue.get(timeout=0.1)
                self.update(*data)
                self._draw()
            except queue.Empty:
                continue

    def update(self, err_h, err_v, speed_h, speed_v, speed_fb):
        self.errors_h.append(err_h)
        self.errors_v.append(err_v)
        self.speed_h.append(speed_h)
        self.speed_v.append(speed_v)
        self.speed_fb.append(speed_fb)

    def _draw(self):
        canvas = np.full((self.canvas_size[1], self.canvas_size[0], 3), self.bg_color, dtype=np.uint8)

        max_error_h = max(max(self.errors_h, default=0), abs(min(self.errors_h, default=0)))
        max_error_v = max(max(self.errors_v, default=0), abs(min(self.errors_v, default=0)))

        error_range_h = max(200, int(max_error_h * 1.2))  # 保证最小范围是200
        error_range_v = max(200, int(max_error_v * 1.2))

        graphs = [
            ("Error H", self.errors_h, self.error_h_color, error_range_h),
            ("Error V", self.errors_v, self.error_v_color, error_range_v),
            ("Speed H", self.speed_h, self.speed_h_color, self.speed_limit),  # 行2列1
            ("Speed V", self.speed_v, self.speed_v_color, self.speed_limit),  # 行2列2
            ("Speed FB", self.speed_fb, self.speed_fb_color, 30),  # 行3列中间
        ]

        rows = 3
        cols = 2
        spacing_x = 20
        spacing_y = 20
        graph_w = (self.canvas_size[0] - (cols + 1) * spacing_x) // cols
        graph_h = (self.canvas_size[1] - (rows + 1) * spacing_y) // rows

        for i, (label, data, color, limit) in enumerate(graphs):
            if i < 4:
                row = i // 2
                col = i % 2
            else:
                row = 2
                col = 0  # 第3行居中显示
                graph_w = self.canvas_size[0] - 2 * spacing_x

            left = spacing_x + col * (graph_w + spacing_x)
            top = spacing_y + row * (graph_h + spacing_y)
            right = left + graph_w
            bottom = top + graph_h
            center_y = (top + bottom) // 2
            label_x = left + (graph_w // 2) - 20
            label_y = bottom + 15

            # 绘制坐标轴
            cv2.line(canvas, (left, center_y), (right, center_y), self.line_color, 1)
            cv2.line(canvas, (left, top), (left, bottom), self.line_color, 1)

            # 绘制曲线
            if len(data) > 1:
                points = [
                    (
                        left + int(i * graph_w / len(data)),
                        int(center_y - (v * (graph_h // 2 - 5) / limit))
                    )
                    for i, v in enumerate(data)
                ]
                cv2.polylines(canvas, [np.int32(points)], False, color, 2)

            # Y轴标签和图名
            cv2.putText(canvas, f"{limit}", (left + 5, top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"0", (left + 5, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"-{limit}", (left + 5, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{label}", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)
