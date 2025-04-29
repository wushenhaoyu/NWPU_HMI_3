import time
import cv2
import numpy as np
from collections import deque
import threading
import queue


class PIDController:
    def __init__(self, Kp, Kd, Ki, speed_limit):
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

        self.speed_limit = 50

        self.VIDEO_WIDTH = 960
        self.VIDEO_HEIGHT = 720
        self.target_area = 20000
        self.fb_scale = 100

        # PID 控制器初始化
        self.pid_h = PIDController(Kp=0.25, Kd=0.02, Ki=0.005, speed_limit=self.speed_limit)
        self.pid_v = PIDController(Kp=0.25, Kd=0.02, Ki=0.005, speed_limit=self.speed_limit)
        self.pid_fb = PIDController(Kp=0.25, Kd=0.02, Ki=0.005, speed_limit=self.speed_limit)

        self.last_time = time.time()

        self.visualizer = PIDVisualizer(self.pid_h.speed_limit)
        self.visualization_enabled = False

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

        error_h = x - self.VIDEO_WIDTH // 2
        error_v = y - self.VIDEO_HEIGHT // 2
        error_fb = (area - self.target_area) / self.fb_scale

        if x == 0 or y == 0:
            self.pid_h.reset()
            self.pid_v.reset()
            self.pid_fb.reset()
            self.tello.send_rc_control(0, 0, 0, 0)
            return

        speed_h = self.pid_h.compute(error_h, dt)
        speed_v = self.pid_v.compute(error_v, dt)
        speed_fb = self.pid_fb.compute(error_fb, dt)

        # print(f"speed_fb: {speed_fb}, speed_v: {-speed_v}, speed_h: {speed_h}, area: {area}")
        self.tello.send_rc_control(0, -speed_fb, -speed_v, speed_h)

        if self.visualization_enabled:
            self.visualizer.plot_queue.put((error_h, error_v, error_fb, speed_h, -speed_v, -speed_fb))


class PIDVisualizer:
    def __init__(self, speed_limit, width=800, height_per_row=150, max_points=50):
        self.width = width
        self.height_per_row = height_per_row
        self.spacing_y = 20
        self.spacing_x = 50
        self.max_points = max_points
        self.speed_limit = speed_limit

        self.errors_h = deque(maxlen=max_points)
        self.errors_v = deque(maxlen=max_points)
        self.errors_fb = deque(maxlen=max_points)
        self.speed_h = deque(maxlen=max_points)
        self.speed_v = deque(maxlen=max_points)
        self.speed_fb = deque(maxlen=max_points)

        self.plot_queue = queue.Queue()
        self.is_running = False
        self.thread = None
        self.window_name = "PID Visualization"

        self.axis_color = (150, 150, 150)
        self.error_color = (0, 0, 255)
        self.speed_color = (0, 255, 0)

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self.plot_loop)
            self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
            self.clear()
            cv2.destroyWindow(self.window_name)
            self.thread = None

    def plot_loop(self):
        while self.is_running:
            try:
                data = self.plot_queue.get(timeout=0.1)
                self.update(*data)
                self.draw()
            except queue.Empty:
                continue

    def update(self, err_h, err_v, err_fb, speed_h, speed_v, speed_fb):
        self.errors_h.append(err_h)
        self.errors_v.append(err_v)
        self.errors_fb.append(err_fb)
        self.speed_h.append(speed_h)
        self.speed_v.append(speed_v)
        self.speed_fb.append(speed_fb)

    def clear(self):
        self.errors_h.clear()
        self.errors_v.clear()
        self.errors_fb.clear()
        self.speed_h.clear()
        self.speed_v.clear()
        self.speed_fb.clear()

    def draw_single_plot(self, canvas, data, top, left, width, height, color, label):
        right = left + width
        bottom = top + height
        center_y = (top + bottom) // 2
        label_x = left + (width // 2) - 20
        label_y = bottom + 15

        # 画背景网格
        num_horizontal_lines = 4
        num_vertical_lines = 4
        grid_color = (220, 220, 220)

        # 横线
        for i in range(1, num_horizontal_lines):
            y = top + i * height // num_horizontal_lines
            cv2.line(canvas, (left, y), (right, y), grid_color, 1, lineType=cv2.LINE_AA)

        # 竖线
        for i in range(1, num_vertical_lines):
            x = left + i * width // num_vertical_lines
            cv2.line(canvas, (x, top), (x, bottom), grid_color, 1, lineType=cv2.LINE_AA)

        # 坐标轴线
        cv2.line(canvas, (left, center_y), (right, center_y), self.axis_color, 1)
        cv2.line(canvas, (left, top), (left, bottom), self.axis_color, 1)

        # 计算纵轴缩放
        if len(data) > 0:
            data_max = max(abs(max(data)), abs(min(data)))
            if data_max < 1e-6:
                data_max = 1
        else:
            data_max = 1

        # 画曲线
        if len(data) > 1:
            points = []
            for i, value in enumerate(data):
                x = left + int(i * (width) / (len(data) - 1))
                y = int(center_y - (value * (height // 2 - 5) / data_max))
                points.append((x, y))
            cv2.polylines(canvas, [np.int32(points)], False, color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_color = (80, 80, 80)

        max_text = f"{+data_max:.1f}"
        cv2.putText(canvas, max_text, (left + 5, top + 10),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        cv2.putText(canvas, "0", (left + 5, center_y - 2),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        min_text = f"{-data_max:.1f}"
        cv2.putText(canvas, min_text, (left + 5, bottom - 5),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        cv2.putText(canvas, label, (label_x,label_y),
                    font, font_scale, color, font_thickness, cv2.LINE_AA)

    def draw(self):
        rows = 3
        cols = 2
        total_height = rows * self.height_per_row + (rows + 1) * self.spacing_y
        total_width = self.width
        single_width = (total_width - 3 * self.spacing_x) // 2

        canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

        plots = [
            (self.errors_v, self.speed_v, "error_v", "speed_v"),
            (self.errors_h, self.speed_h, "error_h", "speed_h"),
            (self.errors_fb, self.speed_fb, "error_fb", "speed_fb"),
        ]

        for i, (error_data, speed_data, error_label, speed_label) in enumerate(plots):
            top = self.spacing_y + i * (self.height_per_row + self.spacing_y)

            self.draw_single_plot(canvas,
                                  error_data,
                                  top,
                                  self.spacing_x,
                                  single_width,
                                  self.height_per_row,
                                  self.error_color,
                                  error_label)

            self.draw_single_plot(canvas,
                                  speed_data,
                                  top,
                                  self.spacing_x * 2 + single_width,
                                  single_width,
                                  self.height_per_row,
                                  self.speed_color,
                                  speed_label)

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)
