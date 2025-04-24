import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class PIDController:
    def __init__(self, Kp, Kd, Ki, output_limit=20):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.output_limit = output_limit

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
        output = int(np.clip(output, -self.output_limit, self.output_limit))
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

        # self.visualizer = PIDVisualizer()
        # plt.ion()

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
        fb = 0

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
            fb = 0
        elif area > self.FBRANGE[1]:
            fb = -20
        elif area < self.FBRANGE[0] and area != 0:
            fb = 20

        print(f"fb: {fb}, speed_v: {-speed_v}, speed_h: {speed_h}, area: {area}")
        # self.tello.send_rc_control(0, fb, -speed_v, speed_h)

        # self.visualizer.update(error_h, error_v, speed_h, speed_v, fb)
        # self.visualizer.plot()


class PIDVisualizer:
    def __init__(self, max_len=100):
        self.errors_h = deque(maxlen=max_len)
        self.errors_v = deque(maxlen=max_len)
        self.outputs_h = deque(maxlen=max_len)
        self.outputs_v = deque(maxlen=max_len)
        self.fb_cmds = deque(maxlen=max_len)

    def update(self, err_h, err_v, out_h, out_v, fb):
        self.errors_h.append(err_h)
        self.errors_v.append(err_v)
        self.outputs_h.append(out_h)
        self.outputs_v.append(out_v)
        self.fb_cmds.append(fb)

    def plot(self):
        plt.clf()
        plt.subplot(3, 1, 1)
        plt.plot(self.errors_h, label="Horizontal Error")
        plt.plot(self.errors_v, label="Vertical Error")
        plt.legend()
        plt.ylabel("Error")

        plt.subplot(3, 1, 2)
        plt.plot(self.outputs_h, label="H Speed")
        plt.plot(self.outputs_v, label="V Speed")
        plt.legend()
        plt.ylabel("Output")

        plt.subplot(3, 1, 3)
        plt.plot(self.fb_cmds, label="FB Command")
        plt.legend()
        plt.ylabel("FB")

        plt.pause(0.01)
