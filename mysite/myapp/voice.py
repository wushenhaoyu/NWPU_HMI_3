import logging
import os
import wave
import torch
import opencc
import librosa
import whisper
import pyaudio
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn
from threading import Lock
from contextlib import contextmanager
from django.http import JsonResponse
from pyaudio import PyAudio

from .commands import COMMANDS_MAP_CN, COMMANDS_MAP_CNN

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2
TOTAL_FRAMES = int(RATE / CHUNK * RECORD_SECONDS)

AUDIO_PATH = os.path.join(os.path.dirname(__file__), 'output.wav')

COLOR_CODE = "\033[37m"


# 上下文管理器确保 PyAudio 资源正确释放
@contextmanager
def pyaudio_stream(audio, format_, channels, rate, input_=True, frames_per_buffer=CHUNK):
    stream = audio.open(format=format_,
                        channels=channels,
                        rate=rate,
                        input=input_,
                        frames_per_buffer=frames_per_buffer)
    try:
        yield stream
    finally:
        stream.stop_stream()
        stream.close()


# 录音函数优化
def record_save(output_path):
    pa = PyAudio()
    frames = []

    with pyaudio_stream(pa, FORMAT, CHANNELS, RATE) as stream:
        print('Start recording.')
        # for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        for _ in tqdm(range(TOTAL_FRAMES), desc=f"{COLOR_CODE}Recording",
                      bar_format="{l_bar}{bar}",
                      colour="white"):
            data = stream.read(CHUNK)
            frames.append(data)
        print('End recording.')

    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))


class VoiceCNN(nn.Module):
    def __init__(self):
        super(VoiceCNN, self).__init__()

        # Conv2d采用NCHW格式，N代表批数据图像数量（batch_size），C代表通道数，H、W分别代表图像高和宽
        # 不考虑批数据维度时，第一个维度为通道数
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=1)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout(0.5)  # 添加 Dropout 层
        self.fc1 = nn.Linear(256 * 27 * 1, 2048)
        self.fc2 = nn.Linear(2048, 5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = Lock()
        self.load_weight()
        self.valid = False  # 是否开启语音控制
        self.isRecording = False

    def load_weight(self):
        model_weights_path = os.path.join(os.path.dirname(__file__), 'best_model_weights.pth')
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Model weights file not found at {model_weights_path}")
        self.to(self.device)
        self.load_state_dict(torch.load(model_weights_path, map_location=self.device, weights_only=True))
        self.eval()

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.pool1(self.relu1(self.conv1(x)))
        # print("After pool1: ", x.shape)
        x = self.pool2(self.relu2(self.conv2(x)))
        # print("After pool2: ", x.shape)
        x = self.pool3(self.relu3(self.conv3(x)))
        # print("After pool3: ", x.shape)
        x = self.pool4(self.relu4(self.conv4(x)))

        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)  # 应用Dropout层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def run(self):
        if self.isRecording:
            return
        self.isRecording = True
        try:
            record_save(AUDIO_PATH)

            # 加载音频并提取特征
            data, sr = librosa.load(AUDIO_PATH, sr=None)
            if len(data) == 0:
                raise ValueError("Audio data is empty")
            data = librosa.effects.preemphasis(data)
            mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13).T
            mfcc = zero_pad(mfcc)
            mfcc = torch.tensor(mfcc).unsqueeze(0).to(self.device)

            self.isRecording = False
            prediction = self(mfcc)
            command_index = torch.argmax(prediction).item()

            command = COMMANDS_MAP_CNN[command_index]
            print(f"command_index: {command_index}, command: {command}")
            self.isRecording = False
            return command
        except Exception as e:
            logging.error(f'Error in run: {e}')  # 添加日志
            self.isRecording = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            return None


def zero_pad(feature, max_length=216):
    # 计算需要填充的长度
    difference = max_length - feature.shape[0]
    # 对单个特征进行零填充
    padded_feature = np.pad(feature, ((0, difference), (0, 0)), "constant")
    return padded_feature


class VoiceWhisper:
    def __init__(self):
        self.model = whisper.load_model("small")
        self.converter = opencc.OpenCC('t2s.json')  # 繁体转简体
        self.isRecording = False
        self.valid = False
        self.lock = Lock()

    def run(self):
        if self.isRecording:
            return
        self.isRecording = True
        try:
            record_save(AUDIO_PATH)
            result_t = self.model.transcribe(AUDIO_PATH)  # 繁体
            result_s = self.converter.convert(result_t["text"])  # 简体
            command_s = result_s.split('\n')[0]
            print(f'command_s: {command_s}')
            command_en = COMMANDS_MAP_CN[command_s]  # 返回英文指令

            self.isRecording = False
            return command_en
        except Exception as e:
            logging.error(f'Error in run: {e}')  # 添加日志
            self.isRecording = False
            return None


def turn_voice(request):
    try:
        # 使用辅助函数检查无人机是否已连接
        from myapp.drone import is_drone_connected, is_stream_on

        # if not is_drone_connected():
        #     logging.error("无人机未连接或未正确初始化")
        #     voice.valid = False
        #     return JsonResponse({'status': 0, 'message': '无人机未连接，无法开启语音控制'})

        with voice.lock:
            if voice.valid:
                voice.valid = False
                return JsonResponse({'status': 0, 'message': '关闭语音控制'})
            elif not voice.valid:
                voice.valid = True
                return JsonResponse({'status': 1, 'message': '开启语音控制'})
    except Exception as e:
        logging.error(f'Error in turn_voice: {e}')
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def record_voice(request):
    try:
        command = voice.run()
        print(f'command: {command}')
        return JsonResponse({'status': 1, 'command': command})
    except Exception as e:
        logging.error(f'Error in record_voice: {e}')  # 添加日志
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


# voice = VoiceCNN()
voice = VoiceWhisper()

if __name__ == "__main__":
    # voice = VoiceCNN()
    voice = VoiceWhisper()
    voice.run()
