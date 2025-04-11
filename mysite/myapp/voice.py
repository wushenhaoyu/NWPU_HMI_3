import os
import time
import wave
from django.http import JsonResponse
import librosa
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import pyaudio

orders_dic = {
    0: "takeoff",
    1: "landing",
    2: "forward",
    3: "backward",
    4: "up"
}


class Voice(nn.Module):
    def __init__(self):
        super(Voice, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        # self.dropout = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(64 * 54 * 3, 2048)
        # self.fc2 = nn.Linear(2048, 5)
        # self.softmax = nn.Softmax(dim=1)

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
        self.softmax = nn.Softmax(dim=1)

        self.recording = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_weight()

        self.audio = pyaudio.PyAudio()

        self.vaild = False  #是否有效
        self.frames = []

    def load_weight(self):
        model_weights_path = os.path.join(os.path.dirname(__file__), 'best_model_weights.pth')
        self.to(self.device)
        self.load_state_dict(torch.load(model_weights_path, map_location=self.device, weights_only=True))
        self.eval()

    def forward(self, x):
        x = x.unsqueeze(1)
        # x = self.pool1(self.relu1(self.conv1(x)))
        # x = self.pool2(self.relu2(self.conv2(x)))
        # x = x.view(x.size(0), -1)  # 展平
        # x = self.dropout(x)  # 应用Dropout层
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)

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
        if self.recording:
            return
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=44100,
                                      input=True,
                                      frames_per_buffer=1024)
        duration_seconds = 2
        start_time = time.time()
        self.frames = []
        print("Recording...")
        while time.time() - start_time < duration_seconds:
            data = self.stream.read(1024)
            self.frames.append(data)
        print("Finished recording.")

        self.stream.stop_stream()
        self.stream.close()

        # 保存录音到文件
        with wave.open(os.path.join(os.path.dirname(__file__), 'output.wav'), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(self.frames))

        self.frames = []
        self.audio.terminate()

        # 加载音频并提取特征
        data, sr = librosa.load(os.path.join(os.path.dirname(__file__), 'output.wav'), sr=None)
        data = librosa.effects.preemphasis(data)
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13).T
        mfcc = zero_pad(mfcc)
        mfcc = torch.tensor(mfcc).unsqueeze(0).to(self.device)

        self.recording = False
        prediction = self(mfcc)
        print(torch.argmax(prediction).item())
        index = torch.argmax(prediction).item()
        command = orders_dic[index]
        print(command)

        return command
        # return [torch.argmax(prediction).item()]


def zero_pad(feature, max_length=216):
    # 计算需要填充的长度
    difference = max_length - feature.shape[0]

    # 对单个特征进行零填充
    padded_feature = np.pad(feature, ((0, difference), (0, 0)), "constant")

    return padded_feature


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


voice = Voice()


def turn_voice(request):
    try:
        if voice.vaild:
            voice.vaild = False
            return JsonResponse({'status': 0, 'message': 'turned off'})
        elif not voice.vaild:
            voice.vaild = True
            return JsonResponse({'status': 1, 'message': 'turned on'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def record_voice(request):
    try:
        command = voice.run()
        return JsonResponse({'status': 1, 'message': 'recording started', 'command': command})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


if __name__ == "__main__":
    voice = Voice()
    voice.run()
