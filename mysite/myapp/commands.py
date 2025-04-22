from django.http import JsonResponse

COMMANDS_MAP = {
    "takeoff": "起飞",
    "land": "降落",
    "stop": "停止",
    "up": "上升",
    "down": "下降",
    "forward": "前进",
    "backward": "后退",
    "left": "向左飞",
    "right": "向右飞",
    "rotate_left": "向左转",
    "rotate_right": "向右转",
    "battery": "查询电量"
}

# 反向映射
COMMANDS_MAP_CN = {v: k for k, v in COMMANDS_MAP.items()}

COMMANDS_MAP_CNN = {
    0: "takeoff",
    1: "land",
    2: "forward",
    3: "backward",
    4: "up"
}
