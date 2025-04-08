import cv2
from django.http import JsonResponse
from djitellopy import Tello
import time


# -------------------------------------------
# 键盘按键映射表
KEY_MAP = {
    "t": ('takeoff', '起飞'),  # 示例：按下 "t" 键起飞
    "l": ('land', '降落'),     # 示例：按下 "l" 键降落

    "h": ('hover', '悬停'),    # 示例：按下 "h" 键悬停

    "w": ('forward', '前进'),  # 示例：按下 "w" 键前进
    "s": ('back', '后退'),     # 示例：按下 "s" 键后退
    "a": ('left', '左移'),     # 示例：按下 "a" 键左移
    "d": ('right', '右移'),     # 示例：按下 "d" 键右移
    "b": ('battery?', '查询电量')
}
# -------------------------------------------

# -------------------------------------------
# 手势映射表
GESTURE_MAP = {
    "t": 'takeoff',  # 示例：按下 "t" 键起飞
    "l": 'land',     # 示例：按下 "l" 键降落

    "h": 'hover',    # 示例：按下 "h" 键悬停

    "w": 'forward',  # 示例：按下 "w" 键前进
    "s": 'back',     # 示例：按下 "s" 键后退
    "a": 'left',     # 示例：按下 "a" 键左移
    "d": 'right',     # 示例：按下 "d" 键右移
    "b": 'battery?'
}
# -------------------------------------------

# -------------------------------------------
# 语音映射表
VOICE_MAP = {
    "t": 'takeoff',  # 示例：按下 "t" 键起飞
    "l": 'land',     # 示例：按下 "l" 键降落

    "h": 'hover',    # 示例：按下 "h" 键悬停

    "w": 'forward',  # 示例：按下 "w" 键前进
    "s": 'back',     # 示例：按下 "s" 键后退
    "a": 'left',     # 示例：按下 "a" 键左移
    "d": 'right',     # 示例：按下 "d" 键右移
    "b": 'battery?'
}
# -------------------------------------------


def key_ctrl(key, drone):
    """
    根据键盘输入控制无人机动作
    :param key: 用户输入的键盘按键
    :param drone: Tello 对象
    """
    match key:
        # case 't':
        #     drone.takeoff()
        # case 'l':
        #     drone.land()
        case 'p':
            if drone.stream_on == True:
                drone.streamoff()
            else:
                drone.streamon()
                while True:

                    frame = drone.get_frame_read().frame
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imshow("Tello Stream", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
        # case 'h':
        #     drone.hover()
        # case 'w':
        #     drone.move_forward(50)
        # case 's':
        #     drone.move_back(50)
        # case 'a':
        #     drone.move_left(50)
        # case 'd':
        #     drone.move_right(50)

        case 'b':
            print(drone.get_battery())
        case _:
            print(f"无效按键{key}")


def key_ctrl_test(key):
    """
    根据键盘输入控制无人机动作
    :param key: 用户输入的键盘按键
    """
    # 输入验证：确保 key 是字符串类型
    if not isinstance(key, str):
        print("无效输入类型")
        return {'status': 0, 'message': "无效输入类型"}

    # 匹配按键并执行对应操作
    if key in KEY_MAP:
        action, action_ch = KEY_MAP[key]
        print(f"action: {action}, action_ch: {action_ch}")
        return {'status': 1, 'message': action}
    else:
        print(f"无效按键 {key}")
        return {'status': 0, 'message': f"无效按键 {key}"}

def gesture_ctrl(gesture, drone):
    """
    根据手势控制无人机动作
    :param gesture: 手势识别结果
    :param drone: Tello 对象
    """
    match gesture:
        case _:
            print(f"无效手势{gesture}")
    pass


# 语音控制（预留接口）
def voice_ctrl(voice, drone):
    """
    根据语音命令控制无人机动作
    :param voice: 语音识别结果
    :param drone: Tello 对象
    """
    match voice:
        case _:
            print(f"无效手势{voice}")
    pass


def key_input(request_key):
    """
    控制面板-键盘控制 对应的按键click后返回request_key
    """
    try:
        action, message = key_ctrl_test(request_key)
        return JsonResponse({'status': 1, 'message': action})
    except Exception as e:
        print(f"Error: {e}")
        return JsonResponse({'status': 0, 'message': f"Error: {e}"})


if __name__ == "__main__":
    while True:
        key = input("请输入控制指令 (t/l/h/w/s/a/d): ")
        action = key_ctrl_test(key)
        print(action)


# # 主函数示例
# if __name__ == "__main__":
#     # # 初始化无人机
#     # drone = Tello()
#     # drone.connect()
#
#     try:
#         while True:
#             # 示例：从用户输入获取按键
#             user_input = input("请输入控制指令 (t/l/h/w/s/a/d): ")
#             time.sleep(1)
#             # key_ctrl(user_input, drone)
#             key_ctrl(user_input)
#
#     except KeyboardInterrupt:
#         print("程序已退出")
    # finally:
    #     drone.land()
    #     # drone.disconnect()