import json

import cv2
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from djitellopy import Tello
import time


# -------------------------------------------
# 指令映射表
CTRL_MAP = {
    "takeoff": '起飞',  # 示例：按下 "t" 键起飞
    "land":  '降落',     # 示例：按下 "l" 键降落
    "up":  '上升',     # 示例：按下 "l" 键降落
    "down":  '下降',     # 示例：按下 "l" 键降落
    "forward": '前进',  # 示例：按下 "w" 键前进
    "back":  '后退',     # 示例：按下 "s" 键后退
    "left":  '左移',     # 示例：按下 "a" 键左移
    "right":  '右移',     # 示例：按下 "d" 键右移
    "rotate_left":  '向左转',     # 示例：按下 "d" 键右移
    "rotate_right":  '向右转',     # 示例：按下 "d" 键右移

    "battery":  '查询电量'
}
# -------------------------------------------




def key_ctrl(key, drone):
    """
    根据键盘输入控制无人机动作
    :param key: 用户输入的键盘按键
    :param drone: Tello 对象
    """
    match key:
        case 'takeoff':
            print(f"Sending command: {key}")
            drone.takeoff()
        case 'land':
            print(f"Sending command: {key}")
            drone.land()
        case 'up':
            if drone.stream_on:
                print(f"Stream is already on")
            else:
                print(f"Sending command: {key}")
                drone.streamon()
                while True:
                    frame = drone.get_frame_read().frame
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imshow("Tello Stream", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        drone.streamoff()
                        break
        case 'down':
            if drone.stream_on:
                print(f"Sending command: {key}")
                drone.streamoff()
            else:
                print(f"Stream is already off")
        # case 'up':
        #     print(f"Sending command: {key}")
        #     drone.move_up(50)
        # case 'down':
        #     print(f"Sending command: {key}")
        #     drone.move_down(50)
        # case 'forward':
        #     print(f"Sending command: {key}")
        #     drone.move_forward(50)
        # case 'back':
        #     print(f"Sending command: {key}")
        #     drone.move_back(50)
        # case 'left':
        #     print(f"Sending command: {key}")
        #     drone.move_left(50)
        # case 'right':
        #     print(f"Sending command: {key}")
        #     drone.move_right(50)
        case 'left':
            battery = drone.get_battery()
            print(f"Battery: {battery}%")
            return {'status': 1, 'message': f"电量: {battery}%"}
        case _:
            print(f"无效按键{key}")
            return {'status': 0, 'message': f"无效按键 {key}"}


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
    if key in CTRL_MAP:
        action_ch = CTRL_MAP[key]
        print(f"action: {key}, action_ch: {action_ch}")
        return {'status': 1, 'message': action_ch}
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


drone = Tello()
drone.connect()

@csrf_exempt
def key_input(request):
    """
    控制面板-键盘控制 对应的按键click后返回request_key
    """
    try:
        data = json.loads(request.body)
        request_key = data.get('request_key')
        print(f"request_key: {request_key}")
        if not request_key:
            return JsonResponse({'status': 0, 'message': "无效按键"})
        
        # response = key_ctrl_test(request_key)
        response = key_ctrl(request_key, drone)

        if response:
            return JsonResponse(response)
        else:
            action_ch = CTRL_MAP.get(request_key, "未知命令")
            return JsonResponse({'status': 1, 'message': action_ch})
    except Exception as e:
        print(f"Error: {e}")
        return JsonResponse({'status': 0, 'message': f"Error: {e}"})


# if __name__ == "__main__":
#     while True:
#         key = input("请输入控制指令 (t/l/h/w/s/a/d): ")
#         action = key_ctrl_test(key)
#         print(action)


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