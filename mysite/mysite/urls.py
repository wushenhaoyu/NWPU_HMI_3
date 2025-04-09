"""
URL configuration for mysite project.

The `urlpatterns` list routes URLs to login. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function login
    1. Add an import:  from my_app import login
    2. Add a URL to urlpatterns:  path('', login.home, name='home')
Class-based login
    1. Add an import:  from other_app.login import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from myapp import login,voice, drone,drone_control

urlpatterns = [
    path("admin/", admin.site.urls),
    path("video", login.video_feed, name="video"),#output
    
    # ---------登录部分---------#
    path("turn_camera", login.turn_camera, name="turn"),
    # path("turn_point", login.turn_point, name="turn_point"),
    path("storage_face", login.storage_face, name="storage_face"),
    path("login_get_frame_info", login.get_frame_info, name="login_get_frame_info"),

    # ----------电脑控制部分---------#
    path("turn_hand", login.turn_hand, name="turn_hand"),
    path("turn_voice",voice.turn_voice, name="turn_voice"),
    path("record_voice", voice.record_voice, name="record_voice"),
    # path("turn_hand_point", login.turn_hand_point, name="turn_hand_point"),

    # ---------无人机部分---------#
    # path("wifi_connect", drone.get_wifi_state, name="wifi_connect"),
    path("wifi_connect", drone.connect_drone, name="wifi_connect"),
    path("drone_video", drone.video_stream, name="drone_video"),
    path("get_current_state", drone.get_current_state, name="get_current_state"),
    # path("key_input", drone_control.key_input, name="key_input"),
    path("drone_control", drone.control, name="drone_control"),
    path("update_speed", drone.update_speed, name="update_speed"),
]
