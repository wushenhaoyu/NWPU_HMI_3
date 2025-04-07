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

from myapp import login,drone

urlpatterns = [
    path("admin/", admin.site.urls),
    path("video", login.video_feed, name="video"),#output
    
    # ---------登录部分---------#
    path("turn_camera", login.turn_camera, name="turn"),
<<<<<<< HEAD
    #path("turn_point", login.turn_point, name="turn_point"),
=======
    # path("turn_point", login.turn_point, name="turn_point"),
>>>>>>> 718096d3ca932848bd846db6ca12d432e666628f
    path("storage_face", login.storage_face, name="storage_face"),
    path("login_get_frame_info", login.get_frame_info, name="login_get_frame_info"),

    # ---------无人机部分---------#
<<<<<<< HEAD
    #path("turn_hand", login.turn_hand, name="turn_hand"),
    #path("turn_hand_point", login.turn_hand_point, name="turn_hand_point"),
=======
    # path("turn_hand", login.turn_hand, name="turn_hand"),
    # path("turn_hand_point", login.turn_hand_point, name="turn_hand_point"),
>>>>>>> 718096d3ca932848bd846db6ca12d432e666628f

    path("drone_video", drone.video_feed, name="drone_video"),
]
