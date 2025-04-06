"""
URL configuration for mysite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
# from myapp.views import turn_head, video_feed,turn_camera,turn_face,turn_align,turn_point,storage_face,turn_eye,turn_mouth,reset_count,get_count,turn_hand,turn_hand_point
# from myapp.views import video_feed,turn_face,turn_align,turn_point,storage_face,turn_hand,turn_hand_point
from myapp import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("video", views.video_feed, name="video"),#output
    # path("turn_camera", turn_camera, name="turn"),
    path("turn_face", views.turn_face, name="turn_face"),
    path("turn_align", views.turn_align, name="turn_align"),
    path("turn_point", views.turn_point, name="turn_point"),
    path("storage_face", views.storage_face, name="storage_face"),
    path("get_frame_info",views.get_frame_info, name="get_frame_info"),
    # path("turn_eye", turn_eye, name="turn_eye"),
    # path("turn_mouth", turn_mouth, name="turn_mouth"),
    # path("reset_count", reset_count, name="reset_count"),
    # path("turn_head", turn_head, name="turn_head"),
    # path("get_count", get_count, name="get_count"),
    path("turn_hand", views.turn_hand, name="turn_hand"),
    path("turn_hand_point", views.turn_hand_point, name="turn_hand_point"),

]
