from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/status/<str:task_id>/', views.task_status_api, name='task_status_api'),
    path('api/rate/', views.rate_api, name='rate_api'),
    path('api/upload/', views.upload_file_api, name='upload_file_api'),
]
