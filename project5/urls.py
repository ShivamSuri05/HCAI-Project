from django.urls import path
from . import views

app_name = "project5"

urlpatterns = [
    path('index/', views.index, name='index'),  # Page rendering
    path('trajectory/', views.get_trajectory_pair, name='get_trajectory_pair'),  # JSON API 
    path('get_trajectory/', views.get_trajectory, name='get_trajectory'),
    path('save-feedback/', views.save_feedback, name='save_feedback'),
    path('save_feedback_final/', views.save_feedback_final, name='save_feedback_final'),
    path('get_training_logs/', views.get_training_logs, name='get_training_logs')
]
