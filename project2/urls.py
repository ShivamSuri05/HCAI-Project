from django.urls import path
from . import views

app_name = "project2"

urlpatterns = [
    path('index', views.index, name='index'),
    path('train/', views.train_model, name='train_model'),
    path('load/', views.load_model, name='load_model'),
    path('download/model/', views.download_model, name='download_model'),
    path('download/vectorizer/', views.download_vectorizer, name='download_vectorizer'),
    path('active_learning/', views.active_learning_view, name='active_learning'),
    path('progress/', views.get_active_learning_progress, name='get_progress'),
]