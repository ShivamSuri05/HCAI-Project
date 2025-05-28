from django.urls import path
from . import views

app_name = "project1"

urlpatterns = [
    path('index/', views.upload_csv, name='upload'), 
    path('generate_plots/', views.generate_plots, name='generate_plots')
]