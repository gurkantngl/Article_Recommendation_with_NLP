from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('index', views.index),
    path('register-db', views.register_db),
    path('main_page', views.main_page),
    path('register', views.register),
]