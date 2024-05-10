from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('index', views.index),
    path('register-db', views.register_db),
    path('login-db', views.login_db),
    path('register', views.register),
]