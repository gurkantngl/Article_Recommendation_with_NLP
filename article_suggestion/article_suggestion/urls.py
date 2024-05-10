from django.contrib import admin
from django.urls import path
from article_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('login-db', views.login_db, name='login-db'),
    path('register', views.register, name='register'),
    path('register-db', views.register_db, name='register_db'),
]
