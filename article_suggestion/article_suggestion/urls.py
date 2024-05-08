from django.contrib import admin
from django.urls import path
from article_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('main_page', views.main_page, name='main_page'),
    path('register', views.register, name='register'),
]
