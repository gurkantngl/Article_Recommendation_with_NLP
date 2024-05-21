from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('index', views.index),
    path('register-db', views.register_db),
    path('login-db', views.login_db),
    path('register', views.register),
    path('main_page', (views.main_page,)),
    path('updating_recommendations', views.updating_recommendations),
    path('updated_recommendations', views.updated_recommendations),
    path('search_view', views.search_view),
]