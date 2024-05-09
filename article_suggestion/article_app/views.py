from .models import User
from django.shortcuts import render, redirect
from django.contrib.auth.hashers import make_password
from django.http.response import HttpResponse
# Create your views here.

def index(request):
    return render(request, 'login.html')

def main_page(request):
    return HttpResponse("Başarıyla giriş yaptınız.")

def register(request):
    return render(request, 'register.html')

def register_db(request):
    if request.method == 'POST':
        fullname = request.POST.get('fullname')
        email = request.POST.get('email')
        gender = request.POST.get('gender')
        birth_date = request.POST.get('birth_date')
        password = request.POST.get('password')

        hashed_password = make_password(password)

        user = User.objects.create_user(
            email=email,
            full_name=fullname,
            gender=gender,
            birth_date=birth_date,
            password=hashed_password
        )
        
        return redirect('index')

    return render(request, 'register.html')