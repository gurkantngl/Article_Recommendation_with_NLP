from .models import User
from django.shortcuts import render, redirect
from django.contrib.auth.hashers import make_password
from django.http.response import HttpResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.hashers import check_password

def index(request):
    return render(request, 'login.html')

def main_page(request):
    return HttpResponse("Başarıyla giriş yaptınız.")

def register(request):
    interestList = ["Internet",
                    "Psychology",
                    "Mobile Computing",
                    "Simulation",
                    "Informational Complexity",
                    "Robust Control",
                    "Electronic Commerce",
                    "Computer Science",
                    "Genetic Algorithms",
                    "Informations Systems"]


    return render(request, 'register.html', {'interestList': interestList})

def register_db(request):
    if request.method == 'POST':
        fullname = request.POST.get('fullname')
        email = request.POST.get('email')
        gender = request.POST.get('gender')
        birth_date = request.POST.get('birth_date')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        if password != confirm_password:
            return HttpResponse("Şifreler uyuşmuyor.")


        User.objects.create_user(
            email=email,
            full_name=fullname,
            gender=gender,
            birth_date=birth_date,
            password=password
        )
        
        return redirect('index')

    return render(request, 'register.html')

def login_db(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            user = None
            return HttpResponse("Kullanıcı bulunamadı.")
        
        if user is not None and check_password(password, user.password):
            login(request, user)
            return HttpResponse("Başarıyla giriş yaptınız.")
        else:
            return HttpResponse("Email veya şifre yanlış.")
    
    else:
        return render(request, 'login.html')