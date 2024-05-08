from django.shortcuts import render
from django.http.response import HttpResponse
# Create your views here.

def index(request):
    return render(request, 'login.html')

def main_page(request):
    return HttpResponse("Başarıyla giriş yaptınız.")

def register(request):
    return render(request, 'register.html')