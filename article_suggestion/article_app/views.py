from .models import User
from django.shortcuts import render, redirect
from django.http.response import HttpResponse
from django.contrib.auth import authenticate, login
from django.urls import reverse
from django.http import HttpResponseRedirect
from pymongo import MongoClient
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import os
import gc
import torch

client = MongoClient('mongodb://localhost:27017/')
db = client['article_vectors']
collection = db['vectors']

def index(request):
    return render(request, 'login.html')

def main_page(request):
    if request.user:
        interests = ["internet", "simulation"]
        model = fasttext.load_model('wiki.en.bin')
        fasttext_vec = collection.find_one({'fasttext_vectors': {'$exists': True}})
        fasttext_vectors = fasttext_vec['fasttext_vectors']
        print("---------------------",interests,"---------------------\n")
        interest_vectors = [model.get_word_vector(word) for word in interests if word in model.words]
        abstracts = []
        recommendations = []
        interest_vectors = np.array(interest_vectors)
        abstract_vectors = np.array(fasttext_vectors)
        folder_path = 'Inspec/docsutf8'

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                    abstract = file.read()
                    abstracts.append(abstract)
        print("---------------",len(abstracts),"------------------")

        for i, abstract_vector in enumerate(abstract_vectors):
            if abstract_vector.any():
                abstract_vector = np.expand_dims(abstract_vector, axis=0)
                similarities = cosine_similarity(interest_vectors, abstract_vector)
                for j, similarity in enumerate(similarities):
                    recommendations.append((i, similarity[0]))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        fasttext_recommendations = recommendations[:5]
        for idx, similarity in fasttext_recommendations:
            print(f"Idx: {idx}")
            print(f"Similarity: {similarity}")
            print("Abstract: ", abstracts[idx])
            print("\n ------------------------------------------------- \n")
        
        # gc.collect()
        # torch.cuda.empty_cache()

        model_name = 'allenai/scibert_scivocab_uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        scibert_vec = collection.find_one({'scibert_vectors': {'$exists': True}})
        scibert_vectors = scibert_vec['scibert_vectors']

        interest_tokens = tokenizer(interests, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            interest_output = model(**interest_tokens)
        interest_vectors = interest_output.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        recommendations = []
        for i, abstract_vector in enumerate(scibert_vectors):
            similarities = cosine_similarity([interest_vectors], [abstract_vector])[0]
            for j, similarity in enumerate(similarities):
                recommendations.append((i, similarity))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = recommendations[:5]
        for idx, similarity in top_recommendations:
            print(f"Similarity: {similarity}")
            print("Abstract: ", abstracts[idx])
            print("\n ------------------------------------------------- \n")

        return HttpResponse(f"Hoş geldiniz {request.user.full_name}")

    

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
        interestList = request.POST.getlist('selected_interests')

        if password != confirm_password:
            return HttpResponse("Şifreler uyuşmuyor.")

        user = User.objects.create_user(
            email=email,
            full_name=fullname,
            username=email,
            gender=gender,
            birth_date=birth_date,
            password=password,
            interests=interestList
        )
        user.save()
        
        return redirect('index')

    return render(request, 'register.html')

def login_db(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = authenticate(request=request, username=email, password=password)

        if user is not None:
            login(request, user)
            return HttpResponseRedirect(reverse('main_page'))
        else:
            return HttpResponse("Email veya şifre yanlış.")
    
    else:
        return render(request, 'login.html')