from .models import User
from django.shortcuts import render, redirect
from django.http.response import HttpResponse
from django.contrib.auth import authenticate, login
from django.urls import reverse
from django.http import HttpResponseRedirect, JsonResponse
from pymongo import MongoClient
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import os
import pickle
from django.views.decorators.csrf import csrf_exempt
import json

client = MongoClient('mongodb://localhost:27017/')
db = client['article_vectors']
collection = db['vectors']

def get_scibert_vector(text):
    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()   

def index(request):
    return render(request, 'login.html')


@csrf_exempt
def updated_recommendations(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            articles = data.get('articles', [])
            # İşleme alınacak makaleler burada işlenebilir
            return JsonResponse({'status': 'success', 'message': 'Öneriler güncellendi', 'data': articles})
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Geçersiz JSON verisi'}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Yalnızca POST istekleri kabul edilir'}, status=400)

def main_page(request):
    if request.user:
        contents = []
        interests = ["internet", "simulation"]
        model = fasttext.load_model('wiki.en.bin')
        with open('fasttext_vectors.pkl', 'rb') as f:
            fasttext_vectors = pickle.load(f)
        
        fasttext_interest_vectors = [model.get_word_vector(word) for word in interests if word in model.words]
        abstracts = []
        recommendations = []
        fasttext_interest_vectors = np.array(fasttext_interest_vectors)
        abstract_vectors = np.array(fasttext_vectors)
        folder_path = 'Inspec/docsutf8'

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                    abstract = file.read()
                    abstracts.append(abstract)


        for i, abstract_vector in enumerate(abstract_vectors):
            if abstract_vector.any():
                abstract_vector = np.expand_dims(abstract_vector, axis=0)
                similarities = cosine_similarity(fasttext_interest_vectors, abstract_vector)
                for j, similarity in enumerate(similarities):
                    recommendations.append((i, similarity[0]))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        fasttext_recommendations = recommendations[:5]

        for idx, similarity in fasttext_recommendations:
            recommendation = abstracts[idx]
            content = {}
            upper_case_indices = [i for i, c in enumerate(recommendation) if c.isupper()]
            if len(upper_case_indices) >= 2:
                title_end_index = upper_case_indices[1]
                title = recommendation[:title_end_index]
                recommendation = recommendation[title_end_index:]
                content["title"] = title
                content["abstract"] = recommendation
                contents.append(content)

        
        # gc.collect()
        # torch.cuda.empty_cache()

        with open('scibert_vectors.pkl', 'rb') as f:
            scibert_vectors = pickle.load(f)
        
        scibert_interest_vectors = [get_scibert_vector(word) for word in interests]
        abstracts = []
        recommendations = []
        scibert_interest_vectors = np.vstack(scibert_interest_vectors)
        abstract_vectors = np.array(scibert_vectors)
        folder_path = 'Inspec/docsutf8'

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                    abstract = file.read()
                    abstracts.append(abstract)


        for i, abstract_vector in enumerate(abstract_vectors):
            if abstract_vector.any():
                abstract_vector = np.expand_dims(abstract_vector, axis=0)
                similarities = cosine_similarity(scibert_interest_vectors, abstract_vector)
                for j, similarity in enumerate(similarities):
                    recommendations.append((i, similarity[0]))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        scibert_recommendations = recommendations[:5]

        for idx, similarity in scibert_recommendations:
            recommendation = abstracts[idx]
            content = {}
            upper_case_indices = [i for i, c in enumerate(recommendation) if c.isupper()]
            if len(upper_case_indices) >= 2:
                title_end_index = upper_case_indices[1]
                title = recommendation[:title_end_index]
                recommendation = recommendation[title_end_index:]
                content["title"] = title
                content["abstract"] = recommendation
                contents.append(content)


        return render(request, 'article_page.html', {'articles': contents})

   

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