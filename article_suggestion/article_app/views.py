from .models import User
from django.shortcuts import render, redirect
from django.http.response import HttpResponse
from django.contrib.auth import authenticate, login
from django.urls import reverse
from django.http import HttpResponseRedirect, JsonResponse
from pymongo import MongoClient
import fasttext
import numpy as np
import nltk
from nltk.corpus import stopwords as sw
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import os
import pickle
from django.views.decorators.csrf import csrf_exempt
import json
import string
import torch
import ast

client = MongoClient('mongodb://localhost:27017/')
db = client['article_vectors']
collection = db['vectors']
nltk.download('stopwords')
nltk.download('punkt')  

abstracts = []
folder_path = 'Inspec/docsutf8'
for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
            abstract = file.read()
            abstracts.append(abstract)


def index(request):
    return render(request, 'login.html')

def main_page(request):
    if request.user:
        contents = []
        interests = request.user.interests
        interests = interests[0]
        interests = ast.literal_eval(interests)
        
        with open('fasttext_vectors.pkl', 'rb') as f:
            fasttext_vectors = pickle.load(f)

        for interest in interests:
            interest = pre_process(interest)
            interest = ' '.join(interest)
        fasttext_model = fasttext.load_model('wiki.en.bin')
        fasttext_interest_vectors = [fasttext_model.get_word_vector(word) for word in interests if word in fasttext_model.words]
        fasttext_interest_vectors = np.array(fasttext_interest_vectors)
       
        recommendations = []
        abstract_vectors = np.array(fasttext_vectors)


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
                similarity = round(similarity, 2)
                similarity = str(similarity)
                content["similarity"] = similarity
                
                contents.append(content)


        with open('scibert_vectors.pkl', 'rb') as f:
            scibert_vectors = pickle.load(f)
        
        scibert_interest_vectors = [get_scibert_vector(word) for word in interests]
        recommendations = []
        scibert_interest_vectors = np.vstack(scibert_interest_vectors)
        abstract_vectors = np.array(scibert_vectors)

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
                similarity = round(similarity, 2)
                similarity = str(similarity)
                content["similarity"] = similarity
                
                contents.append(content)

        return render(request, 'article_page.html', {'articles': contents})

def updating_recommendations(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            fasttext_recommendations = data.get('fasttext_recommendations', [])
            scibert_recommendations = data.get('scibert_recommendations', [])

            # FastText işlemleri
            new_fasttext_recommendations = []
            fasttext_vectors = []

            with open('fasttext_vectors.pkl', 'rb') as f:
                abstract_vectors = pickle.load(f)
            abstract_vectors = np.array(abstract_vectors)
            fasttext_model = fasttext.load_model('wiki.en.bin')
            interests = request.user.interests
            interests = interests[0]
            interests = ast.literal_eval(interests)
            interest_vectors = [fasttext_model.get_word_vector(word) for word in interests if word in fasttext_model.words]
            interest_vectors = np.array(interest_vectors)

            for text in fasttext_recommendations:
                abstract = pre_process(text)
                abstract_vector = [fasttext_model.get_word_vector(word) for word in abstract if word in fasttext_model.words]
                if abstract_vector:
                    fasttext_vectors.append(sum(abstract_vector) / len(abstract_vector))
                else:
                    fasttext_vectors.append([])

            fasttext_vectors = np.array(fasttext_vectors)

            if fasttext_vectors.shape != interest_vectors.shape:
                min_shape = min(fasttext_vectors.shape[0], interest_vectors.shape[0])
                fasttext_vectors = fasttext_vectors[:min_shape, :]
                interest_vectors = interest_vectors[:min_shape, :]

            new_fasttext_vectors = np.mean(fasttext_vectors + interest_vectors, axis=0)

            for i, abstract_vector in enumerate(abstract_vectors):
                if abstract_vector.any():
                    abstract_vector = np.expand_dims(abstract_vector, axis=0)
                    similarities = cosine_similarity(new_fasttext_vectors.reshape(1, -1), abstract_vector)
                    for _, similarity in enumerate(similarities):
                        new_fasttext_recommendations.append((i, similarity[0]))
            
            new_fasttext_recommendations.sort(key=lambda x: x[1], reverse=True)
            new_fasttext_recommendations = new_fasttext_recommendations[:5]
        
            # SciBERT işlemleri
            with open('scibert_vectors.pkl', 'rb') as f:
                abstract_vectors = pickle.load(f)
            abstract_vectors = np.array(abstract_vectors)

            scibert_interest_vectors = [get_scibert_vector(word) for word in interests]
            new_scibert_recommendations = []
            scibert_vectors = []
            scibert_interest_vectors = np.vstack(scibert_interest_vectors)
            
            model_name = 'allenai/scibert_scivocab_uncased'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            scibert_model = AutoModel.from_pretrained(model_name)
            for text in scibert_recommendations:
                text = pre_process(text)
                abstract = ' '.join(text)
                tokenized_input = tokenizer(abstract, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
                with torch.no_grad():
                    model_output = scibert_model(**tokenized_input)
                embeddings = model_output.last_hidden_state.mean(dim=1).squeeze().numpy()
                scibert_vectors.append(embeddings)
            scibert_vectors = np.array(scibert_vectors)

            if scibert_vectors.shape != scibert_interest_vectors.shape:
                min_shape = min(scibert_vectors.shape[0], scibert_interest_vectors.shape[0])
                scibert_vectors = scibert_vectors[:min_shape, :]
                scibert_interest_vectors = scibert_interest_vectors[:min_shape, :]

            new_scibert_vectors = np.mean(scibert_vectors + scibert_interest_vectors, axis=0)

            for i, abstract_vector in enumerate(abstract_vectors):
                if abstract_vector.any():
                    abstract_vector = np.expand_dims(abstract_vector, axis=0)
                    similarities = cosine_similarity(new_scibert_vectors.reshape(1, -1), abstract_vector)
                    for j, similarity in enumerate(similarities):
                        new_scibert_recommendations.append((i, similarity[0]))

            new_scibert_recommendations.sort(key=lambda x: x[1], reverse=True)
            new_scibert_recommendations = new_scibert_recommendations[:5]

            contents = []
            for idx, similarity in new_fasttext_recommendations:
                recommendation = abstracts[idx]
                content = {}
                upper_case_indices = [i for i, c in enumerate(recommendation) if c.isupper()]
                if len(upper_case_indices) >= 2:
                    title_end_index = upper_case_indices[1]
                    title = recommendation[:title_end_index]
                    recommendation = recommendation[title_end_index:]
                    content["title"] = title
                    content["abstract"] = recommendation
                    similarity = round(similarity, 2)
                    similarity = str(similarity)
                    content["similarity"] = similarity
                    
                    contents.append(content)
            
            for idx, similarity in new_scibert_recommendations:
                recommendation = abstracts[idx]
                content = {}
                upper_case_indices = [i for i, c in enumerate(recommendation) if c.isupper()]
                if len(upper_case_indices) >= 2:
                    title_end_index = upper_case_indices[1]
                    title = recommendation[:title_end_index]
                    recommendation = recommendation[title_end_index:]
                    content["title"] = title
                    content["abstract"] = recommendation
                    similarity = round(similarity, 2)
                    similarity = str(similarity)
                    content["similarity"] = similarity

                    contents.append(content)
                    

            true_positive_fasttext = len(fasttext_recommendations)
            true_positive_scibert = len(scibert_recommendations)

            precision_fasttext = true_positive_fasttext / 5
            precision_scibert = true_positive_scibert / 5

            request.session['contents'] = contents
            request.session['precision_fasttext'] = precision_fasttext
            request.session['precision_scibert'] = precision_scibert

            return HttpResponseRedirect(reverse('updated_recommendations'))
        except Exception as e:
            print(e)
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON data.'}, status=400)
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=405)

def updated_recommendations(request):
    contents = request.session.get('contents', [])
    precision_fasttext = request.session.get('precision_fasttext', 0)
    precision_scibert = request.session.get('precision_scibert', 0)

    return render(request, 'updating.html', {'articles': contents, 'fasttext_precision': precision_fasttext, 'scibert_precision': precision_scibert})

@csrf_exempt
def search_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            search_string = data.get('search_string', '')

            contents = []
            with open('fasttext_vectors.pkl', 'rb') as f:
                fasttext_vectors = pickle.load(f)

            fasttext_model = fasttext.load_model('wiki.en.bin')
            fasttext_search_vectors = [fasttext_model.get_word_vector(search_string) if search_string in fasttext_model.words else []]
            fasttext_search_vectors = np.array(fasttext_search_vectors)

            recommendations = []
            abstract_vectors = np.array(fasttext_vectors)

            for i, abstract_vector in enumerate(abstract_vectors):
                if abstract_vector.any():
                    abstract_vector = np.expand_dims(abstract_vector, axis=0)
                    similarities = cosine_similarity(fasttext_search_vectors, abstract_vector)
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
                    similarity = round(similarity, 2)
                    similarity = str(similarity)
                    content["similarity"] = similarity
                    
                    contents.append(content)
            
            with open('scibert_vectors.pkl', 'rb') as f:
                scibert_vectors = pickle.load(f)

            scibert_search_vectors = [get_scibert_vector(search_string)]
            scibert_search_vectors = np.vstack(scibert_search_vectors)
            recommendations = []
            abstract_vectors = np.array(scibert_vectors)

            for i, abstract_vector in enumerate(abstract_vectors):
                if abstract_vector.any():
                    abstract_vector = np.expand_dims(abstract_vector, axis=0)
                    similarities = cosine_similarity(scibert_search_vectors, abstract_vector)
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
                    similarity = round(similarity, 2)
                    similarity = str(similarity)
                    content["similarity"] = similarity
                    
                    contents.append(content)
            
            return render(request, 'search.html', {'articles': contents})
        
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON data.'}, status=400)
        
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=405)
   

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
    
def pre_process(text):
    stopwords = set(sw.words('english'))
    stemmer = PorterStemmer()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    words = word_tokenize(text)
    cleaned_words = [stemmer.stem(word) for word in words if word not in stopwords and word.isalnum()]
    return cleaned_words

def get_scibert_vector(text):
    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy() 