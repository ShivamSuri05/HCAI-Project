import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from django.shortcuts import render
import re
from django.http import FileResponse, Http404

def clean_html(text):
    return re.sub(r'<.*?>', '', text)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def index(request):
    return render(request, 'project2/index.html')

def train_model(request):
    df = pd.read_csv(os.path.join(BASE_DIR, 'project2/data/IMDB Dataset.csv'))
    df['review'] = df['review'].apply(clean_html)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    

    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, stratify=df['sentiment'], random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    # Save
    model_path = os.path.join(BASE_DIR, 'project2/models/model.pkl')
    vectorizer_path = os.path.join(BASE_DIR, 'project2/models/vectorizer.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    return render(request, 'project2/index.html', {
        'msg': f'Model trained successfully! Accuracy: {acc:.4%}',
        'train_model': True
    })

def load_model(request):
    try:
        model_path = os.path.join(BASE_DIR, 'project2/models/model.pkl')
        vectorizer_path = os.path.join(BASE_DIR, 'project2/models/vectorizer.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        df = pd.read_csv(os.path.join(BASE_DIR, 'project2/data/IMDB Dataset.csv'))
        df['review'] = df['review'].apply(clean_html)
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    

        X_train, X_test, y_train, y_test = train_test_split(
            df['review'], df['sentiment'], test_size=0.2, stratify=df['sentiment'], random_state=42
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        msg = f'Pretrained model and vectorizer loaded successfully! Accuracy: {acc:.4%}'
    except Exception as e:
        msg = f"Error loading model: {e}"

    return render(request, 'project2/index.html', {'msg': msg})

def download_model(request):
    path = os.path.join(BASE_DIR, 'project2/models/model.pkl')
    if os.path.exists(path):
        return FileResponse(open(path, 'rb'), as_attachment=True, filename='model.pkl')
    else:
        raise Http404("Model file not found.")

def download_vectorizer(request):
    path = os.path.join(BASE_DIR, 'project2/models/vectorizer.pkl')
    if os.path.exists(path):
        return FileResponse(open(path, 'rb'), as_attachment=True, filename='vectorizer.pkl')
    else:
        raise Http404("Vectorizer file not found.")