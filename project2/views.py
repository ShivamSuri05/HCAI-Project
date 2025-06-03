import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from django.shortcuts import render, redirect
import re
from django.http import FileResponse, Http404
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import io
import base64
from django.http import HttpResponse, JsonResponse


active_learning_progress = {'percent': 0, 'iteration': 0, 'accuracy': 0.0}

def get_active_learning_progress(request):
    return JsonResponse(active_learning_progress)

def clean_html(text):
    return re.sub(r'<.*?>', '', text)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def index(request):
    accuracy = request.session.pop('model_accuracy', None)
    loaded_acc = request.session.pop('loaded_model_accuracy', None)
    context = {
        'model_accuracy': accuracy,
        'loaded_model_accuracy': loaded_acc
    }
    return render(request, 'project2/index.html', context)

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

    # Save accuracy to session
    request.session['model_accuracy'] = acc*100

    return redirect('project2:index')

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
        # Save accuracy to session
        request.session['loaded_model_accuracy'] = acc*100
        msg = f'Pretrained model and vectorizer loaded successfully! Accuracy: {acc:.4%}'
    except Exception as e:
        msg = f"Error loading model: {e}"

    return redirect('project2:index')

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

def run_active_learning(strategy, initial_size = 100, max_iters=10, batch_size=100, stop_cond=0.001):
    df = pd.read_csv(os.path.join(BASE_DIR, 'project2/data/IMDB Dataset.csv'))
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df['review'] = df['review'].apply(clean_html)

    df_pool, df_test = train_test_split(df, test_size=0.2, stratify=df['sentiment'], random_state=42)
    initial_idx = np.random.choice(df_pool.index, size=initial_size, replace=False)
    labeled_df = df_pool.loc[initial_idx].copy()
    unlabeled_df = df_pool.drop(index=initial_idx).copy()

    vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')
    accuracy_scores = []

    patience = 2
    min_delta = stop_cond
    no_improve_count = 0

    for count in range(max_iters):
        # Vectorize
        X_train = vectorizer.fit_transform(labeled_df['review'])
        y_train = labeled_df['sentiment']
        X_test = vectorizer.transform(df_test['review'])
        y_test = df_test['sentiment']

        # Train and evaluate
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Termination condition
        if accuracy_scores and acc - float(accuracy_scores[-1]) < min_delta:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at iteration {count+1}")
                break
        else:
            no_improve_count = 0

        accuracy_scores.append(f"{acc:.4f}")

        if len(unlabeled_df) == 0:
            break

        # Select next samples
        X_pool = vectorizer.transform(unlabeled_df['review'])
        if strategy == 'margin':
            probs = model.predict_proba(X_pool)
            margin = np.abs(probs[:, 0] - probs[:, 1])
            select_idx = np.argsort(margin)[:batch_size]
        elif strategy == 'random':
            select_idx = np.random.choice(len(unlabeled_df), size=batch_size, replace=False)
        else:  # uncertainty
            probs = model.predict_proba(X_pool)
            uncertainty = 1 - np.max(probs, axis=1)
            select_idx = np.argsort(-uncertainty)[:batch_size]

        # Move samples from pool to labeled
        new_samples = unlabeled_df.iloc[select_idx]
        labeled_df = pd.concat([labeled_df, new_samples])
        unlabeled_df.drop(new_samples.index, inplace=True)
        active_learning_progress['percent'] = int((count + 1) / max_iters * 100)
        active_learning_progress['iteration'] = count + 1
        active_learning_progress['accuracy'] = round(acc, 4)

    return accuracy_scores


def plot_accuracy_curve(scores):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(scores) + 1), scores, marker='o')
    plt.title("Accuracy vs. Active Learning Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.grid(True)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()
    return f"data:image/png;base64,{encoded}"

def plot_all_strategies(results):
    import matplotlib.pyplot as plt
    import io, base64

    plt.figure(figsize=(8, 4))
    for strat, scores in results.items():
        flt_scores = [float(score) for score in scores]
        plt.plot(range(1, len(scores) + 1), flt_scores, marker='o', label=strat.capitalize())
    plt.title("Strategy Comparison: Accuracy vs. Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()
    return f"data:image/png;base64,{encoded}"


def active_learning_view(request):
    if request.method == 'POST':
        strategy = request.POST.get('strategy', 'uncertainty')
        initial_size = int(request.POST.get('initial_size', 100))
        max_iters = int(request.POST.get('max_iters', 10))
        batch_size = int(request.POST.get('batch_size', 100))
        stop_cond = float(request.POST.get('stop_cond', 0.001))
        if strategy == 'all':
            results = {}
            for strat in ['uncertainty', 'margin', 'random']:
                active_learning_progress['strategy'] = strat
                scores = run_active_learning(strat, initial_size, max_iters, batch_size, stop_cond)
                results[strat] = scores
                active_learning_progress['percent'] = 0 
                active_learning_progress['iteration'] = 0 
                active_learning_progress['accuracy'] = 0.0 # reset progress stats
            plot_url = plot_all_strategies(results)
            
            for strat in ['uncertainty', 'margin', 'random']:
                while len(results[strat]) < max_iters:
                    results[strat].append("Early Stopping Condition Triggered")

            return render(request, 'project2/index.html', {
                'initial_size': initial_size,
                'strategy': strategy,
                'batch_size': batch_size,
                'comparison_results': results,
                'plot_url': plot_url,
                'max_iters': range(max_iters)
            })
        else:
            active_learning_progress['strategy'] = strategy
            accuracy_scores = run_active_learning(strategy, initial_size, max_iters, batch_size, stop_cond)
            plot_url = plot_accuracy_curve(accuracy_scores)
        
        active_learning_progress['percent'] = 0 
        active_learning_progress['iteration'] = 0 
        active_learning_progress['accuracy'] = 0.0 # reset progress stats
        
        while len(accuracy_scores) < max_iters:
                    accuracy_scores.append("Early Stopping Condition Triggered")
        return render(request, 'project2/index.html', {
            'initial_size': initial_size,
            'batch_size': batch_size,
            'strategy': strategy,
            'accuracy_scores': accuracy_scores,
            'plot_url': plot_url
        })

    return redirect('project2:index')