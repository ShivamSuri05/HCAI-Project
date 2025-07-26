from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, Http404, FileResponse
from project4 import model_cache
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import requests
import re
import random
import pickle
import os
import io
import base64
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

MAX_SKIPS = 5

def index(request):
    return render(request, 'project4/index.html', {})

def download_pdf(request):
    path = os.path.join(BASE_DIR, 'project4/data/Influence_Based_Cold_Start_Recommendation.pdf')
    if os.path.exists(path):
        return FileResponse(open(path, 'rb'), as_attachment=True, filename='Influence_Based_Cold_Start_Recommendation.pdf')
    else:
        raise Http404("Model file not found.")

def get_combined_dataset():
    ratings = pd.read_csv("ml-latest-small/ratings.csv")
    movies = pd.read_csv("ml-latest-small/movies.csv")
    df = pd.merge(ratings, movies, on="movieId")
    return df

def load_and_return_movie_mappings():
    
    ratings_path = os.path.join(BASE_DIR, 'project4/data/ratings.csv')
    movies_path = os.path.join(BASE_DIR, 'project4/data/movies.csv')
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    # Merge for easier access later
    df = pd.merge(ratings, movies, on="movieId")
    user_ids = df['userId'].unique()
    movie_ids = df['movieId'].unique()

    user_id_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_id_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}
    index_to_movie_id = {idx: mid for mid, idx in movie_id_to_index.items()}
    return movies, index_to_movie_id

def create_R_matrix(df):
    user_ids = df['userId'].unique()
    movie_ids = df['movieId'].unique()

    user_id_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_id_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}
    index_to_movie_id = {idx: mid for mid, idx in movie_id_to_index.items()}

    # Create rating matrix
    num_users = len(user_ids)
    num_movies = len(movie_ids)

    R = np.zeros((num_users, num_movies))  # User x Movie matrix

    for _, row in df.iterrows():
        u_idx = user_id_to_index[row['userId']]
        m_idx = movie_id_to_index[row['movieId']]
        R[u_idx, m_idx] = row['rating']

    return R

def matrix_factorization(R, global_mean, K=20, steps=50, alpha=0.005, lambda_reg=0.1):
    num_users, num_items = R.shape
    U = np.random.normal(scale=1./K, size=(num_users, K))
    V = np.random.normal(scale=1./K, size=(num_items, K))

    for step in range(steps):
        total_error = 0
        for i in range(num_users):
            for j in range(num_items):
                if R[i, j] > 0:
                    norm_rating = R[i, j] - global_mean
                    error = norm_rating - np.dot(U[i], V[j])
                    total_error += error ** 2 + lambda_reg * (np.linalg.norm(U[i])**2 + np.linalg.norm(V[j])**2)
                    U[i] += alpha * (error * V[j] - lambda_reg * U[i])
                    V[j] += alpha * (error * U[i] - lambda_reg * V[j])
        print(f"Step {step+1}/{steps}, Loss: {total_error:.4f}")
    
    return U, V

def get_saved_model_data():
    model_path = os.path.join(BASE_DIR, 'project4/models/movie_model_data.pkl')
    with open(model_path, 'rb') as f:
        data_loaded = pickle.load(f)

    V = data_loaded['V']
    global_mean = data_loaded['global_mean']
    return V, global_mean

def compute_user_embedding(user_ratings, movie_embeddings, global_mean, K=30, lambda_reg=0.1):
    V_list, R_list = [], []
    for mid, rating in user_ratings:
        if mid in movie_embeddings:
            V_list.append(movie_embeddings[mid])
            R_list.append(rating - global_mean)  # normalize user rating

    if not V_list:
        return np.zeros(K)

    V_mat = np.array(V_list)
    R_vec = np.array(R_list)

    A = V_mat.T @ V_mat + lambda_reg * np.eye(K)
    b = V_mat.T @ R_vec
    U_i = np.linalg.solve(A, b)
    return U_i

def recommend_movies(U_i, global_mean, movie_embeddings, seen_movie_ids, top_k=5):
    scored = []
    for mid, v in movie_embeddings.items():
        if mid not in seen_movie_ids:
            pred = U_i @ v
            pred += global_mean  # unnormalize the predicted rating
            scored.append((mid, pred))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def transform_score(raw_score, min_rating=0.5, max_rating=5.0):
    # Sigmoid squashes raw_score into (0,1)
    sigmoid = 1 / (1 + np.exp(-raw_score))
    
    # Scale to [min_rating, max_rating]
    scaled_score = min_rating + sigmoid * (max_rating - min_rating)
    
    return scaled_score

def select_next_movie(user_id, rated_movies, movie_embeddings, movie_metadata, U_i, global_mean, switch_threshold=5):
    all_movie_ids = list(movie_metadata['movieId'].values)
    unrated_ids = list(set(all_movie_ids) - rated_movies)
    
    if not unrated_ids:
        return None

    # Filter out movies not in V
    valid_ids = [mid for mid in unrated_ids if mid in movie_embeddings]
    
    if not valid_ids:
        return None

    if len(rated_movies) < switch_threshold:
        # DIVERSITY SAMPLING: Choose the movie most dissimilar to already rated movies
        if not rated_movies:
            return random.choice(valid_ids)

        rated_vecs = np.array([movie_embeddings[mid] for mid in rated_movies if mid in movie_embeddings])
        
        avg_rated_vec = rated_vecs.mean(axis=0).reshape(1, -1)

        # Find the movie whose embedding is furthest from avg_rated_vec
        candidate_vecs = np.array([movie_embeddings[mid] for mid in valid_ids])
        similarities = cosine_similarity(candidate_vecs, avg_rated_vec).flatten()

        # Sort by *lowest* similarity → highest diversity
        selected_idx = np.argmin(similarities)
        return valid_ids[selected_idx]
    
    else:
        # UNCERTAINTY SAMPLING: Pick movie with predicted rating closest to 3.0
        pred_scores = []
        for mid in valid_ids:
            vj = movie_embeddings[mid]
            pred = np.dot(U_i, vj) + global_mean
            pred = transform_score(pred, min_rating=0.5, max_rating=5.0)
            pred_scores.append((mid, abs(pred - 3.0)))  # uncertainty = distance from neutral

        pred_scores.sort(key=lambda x: x[1])  # sort by uncertainty
        return pred_scores[0][0]

def get_movie_info(movies, movie_id):
    row = movies[movies['movieId'] == movie_id]
    if not row.empty:
        title = row['title'].values[0]
        genres = row['genres'].values[0]
        return title, genres
    return "Unknown", "Unknown"

def plot_avg_ratings_bar(user_ratings, movies):
    genre_rating_sum = {}
    genre_rating_count = {}

    for movie_id, rating in user_ratings:
        row = movies[movies['movieId'] == movie_id]
        if row.empty:
            continue
        genres = row['genres'].values[0].split('|')
        for g in genres:
            genre_rating_sum[g] = genre_rating_sum.get(g, 0) + rating
            genre_rating_count[g] = genre_rating_count.get(g, 0) + 1

    if not genre_rating_sum:
        print("No genre data to plot yet.")
        return

    genres = list(genre_rating_sum.keys())
    avg_ratings = [genre_rating_sum[g] / genre_rating_count[g] for g in genres]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(genres, avg_ratings, color='coral')
    plt.xlabel('Genre')
    plt.ylabel('Average Rating')
    plt.title('Average Ratings by Genre')
    plt.ylim(0, 6)

    # Optional: annotate bars with values
    for bar, rating in zip(bars, avg_ratings):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f'{rating:.2f}', ha='center', va='bottom')

    # Save to buffer
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    # Convert to base64 string
    graph = base64.b64encode(image_png).decode('utf-8')
    return graph

def plot_taste_influence_spider(user_ratings, movies):
    genre_rating_sum = {}
    genre_rating_count = {}

    for movie_id, rating in user_ratings:
        row = movies[movies['movieId'] == movie_id]
        if row.empty:
            continue
        genres = row['genres'].values[0].split('|')
        for g in genres:
            genre_rating_sum[g] = genre_rating_sum.get(g, 0) + rating
            genre_rating_count[g] = genre_rating_count.get(g, 0) + 1

    if not genre_rating_sum:
        print("No genre data to plot yet.")
        return

    genres = list(genre_rating_sum.keys())
    num_unique_genres = len(genres)

    normalized_influence = [
        (genre_rating_sum[g] / len(user_ratings))
        for g in genres
    ]

    normalized_influence += [normalized_influence[0]]
    genres += [genres[0]]
    angles = np.linspace(0, 2 * np.pi, len(genres), endpoint=True)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, normalized_influence, color='blue', linewidth=2, label='Taste Influence')
    ax.fill(angles, normalized_influence, color='blue', alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(genres[:-1], fontsize=10)

    max_val = max(normalized_influence)
    ax.set_yticks([])  # No radial tick labels
    ax.set_ylim(0, max_val * 1.1)  # dynamic max with padding

    ax.set_title("🎯 Taste Influence by Genre", size=14, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64

def get_rec_movies_name(movies, recommendations):
    movie_names = []
    for rec_mid, score in recommendations:
        rec_title, genre = get_movie_info(movies, rec_mid)
        movie_names.append(rec_title)
    return movie_names

def get_movie_details_tmdb(movie_title):
    API_KEY = TMDB_API_KEY
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"

    default_poster = '/static/images/default_poster.jpg'
    default_backdrop = '/static/images/default_backdrop.png'

    response = requests.get(url).json()
    if response['results']:
        movie = response['results'][0]
        title = movie['title']
        overview = movie['overview']
        
        poster_path = movie.get('poster_path')
        backdrop_path = movie.get('backdrop_path')
        
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else default_poster
        backdrop_url = f"https://image.tmdb.org/t/p/w780{backdrop_path}" if backdrop_path else default_backdrop
        
        return title, overview, poster_url, backdrop_url
    else:
        return None, None, default_poster, default_backdrop

def recommender(request):
    context = {}
    mode = request.GET.get('start', False)
    if 'skip_count' not in request.session:
        request.session['skip_count'] = 0
    movies = []

    if not model_cache.initialized:
        print("Initializing model and movie embeddings...")
        V, global_mean = get_saved_model_data()
        movie_list, index_to_movie_id = load_and_return_movie_mappings()

        model_cache.V = V
        model_cache.global_mean = global_mean
        model_cache.movie_list = movie_list
        model_cache.index_to_movie_id = index_to_movie_id
        model_cache.movie_embeddings = {
            index_to_movie_id[idx]: V[idx] for idx in range(len(V))
        }
        model_cache.available_movie_ids = list(model_cache.movie_embeddings.keys())
        model_cache.rated_movie_ids = set()
        model_cache.user_ratings = []
        model_cache.initialized = True

    V = model_cache.V
    global_mean = model_cache.global_mean
    movies = model_cache.movie_list
    index_to_movie_id = model_cache.index_to_movie_id
    movie_embeddings = model_cache.movie_embeddings
    available_movie_ids = model_cache.available_movie_ids
    rated_movie_ids = model_cache.rated_movie_ids
    user_ratings = model_cache.user_ratings


    if request.method == 'POST':
        can_skip = False
        if 'skip' in request.POST:
            if request.session['skip_count'] < MAX_SKIPS:
                request.session['skip_count'] += 1
                # Proceed to next movie without rating
                # ... your code to select next movie ...
        
        skip_count = request.session['skip_count']
        can_skip = skip_count < MAX_SKIPS and 'skip' in request.POST
        rating = float(request.POST.get('rating',-1.0))
        movie_id = int(request.POST.get('movie_id'))
        print(f"User rated movie {movie_id} as {rating}")
        if not can_skip and skip_count != MAX_SKIPS:
            user_ratings.append((movie_id, rating))
            can_skip = True
        else:
            if 'submit_rating' in request.POST:
                user_ratings.append((movie_id, rating))
            else:
                print("Rating Skipped for movie id: ",movie_id)
        
        rated_movie_ids.add(movie_id)
        spider_graph = plot_taste_influence_spider(user_ratings, movies)
        graph_base64 = plot_avg_ratings_bar(user_ratings, movies)
        # You could save the rating here if needed
        context['rating_submitted'] = True
        context['user_rating'] = rating
        U_i = compute_user_embedding(user_ratings, movie_embeddings, global_mean)
        # Use smart selection instead of random
        movie_id = select_next_movie(user_id=0,  # dummy user ID if needed
                                     rated_movies=rated_movie_ids,
                                     movie_embeddings=movie_embeddings,
                                     movie_metadata=movies,
                                     U_i=U_i,
                                     global_mean=global_mean)

        rated_movie_ids.add(movie_id)       
        title, genres = get_movie_info(movies, movie_id)
        rating = 5.0

        temp_rating_list = user_ratings.copy()
        temp_rating_list.append((movie_id,5.0))
        U_i = compute_user_embedding(temp_rating_list, movie_embeddings, global_mean)
        recommendations = recommend_movies(U_i, global_mean, movie_embeddings, seen_movie_ids=rated_movie_ids, top_k=3)
        rated_movie_ids.remove(movie_id)

        movie_names = get_rec_movies_name(movies, recommendations)
        msg = f"If you rate this movie as 5, you may also like:<ul> <li>"+movie_names[0]+"</li><li>"+movie_names[1]+"</li><li>"+movie_names[2]+"</li></ul>"
        _, overview, poster_url, backdrop_url = get_movie_details_tmdb(re.sub(r"\s*\([^)]*\)", "", title))

        rec_movies_list = []
        if len(rated_movie_ids) >=10:
            U_i = compute_user_embedding(user_ratings, movie_embeddings, global_mean)
            recommendations = recommend_movies(U_i, global_mean, movie_embeddings, seen_movie_ids=rated_movie_ids, top_k=10)
            for rec_mid, score in recommendations:
                rec_title, genre = get_movie_info(movies, rec_mid)
                _, _, rec_poster_url, _ = get_movie_details_tmdb(re.sub(r"\s*\([^)]*\)", "", rec_title))
                mv = {
                    'id': rec_mid,
                    'poster_url': rec_poster_url,
                    'name': rec_title,
                    'genre': genre
                }
                rec_movies_list.append(mv)
            
        movie = {
            'id': movie_id,
            'backdrop_url': backdrop_url,
            'poster_url': poster_url,
            'name': title,
            'genre': genres,
            'summary': overview
        }
        context = {
            'start_system': True,
            'movie': movie,
            'rating_choices': [0.5 * i for i in range(10, 0, -1)],
            'preview_msg': msg,
            'bar_graph': graph_base64,
            'spider_radar': spider_graph,
            'can_skip': can_skip,
            'remaining_skips': MAX_SKIPS - skip_count,
            'show_rec': len(rated_movie_ids) >= 10,
            'rec_movies_list': rec_movies_list
        }
        return render(request, 'project4/recommender.html', context)                     

    
    if mode or len(user_ratings) == 0:
        # First movie — pick a random one
        candidate_ids = list(set(available_movie_ids) - rated_movie_ids)
        if not candidate_ids:
            print("No more movies left to rate!")
            return render(request, 'project4/recommender.html', context)
        movie_id = random.choice(candidate_ids)
        rated_movie_ids.add(movie_id)
        title, genres = get_movie_info(movies, movie_id)
        rating = 5.0
    
        U_i = compute_user_embedding([(movie_id,5.0)], movie_embeddings, global_mean)
        recommendations = recommend_movies(U_i, global_mean, movie_embeddings, seen_movie_ids=rated_movie_ids, top_k=3)
        rated_movie_ids.remove(movie_id)

        movie_names = get_rec_movies_name(movies, recommendations)
        msg = f"If you rate this movie as 5, you may also like:<ul> <li>"+movie_names[0]+"</li><li>"+movie_names[1]+"</li><li>"+movie_names[2]+"</li></ul>"
        _, overview, poster_url, backdrop_url = get_movie_details_tmdb(re.sub(r"\s*\([^)]*\)", "", title))

        movie = {
            'id': movie_id,
            'backdrop_url': backdrop_url,
            'poster_url': poster_url,
            'name': title,
            'genre': genres,
            'summary': overview
        }
        context = {
            'start_system': True,
            'movie': movie,
            'rating_choices': [0.5 * i for i in range(10, 0, -1)],
            'preview_msg': msg,
            'bar_graph': None,
            'can_skip': True,
            'remaining_skips': 5
        }
        return render(request, 'project4/recommender.html', context)
    
    return render(request, 'project4/recommender.html', context)