# project4/model_cache.py

# These will persist across function calls as long as the server is running
V = None
global_mean = None
movie_list = None
index_to_movie_id = None
movie_embeddings = None
available_movie_ids = None
rated_movie_ids = set()
user_ratings = []
initialized = False
