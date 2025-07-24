from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, Http404

def index(request):
    return render(request, 'project4/index.html', {})

def download_pdf(request):
    raise Http404("Model file not found.")

def recommender(request):
    context = {}
    if request.method == 'POST':
        rating = float(request.POST.get('rating'))
        movie_id = request.POST.get('movie_id')
        print(f"User rated movie {movie_id} as {rating}")
        # You could save the rating here if needed
        context['rating_submitted'] = True
        context['user_rating'] = rating

    mode = request.GET.get('start', False)
    
    if mode or request.method == 'POST':
        movie = {
            'backdrop_url': 'https://image.tmdb.org/t/p/w780/ac0kRKTfiJ4GcoUfb0XIO5vgC8q.jpg',
            'poster_url': 'https://image.tmdb.org/t/p/w500/jexoNYnPd6vVrmygwF6QZmWPFdu.jpg',
            'name': 'Inception',
            'genre': 'Action|Crime|Fantasy',
            'summary': 'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO.'
        }
        context = {
            'start_system': True,
            'movie': movie,
            'rating_choices': [0.5 * i for i in range(10, 0, -1)],
            'preview_msg': "If you rate this movie as 5, you may also like:<ul> <li>Inception (2010)</li><li>Nightmare on Elm Street, A (2010)</li><li>Master of Disguise, The (2002)</li></ul>"
        }
        return render(request, 'project4/recommender.html', context)
    
    return render(request, 'project4/recommender.html', context)