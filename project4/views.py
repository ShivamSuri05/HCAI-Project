from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, Http404

def index(request):
    return render(request, 'project4/index.html', {})

def download_pdf(request):
    raise Http404("Model file not found.")
    
def recommender(request):
    return render(request, 'project4/recommender.html', {})
