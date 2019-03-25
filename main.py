import warnings
import torchvision
import urllib
import requests
import json
import imdb
import time
import itertools
import wget
import os
import tmdbsimple as tmdb
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import os.path

import pprint

pp = pprint.PrettyPrinter(indent=4)

tmdb.API_KEY = '9d82bc45b4569d6608d9fbc809d4c5ac' 
search = tmdb.Search()
imbd_object = imdb.IMDb()


def grab_poster_tmdb(movie):
    response = search.movie(query=movie)
    id = response['results'][0]['id']
    movie = tmdb.Movies(id)
    posterp = movie.info()['poster_path']
    title = movie.info()['original_title']
    if os.path.isfile(poster_folder + title + '.jpg '):
        return
    url = 'image.tmdb.org/t/p/original' + posterp
    title = '_'.join(title.split(' '))
    strcmd = 'wget -O ' + poster_folder + title + '.jpg ' + url
    os.system(strcmd)


def get_movie_id_tmdb(movie):
    response = search.movie(query=movie)
    movie_id = response['results'][0]['id']
    return movie_id


def get_movie_info_tmdb(movie):
    response = search.movie(query=movie)
    id = response['results'][0]['id']
    movie = tmdb.Movies(id)
    info = movie.info()
    return info


def get_movie_genres_tmdb(movie):
    response = search.movie(query=movie)
    id = response['results'][0]['id']
    movie = tmdb.Movies(id)
    genres = movie.info()['genres']
    return genres
"""
info = get_movie_info_tmdb('The Matrix')
print(info['genres'])
results = imbd_object.search_movie('The Matrix')
movie = results[0]
imbd_object.update(movie)
print(movie['genres'])

all_movies=tmdb.Movies(
top_movies=all_movies.popular()

top20_movs=top_movies['results']
"""




def saveMovies():
    if os.path.isfile('movie_list.pckl'):
        print('DataBase already saved !')
        return
    all_movies = tmdb.Movies()
    top1000_movies = []
    print('Pulling movie list, Please wait...')
    for i in range(1, 51):
        if i % 15 == 0:
            time.sleep(7)
        movies_on_this_page = all_movies.popular(page=i)['results']
        top1000_movies.extend(movies_on_this_page)
    len(top1000_movies)
    f3 = open('movie_list.pckl', 'wb')
    pickle.dump(top1000_movies, f3)
    f3.close()
    print('Done!')


def loadMovies():
    print('Loading Database...')
    f3 = open('movie_list.pckl', 'rb')
    top1000_movies = pickle.load(f3)
    f3.close()
    print(str(len(top1000_movies)) + ' Loaded movies')
    return top1000_movies;


def buildDataBase():
    saveMovies()
    return loadMovies()

def list2pairs(l):
    # itertools.combinations(l,2) makes all pairs of length 2 from list l.
    pairs = list(itertools.combinations(l, 2))
    # then the one item pairs, as duplicate pairs aren't accounted for by itertools
    for i in l:
        pairs.append([i,i])
    return pairs
    
 
def getGenresIds():
    genres=tmdb.Genres()
    
    list_of_genres=genres.movie_list()['genres']
    
    GenreIDtoName={}
    for i in range(len(list_of_genres)):
        genre_id=list_of_genres[i]['id']
        genre_name=list_of_genres[i]['name']
        GenreIDtoName[genre_id]=genre_name
        
    return GenreIDtoName





    
top100movies = buildDataBase()    

GenreIDtoName = getGenresIds()

allPairs = []
for movie in top100movies:
    allPairs.extend(list2pairs(movie['genre_ids']))
    
nr_ids = np.unique(allPairs)

visGrid = np.zeros((len(nr_ids), len(nr_ids)))

for p in allPairs:
    visGrid[np.argwhere(nr_ids==p[0]), np.argwhere(nr_ids==p[1])]+=1
    if p[1] != p[0]:
        visGrid[np.argwhere(nr_ids==p[1]), np.argwhere(nr_ids==p[0])]+=1
     
annot_lookup = []
for i in range(len(nr_ids)):
    annot_lookup.append(GenreIDtoName[nr_ids[i]])

sns.heatmap(visGrid, xticklabels=annot_lookup, yticklabels=annot_lookup)
plt.show()



from sklearn.cluster import SpectralCoclustering

model = SpectralCoclustering(n_clusters=5)
model.fit(visGrid)

fit_data = visGrid[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

annot_lookup_sorted = []
for i in np.argsort(model.row_labels_):
    annot_lookup_sorted.append(GenreIDtoName[nr_ids[i]])
    
sns.heatmap(fit_data, xticklabels=annot_lookup_sorted, yticklabels=annot_lookup_sorted, annot=False)
plt.title("After biclustering; rearranged to show biclusters")

plt.show()
