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


def grab_poster_tmdb(movie):
    poster_folder='posters_final/'
    if not poster_folder.split('/')[0] in os.listdir('./'):
       os.mkdir('./'+poster_folder)
    response = search.movie(query=movie)
    id = response['results'][0]['id']
    movie = tmdb.Movies(id)
    posterp = movie.info()['poster_path']
    title = movie.info()['original_title']
    if os.path.isfile(poster_folder + title + '.jpg '):
        return
    url = 'http://image.tmdb.org/t/p/original' + posterp
    title = '_'.join(title.split(' '))
    f = open(poster_folder + title + '.jpg ','wb')
    f.write(urllib.request.urlopen(url).read())
    f.close()

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
        pairs.append([i, i])
    return pairs
    
 
def getGenresIds():
    genres = tmdb.Genres()
    
    list_of_genres = genres.movie_list()['genres']
    
    GenreIDtoName = {}
    for i in range(len(list_of_genres)):
        genre_id = list_of_genres[i]['id']
        genre_name = list_of_genres[i]['name']
        GenreIDtoName[genre_id] = genre_name
    # Add not found "Foreign genre"    
    GenreIDtoName[10769] = "Foreign"        
    return GenreIDtoName


def pull():
    if os.path.isfile('movies_for_posters.pckl'):
        print('Movies already pulled !')
        return
    # Loading populare movies by geners
    movies = []
    baseyear = 2019
    
    print('Starting pulling movies from TMDB, please wait...')
    done_ids = []
    allIds = getGenresIds()
    for g_id in allIds:
        baseyear -= 1
        for page in range(1, 6, 1):
            time.sleep(0.5)
        
            url = 'https://api.themoviedb.org/3/discover/movie?api_key=' + tmdb.API_KEY
            url += '&language=en-US&sort_by=popularity.desc&year=' + str(baseyear) 
            url += '&with_genres=' + str(g_id) + '&page=' + str(page)
    
            data = urllib.request.urlopen(url).read()
    
            dataDict = json.loads(data)
            movies.extend(dataDict['results'])
        done_ids.append(str(g_id))
    print("Pulled movies for genres - " + ','.join(done_ids))
    
    f6 = open("movies_for_posters.pckl", 'wb')
    pickle.dump(movies, f6)
    f6.close()
    print("Movies saved - " + str(len(movies)))
    return movies

    
def clean():
    if not os.path.isfile('movies_for_posters.pckl'):
        print('Movies fiel data not found !')
        return
    print('Starting cleaning movies list')
    f6 = open("movies_for_posters.pckl", 'rb')
    movies = pickle.load(f6)
    f6.close()
    movie_ids = [m['id'] for m in movies]
    print("originally we had ", len(movie_ids), " movies")
    movie_ids = np.unique(movie_ids)
    seen_before = []
    no_duplicate_movies = []
    for i in range(len(movies)):
        movie = movies[i]
        id = movie['id']
        if id in seen_before:
            continue
        else:
            seen_before.append(id)
            no_duplicate_movies.append(movie)
    print("After removing duplicates we have ", len(no_duplicate_movies), " movies")
    return no_duplicate_movies

    
def clover(movies):
    poster_movies = []
    counter = 0
    movies_no_poster = []
    print("Total movies : ", len(movies))
    print("Started downloading posters...")
    for movie in movies:
        if counter % 10 == 0 and counter != 0:
            print(counter)
        id = movie['id']
        title = movie['title']
        if counter % 300 == 0 and counter != 0:
            print("Done with ", counter, " movies!")
            print("Trying to get poster for ", title)
        try:
            grab_poster_tmdb(title)
            time.sleep(1)
            poster_movies.append(movie)
        except Exception as e:
            print('Error on getting poster for ', title, ' caused by , ', str(e) ,', try again...')
            try:
                time.sleep(7)
                grab_poster_tmdb(title)
                poster_movies.append(movie)
            except:
                movies_no_poster.append(movie)
        counter += 1
    print("Done with all the posters!")    
    f = open('poster_movies.pckl', 'w')
    pickle.dump(poster_movies, f)
    f.close()
    f = open('no_poster_movies.pckl', 'w')
    pickle.dump(movies_no_poster, f)
    f.close()


def getWithOverwiews(movies):
    moviesWithOverviews = []
    for i in range(len(movies)):
        movie = movies[i]
        id = movie['id']
        overview = movie['overview']
        if len(overview) == 0:
            continue
        else:
            moviesWithOverviews.append(movie)
    print("After removing movies without overviews we have ", len(moviesWithOverviews), " movies")      
    return moviesWithOverviews        


def getBinarizedVectorOgGenres(movies):
    genres = []
    all_ids = []
    for i in range(len(movies)):
        movie = movies[i]
        id = movie['id']
        genre_ids = movie['genre_ids']
        genres.append(genre_ids)
        all_ids.extend(genre_ids)
    
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(genres)

        
pull()
cleanedMovieList = clean()
#clover(cleanedMovieList)
withOverwiews = getWithOverwiews(cleanedMovieList)
