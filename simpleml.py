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

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
    
pp = pprint.PrettyPrinter(indent=4)

warnings.filterwarnings('ignore') 

tmdb.API_KEY = '9d82bc45b4569d6608d9fbc809d4c5ac' 
search = tmdb.Search()


def grab_poster_tmdb(movie):
    poster_folder = 'posters_final/'
    if not poster_folder.split('/')[0] in os.listdir('./'):
       os.mkdir('./' + poster_folder)
    response = search.movie(query=movie)
    id = response['results'][0]['id']
    movie = tmdb.Movies(id)
    posterp = movie.info()['poster_path']
    title = movie.info()['original_title']
    title = '_'.join(title.split(' '))
    title = '_'.join(title.split('/'))
    title = '_'.join(title.split(':'))
    if os.path.isfile(poster_folder + title + '.jpg '):
        return
    url = 'http://image.tmdb.org/t/p/original' + posterp
    f = open(poster_folder + title + '.jpg ', 'wb')
    f.write(urllib.request.urlopen(url).read())
    f.close()


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
    if os.path.isfile('genresids.pckl'):
        fp = open("genresids.pckl", 'rb')
        GenreIDtoName = pickle.load(fp)
        fp.close()
        return GenreIDtoName
    
    genres = tmdb.Genres()
    
    list_of_genres = genres.movie_list()['genres']
    
    GenreIDtoName = {}
    for i in range(len(list_of_genres)):
        genre_id = list_of_genres[i]['id']
        genre_name = list_of_genres[i]['name']
        GenreIDtoName[genre_id] = genre_name
    # Add not found "Foreign genre"    
    GenreIDtoName[10769] = "Foreign"  
    
    fp = open("genresids.pckl", 'wb')
    pickle.dump(GenreIDtoName, fp)
    fp.close()
          
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
        for page in range(1, 60, 1):
            time.sleep(0.5)
        
            url = 'https://api.themoviedb.org/3/discover/movie?api_key=' + tmdb.API_KEY
            url += '&language=en-US&sort_by=popularity.desc&year=' + str(baseyear) 
            url += '&with_genres=' + str(g_id) + '&page=' + str(page)
    
            data = urllib.request.urlopen(url).read()
    
            dataDict = json.loads(data)
            movies.extend(dataDict['results'])
        done_ids.append(str(g_id))
    print("Pulled movies for genres - " + ','.join(done_ids))
    
    fp = open("movies_for_posters.pckl", 'wb')
    pickle.dump(movies, fp)
    fp.close()
    print("Movies saved - " + str(len(movies)))
    return movies

    
def clean():
    if not os.path.isfile('movies_for_posters.pckl'):
        print('Movies fiel data not found !')
        return
    print('Starting cleaning movies list')
    fp = open("movies_for_posters.pckl", 'rb')
    movies = pickle.load(fp)
    fp.close()
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
            print('Error on getting poster for ', title, ' caused by , ', str(e) , ', try again...')
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


def getBinarizedVectorOfGenres(movies):
    genres = []
    for i in range(len(movies)):
        genres.append(movies[i]['genre_ids'])
    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(genres)

     
def getBinarizedVectorOfOverview(movies):
    content = []
    for i in range(len(movies)):
        movie = movies[i]
        id = movie['id']
        overview = movie['overview']
        overview = overview.replace(',', '')
        overview = overview.replace('.', '')
        content.append(overview)
    vectorize = CountVectorizer(max_df=0.95, min_df=0.005)
    return vectorize.fit_transform(content)


def builingModel():      
    if os.path.isfile('X.pckl') and os.path.isfile('Y.pckl') and os.path.isfile('Movies.pckl'):
        print('Model already builded !')
        xdb = open('X.pckl', 'rb')
        ydb = open('Y.pckl', 'rb')
        X = pickle.load(xdb)
        Y = pickle.load(ydb)
        mdb = open('Movies.pckl', 'rb')
        Movies = pickle.load(mdb)
        xdb.close()
        ydb.close()
        return X, Y, Movies
    
    pull()
    cleanedMovieList = clean()
    # clover(cleanedMovieList)
    withOverwiews = getWithOverwiews(cleanedMovieList)
    
    Y = getBinarizedVectorOfGenres(withOverwiews)
    
    X = getBinarizedVectorOfOverview(withOverwiews)
    
    # save model 
    print('Saving model..')
    xdb = open('X.pckl', 'wb')
    ydb = open('Y.pckl', 'wb')
    pickle.dump(X, xdb)
    pickle.dump(Y, ydb)
    mdb = open('Movies.pckl', 'wb')
    pickle.dump(withOverwiews, mdb)
    genredb = open('Genredict.pckl', 'wb')
    pickle.dump(getGenresIds(), genredb)
    xdb.close()
    ydb.close()
    mdb.close()
    genredb.close()
    return X, Y, withOverwiews


# Standard precision recall metrics
def precisionRecall(gt, preds):
    TP = 0
    FP = 0
    FN = 0
    for t in gt:
        if t in preds:
            TP += 1
        else:
            FN += 1
    for p in preds:
        if p not in gt:
            FP += 1
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / float(TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / float(TP + FN)
    return precision, recall

def calculateMetrics(predictions, GenreIDtoName, testMovies, Movies):
    precs = []
    recs = []
    
    for i in range(len(testMovies)):
        if i % 1 == 0:
            pos = testMovies[i]
            test = Movies[pos]
            gtids = test['genre_ids']
            gt = []
            for g in gtids:
                gname = GenreIDtoName[g]
                gt.append(gname)
            a, b = precisionRecall(gt, predictions[i])
            precs.append(a)
            recs.append(b)
            
    return precs, recs

"""
Start
"""
X, Y, Movies = builingModel()

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)

msk = np.random.rand(X_tfidf.shape[0]) < 0.8
X_train_tfidf = X_tfidf[msk]
X_test_tfidf = X_tfidf[~msk]
Y_train = Y[msk]
Y_test = Y[~msk]
positions = range(len(Movies))

testMovies = np.asarray(positions)[~msk]

GenreIDtoName = getGenresIds()

genreList = sorted(list(GenreIDtoName.keys()))



"""
Use Random Forest Model
"""

classif = RandomForestClassifier(n_estimators=100,n_jobs=10)

print('RF Training...')
train = classif.fit(X_train_tfidf, Y_train)

print('RF Testing...')
predstfidf = classif.predict(X_test_tfidf)

print('RF Report : ')
print(classification_report(Y_test, predstfidf))

predictionsrf = []
for i in range(X_test_tfidf.shape[0]):
    predGenres = []
    movieLabelScores = predstfidf[i]
    for j in range(len(movieLabelScores)):
        if movieLabelScores[j] != 0:
            genre = GenreIDtoName[genreList[j]]
            predGenres.append(genre)
    predictionsrf.append(predGenres)

# View result and compare    
for i in range(X_test_tfidf.shape[0]):
    if i % 50 == 0 and i != 0:
        real = []
        for j in range(len(Movies[i]['genre_ids'])):
            real.append(GenreIDtoName[Movies[i]['genre_ids'][j]])
        print('MOVIE: ', Movies[i]['title'], '\tPREDICTION: ', ','.join(predictionsrf[i]), ',\tREAL: ', ','.join(real))

precs, recs = calculateMetrics(predictionsrf, GenreIDtoName, testMovies, Movies)

print('RF Precision-Recall : ')
print('Precision AVG: ', np.mean(np.asarray(precs)), 'Recall AVG:', np.mean(np.asarray(recs)))


"""
Use SVM (Support vector machine) Model
"""
parameters = {'kernel':['linear'], 'C':[0.01, 0.1, 1.0]}
gridCV = GridSearchCV(SVC(class_weight='balanced'), parameters, cv=3, scoring=make_scorer(f1_score, average='micro'), iid=True)
classif = OneVsRestClassifier(gridCV)

print('SVM Training...')
train = classif.fit(X_train_tfidf, Y_train)

print('SVM Testing...')
predstfidf = classif.predict(X_test_tfidf)

print('SVM Report : ')
print(classification_report(Y_test, predstfidf))

predictions = []
for i in range(X_test_tfidf.shape[0]):
    predGenres = []
    movieLabelScores = predstfidf[i]
    for j in range(len(movieLabelScores)):
        if movieLabelScores[j] != 0:
            genre = GenreIDtoName[genreList[j]]
            predGenres.append(genre)
    predictions.append(predGenres)

# View result and compare    
for i in range(X_test_tfidf.shape[0]):
    if i % 50 == 0 and i != 0:
        real = []
        for j in range(len(Movies[i]['genre_ids'])):
            real.append(GenreIDtoName[Movies[i]['genre_ids'][j]])
        print('MOVIE: ', Movies[i]['title'], '\tPREDICTION: ', ','.join(predictions[i]), ',\tREAL: ', ','.join(real))

precs, recs = calculateMetrics(predictions, GenreIDtoName, testMovies, Movies)

print('SVM Precision-Recall : ')
print('Precision AVG: ', np.mean(np.asarray(precs)), 'Recall AVG:', np.mean(np.asarray(recs)))

"""
Use Multinomial Naive Bayes Model
"""
classifnb = OneVsRestClassifier(MultinomialNB())

print('NB Training...')
classifnb.fit(X[msk].toarray(), Y_train)

print('NB Testing...')
predsnb = classifnb.predict(X[~msk].toarray())

print('NB Report : ')
print(classification_report(Y_test, predsnb))

predictionsnb = []
for i in range(X_test_tfidf.shape[0]):
    predGenres = []
    movieLabelScores = predsnb[i]
    for j in range(len(movieLabelScores)):
        if movieLabelScores[j] != 0:
            genre = GenreIDtoName[genreList[j]]
            predGenres.append(genre)
    predictionsnb.append(predGenres)

for i in range(X_test_tfidf.shape[0]):
    if i % 50 == 0 and i != 0:
        real = []
        for j in range(len(Movies[i]['genre_ids'])):
            real.append(GenreIDtoName[Movies[i]['genre_ids'][j]])
        print('MOVIE: ', Movies[i]['title'], '\tPREDICTION: ', ','.join(predictionsnb[i]), ',\tREAL: ', ','.join(real))

precs, recs = calculateMetrics(predictionsnb, GenreIDtoName, testMovies, Movies)

print('NB Precision-Recall : ')
print('Precision AVG: ', np.mean(np.asarray(precs)), 'Recall AVG:', np.mean(np.asarray(recs)))
