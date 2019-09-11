import pandas as pd
import re
import numpy as np
import pickle
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')


def clean_str(string):
    string = re.sub('[(][0-9]*[)]', "", string)
    string = re.sub('[,]', "", string)
    string = re.sub('[:]', "", string)
    string = re.sub('[(]', "", string)
    string = re.sub('[)]', "", string)
    string = re.sub('[&]', "", string)
    string = re.sub('[?]', "", string)
    string = re.sub('[...]', "", string)
    string = string.strip()
    string = string.split(" ")
    return string


def map_multid(feature_list, feature_map, max_len):
    new_list=[0] * max_len
    i = 0
    for item in feature_list:
        new_list[i] = feature_map[item]
        i += 1
    return new_list


def load_data(movie_file_path, user_file_path, rate_file_path):
    user_feature = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
    movie_feature = ['MovieID', 'Title', 'Genres']
    watch_feature = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    user_table = pd.read_csv(user_file_path, sep='::', names=user_feature)
    movie_table = pd.read_csv(movie_file_path, sep='::', names=movie_feature)
    rating_table = pd.read_csv(rate_file_path, sep='::', names=watch_feature)
    user_table['Age'] = user_table['Age'].map(lambda x: (x - user_table['Age'].min()) / (user_table['Age'].max() - user_table['Age'].min()))
    movie_table['Genres'] = movie_table['Genres'].map(lambda x: x.strip('\n').split('|'))
    movie_table['TitleStr'] = movie_table['Title'].copy()
    movie_table['Title'] = movie_table['Title'].map(lambda x: clean_str(x))
    return user_table, movie_table, rating_table


def build_map(df, col_name):
    if (col_name == 'Genres') or (col_name == 'Title'):
        genre = ['<UNK>']
        for line in df[col_name]:
            genre += line
        key = list(set(genre))
        key[key.index('<UNK>')], key[0] = key[0], key[key.index('<UNK>')]
    else:
        key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    return m, key


def generate_watch_history(df):  # TODO [bug] watch history shouldn't include last video
    watch_history = {}
    for index, row in df.iterrows():
        if row['UserID'] in watch_history:
            watch_history[row['UserID']].append(row['MovieID'])
        else:
            watch_history[row['UserID']] = [row['MovieID']]
    return watch_history


def insert_watch_history(row, watchdict):
    return watchdict[row['UserID']]


def remap_id(df, col_name, feature_map, max_len=None):
    if (col_name == 'Genres') or (col_name == 'Title'):
        # count maximum length
        x = []
        df.apply(lambda row: x.append(len(row[col_name])), axis=1)
        max_len = max(x)
        df[col_name] = df[col_name].map(lambda x: map_multid(x, feature_map, max_len))
    else:
        df[col_name] = df[col_name].map(lambda x: feature_map[x])
    return df


if __name__ == '__main__':

    user, movie, rating = load_data('data/movies.dat', 'data/users.dat', 'data/ratings.dat')

    gender_map, gender_key = build_map(user, 'Gender')
    genre_map, genre_key = build_map(movie, 'Genres')
    occupation_map, occupation_key = build_map(user, 'Occupation')
    geographic_map, geographic_key = build_map(user, 'Zip-code')
    title_map, title_key = build_map(movie, 'Title')

    user = remap_id(user, 'Gender', gender_map)
    user = remap_id(user, 'Occupation', occupation_map)
    user = remap_id(user, 'Zip-code', geographic_map)
    movie = remap_id(movie, 'Genres', genre_map)
    movie = remap_id(movie, 'Title', title_map)

    watch_history_dict = generate_watch_history(rating)

    user['Watch_History'] = user.apply(lambda row: watch_history_dict[row['UserID']], axis=1)

    genre_count = len(genre_key)
    title_count = len(title_key)
    user_count = user.shape[0]
    movie_count = 3953 # TODO error in dataset, 'MovieID' has intervals
    geo_count = len(geographic_key)
    occ_count = len(occupation_key)

    #print('movie:', movie)
    new_movieIDs = [i for i in range(movie_count)]
    new_movie = pd.DataFrame({'MovieID': new_movieIDs})
    new_movie = new_movie.join(movie.set_index('MovieID'), on=['MovieID'], how='left')
    new_movie['TitleStr'].fillna('NULL', inplace=True)
    #print('new_movie:', new_movie)
    new_movie.to_csv('logs/metadata.tsv', columns=['MovieID', 'TitleStr'], sep='\t', index=False, header=True)

    genreNumpy = np.zeros([movie_count, len(genre_key)+1], np.int32)  # TODO original is int64
    for index, row in movie.iterrows():
        for genre in row['Genres']:
            genreNumpy[row['MovieID']][genre] = genre

    with open('data/remap.pkl', 'wb') as f:
        pickle.dump(user, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(movie, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(rating, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(genreNumpy, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((genre_count, title_count, user_count, movie_count, geo_count, occ_count), f, pickle.HIGHEST_PROTOCOL)
