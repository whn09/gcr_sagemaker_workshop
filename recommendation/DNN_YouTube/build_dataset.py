import random
import pickle
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    with open('data/remap.pkl', 'rb') as f:
        user = pickle.load(f)
        movie = pickle.load(f)
        rating = pickle.load(f)
        genreNumpy = pickle.load(f)
        genre_count, title_count, user_count, movie_count, geo_count, occ_count = pickle.load(f)

    # construct timestamp and movie_label data_set
    data_set = []

    for UserID, hist in rating.groupby('UserID'):
        #print('UserID:', UserID)
        #print('hist:', hist)
        nearest_watch_time = hist['Timestamp'].max()
        hist_row = hist.loc[hist[hist['Timestamp'] == nearest_watch_time].index].reset_index(drop=True)
        #print('hist_row:', hist_row)
        hist_row_sub = hist_row.loc[0]
        #print('hist_row_sub:', hist_row_sub)
        #data_set.append((list(hist_row['MovieID']), int(hist_row_sub['Timestamp'])))  # TODO original
        data_set.append((list([hist_row_sub['MovieID']]), int(hist_row_sub['Timestamp'])))  # TODO only use one of last video (same Timestamp)

    # construct user_vector
    train_data = []

    user.apply(lambda row: train_data.append(
        (int(row['Gender']), int(row['Occupation']), float(row['Age']), int(row['Zip-code']), row['Watch_History'])),
               axis=1)

    training_data = list(i+j for i, j in zip(train_data, data_set))

    random.shuffle(training_data)

    train_data, validation_data = train_test_split(training_data, test_size=0.2)

    with open('data/dataset.pkl', 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(validation_data, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(genreNumpy, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((genre_count, title_count, user_count, movie_count, geo_count, occ_count), f, pickle.HIGHEST_PROTOCOL)
