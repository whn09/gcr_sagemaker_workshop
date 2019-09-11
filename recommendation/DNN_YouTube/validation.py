import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from DNN import DNN
import os
import pickle
import numpy as np
from DataInput import *
from sklearn.metrics import roc_auc_score, confusion_matrix


def validation(learning_rate=0.001, batch_size=32, log_dir='logs'):
    with tf.Graph().as_default():
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True
        )
        sess = tf.Session(config=session_config)
        with sess.as_default():
            dnn = DNN(batch_size=batch_size, embedding_size=256, genre_matrix=genreNumpy, genre_size=genre_count, occ_size=occ_count, geo_size=geo_count, movie_count=movie_count, l2_reg_lamda=0.0, mode='eval')  # TODO original embedding_size=64
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(log_dir))

    auc_avg = 0
    auc_cnt = 0
    for batch_num, _validation_data in DataInput(validation_data, batch_size):
        #print('batch_num:', batch_num)
        #print('train_data:', train_data)
        #print('len(train_data[0]):', len(train_data[0]))
        feed_dict = {
            dnn.g: _validation_data[0],
            dnn.o: _validation_data[1],
            dnn.a: _validation_data[2],
            dnn.geo: _validation_data[3],
            dnn.wh: _validation_data[4],
            dnn.time: _validation_data[5],
            dnn.y: _validation_data[6]
        }

        logits, labels_one_hot, loss = sess.run([dnn.logits, dnn.labels_one_hot, dnn.loss], feed_dict)
        #print('batch_num:', batch_num, 'logits:', logits, 'labels_one_hot:', labels_one_hot, 'loss:', loss)
        print('batch_num:', batch_num, 'loss:', loss)

        predictions = logits.reshape([-1])
        labels = labels_one_hot.reshape([-1])
        auc = roc_auc_score(labels, predictions)
        print('auc:', auc)
        auc_avg += auc
        auc_cnt += 1

        vec_func = np.vectorize(lambda x: x > 0.01 and 1 or 0)
        cm = confusion_matrix(labels, vec_func(predictions))
        print('cm:')
        print(cm)

    auc_avg /= auc_cnt
    print('auc_avg:', auc_avg)


if __name__ == '__main__':

    with open('data/dataset.pkl', 'rb') as f:
        train_data = pickle.load(f)
        validation_data = pickle.load(f)
        genreNumpy = pickle.load(f)
        genre_count, title_count, user_count, movie_count, geo_count, occ_count = pickle.load(f)

    print('train_data:', len(train_data), len(train_data[0]))
    print('validation_data:', len(validation_data), len(validation_data[0]))
    print('genreNumpy:', genreNumpy.shape)
    print('genre_count:', genre_count, 'title_count:', title_count, 'user_count:', user_count, 'movie_count:', movie_count, 'geo_count:', geo_count, 'occ_count:', occ_count)

    validation()
