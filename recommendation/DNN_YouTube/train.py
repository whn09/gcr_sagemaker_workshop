import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from DNN import DNN
import os
import pickle
from DataInput import *


def train(learning_rate=0.001, batch_size=32, epochs=2, log_dir='logs'):
    with tf.Graph().as_default():
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True
        )
        sess = tf.Session(config=session_config)
        with sess.as_default():
            dnn = DNN(batch_size=batch_size, embedding_size=256, genre_matrix=genreNumpy, genre_size=genre_count, occ_size=occ_count, geo_size=geo_count, movie_count=movie_count)  # TODO original embedding_size=64

            saver = tf.train.Saver()
            file_writer = tf.summary.FileWriter(log_dir, sess.graph)

            config = projector.ProjectorConfig()
            # One can add multiple embeddings.
            embedding = config.embeddings.add()
            embedding.tensor_name = dnn.movie_embedding.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = 'metadata.tsv'  # not compulsory
            # Saves a config file that TensorBoard will read during startup.
            projector.visualize_embeddings(file_writer, config)

            merged = tf.summary.merge_all()

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(dnn.loss)
        sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        for batch_num, _train_data in DataInput(train_data, batch_size):
            #print('batch_num:', batch_num)
            #print('train_data:', train_data)
            #print('len(train_data[0]):', len(train_data[0]))
            feed_dict = {
                dnn.g: _train_data[0],
                dnn.o: _train_data[1],
                dnn.a: _train_data[2],
                dnn.geo: _train_data[3],
                dnn.wh: _train_data[4],
                dnn.time: _train_data[5],
                dnn.y: _train_data[6]
            }

            summary, step, _, loss = sess.run([merged, global_step, train_op, dnn.loss], feed_dict)
            print('i:', i, 'batch_num:', batch_num, 'step:', step, 'loss:', loss)
            file_writer.add_summary(summary, (i+1)*batch_num)
        saver.save(sess, os.path.join(log_dir, 'model.ckpt'), i)


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

    train()
