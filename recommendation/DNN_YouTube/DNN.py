import math
import tensorflow as tf


class DNN:
    def __init__(self, batch_size, embedding_size, genre_matrix, genre_size, occ_size, geo_size, movie_count,
                 l2_reg_lamda=0.0, mode='train'):
        self.g = tf.placeholder(tf.float32, [batch_size])  # TODO original is [int64]
        self.a = tf.placeholder(tf.float32, [batch_size])  # TODO original is [float64]
        self.o = tf.placeholder(tf.int32, [batch_size])  # TODO original is [int64]
        self.geo = tf.placeholder(tf.int32, [batch_size])  # TODO original is [int64]
        self.wh = tf.placeholder(tf.int32, [batch_size, None])  # TODO original is [int64]
        self.time = tf.placeholder(tf.float32, [batch_size])  # TODO original is [int64]
        self.y = tf.placeholder(tf.int32, [batch_size, 1])  # TODO original is [int64]

        self.genre_Query = genre_matrix

        self.genre_emb_w = tf.Variable(tf.random_uniform([genre_size, embedding_size], -1.0, 1.0), name='genre_emb_w')
        self.occupation_emb_w = tf.Variable(tf.random_uniform([occ_size, embedding_size], -1.0, 1.0), name='occupation_emb_w')
        self.geographic_emb_w = tf.Variable(tf.random_uniform([geo_size, embedding_size], -1.0, 1.0), name='geographic_emb_w')
        # embedding process
        with tf.name_scope("embedding"):
            watch_genre = tf.gather(self.genre_Query, self.wh)
            watch_history_genre_emb = tf.nn.embedding_lookup(self.genre_emb_w, watch_genre)
            watch_history_final = tf.reduce_mean(tf.reduce_mean(watch_history_genre_emb, axis=1),axis=1)

            geo_emb = tf.nn.embedding_lookup(self.geographic_emb_w, self.geo)
            occ_emb = tf.nn.embedding_lookup(self.occupation_emb_w, self.o)

        print('geo_emb:', geo_emb)
        print('occ_emb:', occ_emb)
        print('watch_history_final:', watch_history_final)
        self.user_vector = tf.concat([geo_emb, occ_emb, watch_history_final, tf.stack([self.g, self.a, self.time], axis=1)], axis=1, name='user_vector')
        print('user_vector:', self.user_vector)

        # Deep Neural Network
        with tf.name_scope("layer"):
            d_layer_1 = tf.layers.dense(self.user_vector, units=1024, activation=tf.nn.relu, use_bias=True, name='f1',
                                        trainable=True)
            d_layer_2 = tf.layers.dense(d_layer_1, units=512, activation=tf.nn.relu, use_bias=True, name='f2',
                                        trainable=True)
            d_layer_3 = tf.layers.dense(d_layer_2, units=256, activation=tf.nn.relu, use_bias=True, name='f3',
                                        trainable=True)
        self.movie_embedding = tf.Variable(
            tf.truncated_normal([movie_count, embedding_size], stddev=1.0 / math.sqrt(embedding_size)), trainable=True, name='movie_embedding')
        self.biases = tf.Variable(tf.zeros([movie_count]))
        print('movie_embedding:', self.movie_embedding)
        print('biases:', self.biases)

        with tf.name_scope("loss"):
            if mode == 'train':
                self.loss = tf.reduce_mean(
                    tf.nn.sampled_softmax_loss(self.movie_embedding, self.biases, self.y, d_layer_3,
                                               num_sampled=100, num_classes=movie_count, num_true=1,
                                               partition_strategy="div"
                                               ))
                tf.summary.scalar('loss', self.loss)
            elif mode == 'eval':
                self.logits = tf.matmul(d_layer_3, tf.transpose(self.movie_embedding))
                self.logits = tf.nn.bias_add(self.logits, self.biases)
                self.labels_one_hot = tf.one_hot(self.y, movie_count)
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.labels_one_hot,
                    logits=self.logits)
