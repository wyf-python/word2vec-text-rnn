# encoding:utf-8
import tensorflow as tf


class TextConfig():
    embedding_size = 128  # dimension of word embedding  128
    vocab_size = 8000  # number of vocabulary
    pre_trianing = None  # use vector_char trained by word2vec

    seq_length = 200  # max length of sentence
    num_classes = 6  # number of labels

    num_layers = 2  # 隐藏层层数 2
    hidden_dim = 128  # 隐藏层神经元

    keep_prob = 0.8  # droppout
    lr = 1e-3  # learning rate
    lr_decay = 0.9  # learning rate decay
    clip = 6.0  # gradient clipping threshold

    num_epochs = 100  # epochs
    batch_size = 64  # batch_size
    print_per_batch = 100  # print result

    train_filename = './data/train.txt'  # train data
    test_filename = './data/test.txt'  # test data
    val_filename = './data/val.txt'  # validation data
    vocab_filename = './data/vocab.txt'  # vocabulary
    vector_word_filename = './data/vector_word.txt'  # vector_word trained by word2vec
    vector_word_npz = './data/vector_word.npz'  # save vector_word to numpy file


class TextRNN(object):

    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        # self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.rnn()

    def rnn(self):
        def basic_rnn_cell(rnn_size):
            # return tf.contrib.rnn.GRUCell(rnn_size)
            return tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)

        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
                                             initializer=tf.constant_initializer(self.config.pre_trianing))
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope('rnn'):
            rnn_cell = tf.contrib.rnn.MultiRNNCell(
                [basic_rnn_cell(self.config.hidden_dim) for _ in range(self.config.num_layers)])
            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=self.keep_prob)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.embedding_inputs, dtype=tf.float32)
            self.outputs = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("dropout"):
            self.final_output = tf.nn.dropout(self.outputs, self.keep_prob)

        with tf.name_scope('output'):
            fc_w = tf.get_variable('fc_w', shape=[self.final_output.shape[1].value, self.config.num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer())
            fc_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='fc_b')
            self.logits = tf.matmul(self.final_output, fc_w) + fc_b
            # self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(self.logits, 1, name='predictions')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
