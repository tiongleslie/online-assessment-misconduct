import tensorflow as tf
from model import tensorflow_utils as tf_utils
from tensorflow.contrib.layers import flatten


class DenseLSTM(object):
    def __init__(self, sess, batch=16, lr_rate=0.00001, epoch=1000, N=23, class_num=None, hidden_units=128):
        self.sess = sess
        self.class_num = class_num
        self.N = N
        self.iter = epoch
        self.bta1 = 0.9
        self.bta2 = 0.999
        self.epsln = 0.00000001
        self.learning_rate = lr_rate
        self.batch_size = batch
        self.hidden_units = hidden_units
        self._build_net()
        print('Initialized DenseLSTM SUCCESS!')

    def _build_net(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 1, self.N], name='x')
        self.drop_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.actual_y = tf.placeholder(tf.int32, shape=[None, self.class_num], name='sg')
        self.label_batch = tf.placeholder(tf.int32, name='label_batch')
        self.trained = tf.placeholder(tf.bool, name='trained')
        self.pred = self.DenseLSTM(self.x)

        self.total_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred, labels=self.actual_y,
                                                       name="softmax"))
        self.train_op_pred = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
        self.output_score = tf.nn.softmax(self.pred, name="softmax_scores")
        self.correct_pred = tf.equal(tf.argmax(self.output_score, 1), tf.argmax(self.actual_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def DenseLSTM(self, x, name='DenseLSTM'):
        layers_block = [4, 8, 4]

        with tf.variable_scope(name):
            conv1 = tf_utils.conv1d(x, 64, k=2, h=2, name=name + '_conv')

            # 1st dense block layer
            with tf.variable_scope('dense_block1'):
                self.denseblock1 = tf_utils.dense_RNN_block(conv1, layers_block[0], 64, name='dense{}'.format(1))
                transition1 = tf_utils.transition_layer(self.denseblock1, self.drop_prob, trained=self.trained,
                                                        name='transition1')

            # 2nd dense block layer
            with tf.variable_scope('dense_block2'):
                self.denseblock2 = tf_utils.dense_RNN_block(transition1, layers_block[1], 128, name='dense{}'.format(2))

        hidden3 = tf_utils.rnn_layer(self.denseblock2, hidden_units=256, last_state=True, mode='LSTM', name='rnn3')
        hidden3_relu = tf_utils.relu(hidden3, name='rnn3_relu')
        hidden3_drop = tf.nn.dropout(hidden3_relu, self.drop_prob)

        out = tf_utils.fc(hidden3_drop, self.class_num, name="output")

        return out

    def train_step(self, x, label_batch):
        train_ops = [self.total_loss, self.train_op_pred, self.accuracy, self.output_score]
        train_feed = {self.x: x, self.actual_y: label_batch, self.drop_prob: 0.5, self.trained: True}

        total_loss, _, acc, score = self.sess.run(train_ops, feed_dict=train_feed)

        return total_loss, acc, score

    def validation_step(self, x, label_batch):
        validation_ops = [self.total_loss, self.accuracy]
        validation_feed = {self.x: x, self.actual_y: label_batch, self.drop_prob: 1.0, self.trained: False}

        total_loss, acc = self.sess.run(validation_ops, feed_dict=validation_feed)

        return total_loss, acc

    def test_step(self, x, label_batch):
        test_ops = [self.accuracy, self.output_score]
        test_feed = {self.x: x, self.actual_y: label_batch, self.drop_prob: 1.0, self.trained: False}

        acc, score = self.sess.run(test_ops, feed_dict=test_feed)

        return acc, score

    @staticmethod
    def print_status(iter_time, loss, acc, step, s_steps):
        print("Epoch: [%2d] [%4d/%4d] \tloss: %.6f\taccuracy: %.6f" % (iter_time, step, s_steps, loss, acc))
