import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicRNNCell, LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn


def conv1d(x, output_dim, k=3, h=2, stddev=0.02, padding='SAME', name='conv1d'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k, x.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv1d(x, w, stride=h, padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv


def fc(x, output_size, bias_start=0.0, with_w=False, name='fc'):
    shape = x.get_shape().as_list()

    with tf.variable_scope(name):
        matrix = tf.get_variable(name="matrix", shape=[shape[1], output_size],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(name="bias", shape=[output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias


def rnn_layer(x, hidden_units, last_state=False, mode='LSTM', name='rnn_layer'):
    with tf.variable_scope(name):
        if mode is 'RNN':
            cell_fw = BasicRNNCell(hidden_units)
            cell_bw = BasicRNNCell(hidden_units)
        elif mode is 'LSTM':
            cell_fw = LSTMCell(hidden_units)
            cell_bw = LSTMCell(hidden_units)

        if last_state:
            _, ((_, output_fw), (_, output_bw)) = bidirectional_dynamic_rnn(cell_fw, cell_bw, x, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
        else:
            (output_fw, output_bw), _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, x, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)

    return output


def dense_RNN_block(x, nb_layers, hidden_units, last_unit=False, name=None):
    with tf.name_scope(name):
        layers_concat = list()
        layers_concat.append(x)

        x = rnn_layer(x, hidden_units=hidden_units, name=name+'_dense_'+str(0))

        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = concatenation(layers_concat)
            if i == nb_layers - 1:
                x = rnn_layer(x, last_state=last_unit, hidden_units=hidden_units, name=name+'_dense_'+str(i+1))
            else:
                x = rnn_layer(x, hidden_units=hidden_units, name=name + '_dense_' + str(i + 1))
            layers_concat.append(x)

        x = concatenation(layers_concat)

    return x


def transition_layer(x, drop_prob, theta=0.5, trained=True, name=None):
    with tf.name_scope(name):
        num_transition_output_filters = int(int(x.shape[2]) * float(theta))
        relu1 = relu(x)
        conv1 = conv1d(relu1, num_transition_output_filters, k=1, h=1, name=name+'_conv')
        drop1 = tf.nn.dropout(conv1, drop_prob)
        out = avg_pool(drop1, k=2, h=2, name='avg_pool')

    return out


def concatenation(layers):
    return tf.concat(layers, axis=-1)


def avg_pool(x, k=2, h=2, name='avg_pool'):
    with tf.name_scope(name):
        return tf.keras.layers.AveragePooling1D(pool_size=k, strides=h, padding="same")(x)


def sigmoid(x, name='sigmoid'):
    output = tf.nn.sigmoid(x, name=name)

    return output


def tanh(x, name='tanh'):
    output = tf.nn.tanh(x, name=name)

    return output


def relu(x, name='relu'):
    output = tf.nn.relu(x, name=name)

    return output
