import tensorflow as tf

"""
provides additional often used operations for tf
"""


def conv(tensor_in, conv_size, out_size, activation, name, stride=None):
    with tf.variable_scope(name):
        in_size = tensor_in.shape.as_list()[-1]
        W = tf.get_variable(shape=[conv_size, conv_size, in_size, out_size], name='W',
                            initializer=tf.keras.initializers.he_normal())
        b = tf.get_variable(shape=[out_size], initializer=tf.zeros_initializer, name='b')
    with tf.name_scope(name):
        strides = [1, stride, stride, 1] if stride is not None else [1, 1, 1, 1]
        conv = tf.nn.conv2d(tensor_in, W, strides, 'SAME')
        bias = tf.nn.bias_add(conv, b)
        if activation is not None:
            return activation(bias)
        else:
            return bias


def grouped_conv(tensor_in, conv_size, out_size, activation, name, is_training, stride=None, num_groups=1):
    with tf.variable_scope(name):
        conv2d_layers = [conv(tensor_in, conv_size, out_size // num_groups, None, name + '_%i' % j, stride)
                         for j in range(num_groups)]
        tensor = tf.concat(conv2d_layers, axis=-1)
        tensor = batch_norm(tensor, is_training, 'bn')
        if activation is not None:
            tensor = activation(tensor)
        return tensor


def depthwise_conv(tensor_in, conv_size, activation, name, is_training, stride=None):
    stride = stride if stride is not None else 1
    with tf.variable_scope(name):
        tensor = tensor_in
        c = tensor_in.shape.as_list()[-1]
        W = tf.get_variable(shape=[conv_size, conv_size, c, 1], name='W', initializer=tf.keras.initializers.he_normal())
        b = tf.get_variable(shape=[c], name='b', initializer=tf.keras.initializers.he_normal())

        tensor = tf.nn.depthwise_conv2d(tensor, W, strides=[1, stride, stride, 1], padding='SAME')
        tensor = tf.nn.bias_add(tensor, b)
        tensor = batch_norm(tensor, is_training, 'bn')
        if activation:
            tensor = activation(tensor)
        return tensor


def fc(tensor_in, out_size, name):
    with tf.variable_scope(name):
        in_size = tensor_in.shape.as_list()[-1]
        W = tf.get_variable(shape=[in_size, out_size], name='W', initializer=tf.keras.initializers.he_normal())
        b = tf.get_variable(shape=[out_size], initializer=tf.zeros_initializer, name='b')
    with tf.name_scope(name):
        return tensor_in @ W + b


def batch_norm(tensor_in, is_training, name):
    with tf.variable_scope(name):
        # with tf.device('/cpu:0'):
        return tf.layers.batch_normalization(tensor_in, training=is_training)
    # return tf.contrib.layers.batch_norm(tensor_in, is_training=is_training)


def fun(tensor, activation, name):
    with tf.name_scope(name):
        return activation(tensor)


def max_pool(tensor, k, stride=2):
    return tf.nn.max_pool(tensor, [1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME')


def avg_pool(tensor, k, stride=2):
    return tf.nn.avg_pool(tensor, [1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME')


def global_average_pool(tensor_in):
    return tf.reduce_mean(tf.reduce_mean(tensor_in, 2), 1)
