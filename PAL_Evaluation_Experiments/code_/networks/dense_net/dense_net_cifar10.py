import numpy as np
import tensorflow as tf

from code_ import abstract_net_class as a
from code_.dropout.dropout import dropout_v2


class DenseNet(a.AbstractNetClass):

    @classmethod
    def get_name(cls):
        return "Dense_Net_CIFAR10"

    @classmethod
    def _get_loss_y_pred_y_true_batch_assign_op(cls, iterator, inference_mode_var, batch_size):
        """
        get  the DenseNet model as defined in  https://arxiv.org/pdf/1608.06993.pdf.
        we usee Dense-Net-BC L=100 k=12

        overrides :class:`abstract_net_class.get_model()`

        :param iterator:
        :param inference_mode_var: must hold an element of type :class:`code_.abstract_net_class.inferenceMode`
        :param batch_size:
        :return:  loss , __y_pred, acc_op, acc_update_op, batch_assign_ops
        """
        with tf.variable_scope("DenseNet"):
            # Build model
            num_classes = 10
            x = tf.Variable(tf.zeros([batch_size, 32, 32, 3]), dtype=tf.float32, trainable=False)
            y_true = tf.Variable(tf.zeros([batch_size, 1], dtype=tf.uint8), dtype=tf.uint8, trainable=False)

            cx, cy = iterator.get_next()
            # cx = tf.Print(cx,[cx])
            x_assign = x.assign(cx).op
            y_true_assign = y_true.assign(cy).op

            batch_assign_ops = (x_assign, y_true_assign)

            dropout_keep_prob = 0.8
            # DenseNet model from https://arxiv.org/pdf/1608.06993.pdf

            is_train = tf.equal(inference_mode_var, tf.cast(a.InferenceMode.TRAIN, tf.uint8))
            growth_rate = 12  # 12 # k -> each dense block produces k feature maps
            layers_per_block = 12  # 16  # L100 -> 32 Layers per block (=16*dense_bc_conv) -> 96  + first conv +2 times transition conv +fc
            # L40  non BC -> 12 Layers per block -> 36  + first conv +2 times transition conv +fc
            net = x

            net = DenseNet.dense_conv(net, is_train, 2 * growth_rate)  # 24*3*3*3 params  + 24 bias   -->  240
            switch_reuse_binary_tensor_vars = []
            # dense block 1
            for i, l in enumerate(range(layers_per_block)):
                with tf.variable_scope("denseBlock1" + str(i)):
                    net_p, srb1 = DenseNet.dense_bc_conv(net, is_train, growth_rate, dropout_keep_prob)  # 2 layer each
                    net = tf.concat([net_p, net], 3)
                    switch_reuse_binary_tensor_vars.append(srb1)
            net = DenseNet.transition_Layer_c(net, is_train)  # ->  1*1*96*0.5

            # dense block 2
            for i, l in enumerate(range(layers_per_block)):
                with tf.variable_scope("denseBlock2" + str(i)):
                    net_p, srb2 = DenseNet.dense_bc_conv(net, is_train, growth_rate, dropout_keep_prob)
                    net = tf.concat([net_p, net], 3)
                    switch_reuse_binary_tensor_vars.append(srb2)
            net = DenseNet.transition_Layer_c(net, is_train)

            # dense block 3
            for i, l in enumerate(range(layers_per_block)):
                with tf.variable_scope("denseBlock3" + str(i)):
                    net_p, srb3 = DenseNet.dense_bc_conv(net, is_train, growth_rate, dropout_keep_prob)
                    net = tf.concat([net_p, net], 3)
                    switch_reuse_binary_tensor_vars.append(srb3)

            net = tf.layers.average_pooling2d(net, [8, 8], 1)  #
            net = tf.layers.flatten(net)
            logits = tf.layers.dense(net, num_classes)

            y_pred = logits
            y1h = tf.one_hot(y_true, num_classes, on_value=1, name="oneHot")
            y1h = tf.cast(tf.reshape(y1h, (batch_size, num_classes)), tf.float32)
            y_pred = tf.cast(tf.reshape(y_pred, (batch_size, num_classes)), tf.float32)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y1h,
                                                                             logits=y_pred))

            loss = tf.identity(loss, name="loss")

            return loss, y_pred, y_true, batch_assign_ops, switch_reuse_binary_tensor_vars

    @staticmethod
    def dense_bc_conv(net, is_train, growth_rate, dropout_keep_prob):
        # net = DenseNet.dense_bn(net, is_train, growth_rate)
        net = DenseNet.dense_conv(net, is_train, growth_rate)
        net, switch_reuse_binary_tensor_var = dropout_v2(net, 1 - dropout_keep_prob, is_train)
        return net, switch_reuse_binary_tensor_var

    @staticmethod
    def dense_bn(net, is_train, growth_rate):  # before each dense_conv
        """
        bottle neck layer
        :param net:
        :param is_train:
        :param growth_rate:
        :return:
        """
        net = tf.layers.batch_normalization(net, training=is_train)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net, 4 * growth_rate, [1, 1])  #
        return net

    @staticmethod
    def dense_conv(net, is_train, growth_rate):
        net = tf.layers.batch_normalization(net, training=is_train)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net, growth_rate, [3, 3], padding='same')
        return net

    @staticmethod
    def transition_Layer_c(net, is_train):
        compression_factor = 1
        net = tf.layers.batch_normalization(net, training=is_train)
        net = tf.layers.conv2d(net, np.floor(net.shape[3].value * compression_factor), [1, 1])
        net = tf.layers.average_pooling2d(net, [2, 2], 2)
        return net
