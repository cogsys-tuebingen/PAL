import tensorflow as tf

from code_ import abstract_net_class as a
import code_.networks.efficient_net.efficientnet_builder_cifar as eb


class SimpleMnistNet(a.AbstractNetClass):

    @classmethod
    def get_name(cls):
        return "SSimpleMnistNet"

    @classmethod
    def _get_loss_y_pred_y_true_batch_assign_op(cls, iterator, inference_mode_var, batch_size):
        """
        get  the EfficientNet model https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet  and https://arxiv.org/abs/1905.11946

        overrides :class:`abstract_net_class.get_model()`

        :param iterator:
        :param inference_mode_var: must hold an element of type :class:`code_.abstract_net_class.inferenceMode`
        :param batch_size:
        :return:  loss , __y_pred, acc_op, acc_update_op, batch_assign_ops
        """
        with tf.variable_scope("EfficientNet", reuse=tf.AUTO_REUSE):
            # Build model
            num_classes = 10
            x = tf.Variable(tf.zeros([batch_size, 28, 28, 1]), dtype=tf.float32, trainable=False)
            y_true = tf.Variable(tf.zeros([batch_size, 1], dtype=tf.uint8), dtype=tf.uint8, trainable=False)

            cx, cy = iterator.get_next()
            x_assign = x.assign(cx).op
            y_true_assign = y_true.assign(cy).op

            batch_assign_ops = (x_assign, y_true_assign)

            conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                                     name="conv1")
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="pool1")
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same",
                                     activation=tf.nn.relu,
                                     name="conv2")
            pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="pool2")
            flatten = tf.reshape(pool1, shape=[-1, 1, 7 * 7 * 64], name="flatten1")
            dense1 = tf.layers.dense(inputs=flatten, units=1024, activation=tf.nn.relu)

            logits = tf.layers.dense(dense1, units=10, name="logits")

            y_pred = logits
            y1h = tf.one_hot(y_true, num_classes, on_value=1, name="oneHot")
            y1h = tf.cast(tf.reshape(y1h, (batch_size, num_classes)), tf.float32)
            y_pred = tf.cast(tf.reshape(y_pred, (batch_size, num_classes)), tf.float32)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y1h,
                                                                             logits=y_pred))

            return loss, y_pred, y_true, batch_assign_ops, None
