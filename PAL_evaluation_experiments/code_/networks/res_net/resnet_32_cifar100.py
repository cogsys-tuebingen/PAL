import tensorflow as tf

from code_ import abstract_net_class as a
from code_.networks.res_net.resnet_model import Model


class ResNet(a.AbstractNetClass):

    @classmethod
    def get_name(cls):
        return "ResNet_32_CIFAR100"

    @classmethod
    def _get_loss_y_pred_y_true_batch_assign_op(cls, iterator, inference_mode_var, batch_size):
        """
        get  the ResNet50 model as defined in  https://arxiv.org/pdf/1512.03385.pdf,
        ResNet V2 is used.

        overrides :class:`abstract_net_class.get_model()`

        :param iterator:
        :param inference_mode_var: must hold an element of type :class:`code_.abstract_net_class.inferenceMode`
        :param batch_size:
        :return:  loss , __y_pred, acc_op, acc_update_op, batch_assign_ops
        """
        num_classes = 100
        with tf.variable_scope("Res_Net", reuse=tf.AUTO_REUSE):
            # Build model
            x = tf.Variable(tf.zeros([batch_size, 32, 32, 3]), dtype=tf.float32, trainable=False)
            y_true = tf.Variable(tf.zeros([batch_size, 1], dtype=tf.uint8), dtype=tf.uint8, trainable=False)

            cx, cy = iterator.get_next()
            x_assign = x.assign(cx).op
            y_true_assign = y_true.assign(cy).op

            batch_assign_ops = (x_assign, y_true_assign)

            # params from resnet paper https://arxiv.org/pdf/1512.03385.pdf
            res_model = Model(resnet_size=32, bottleneck=False, num_classes=num_classes, num_filters=16, kernel_size=3,
                              conv_stride=1, first_pool_size=None, first_pool_stride=1, block_sizes=[5, 5, 5],
                              block_strides=[1, 2, 2], resnet_version=2)

            logits = res_model(x, (tf.equal(inference_mode_var, tf.cast(a.InferenceMode.TRAIN, tf.uint8))))

            y_pred = logits
            y1h = tf.one_hot(y_true, num_classes, on_value=1, name="oneHot")
            y1h = tf.cast(tf.reshape(y1h, (batch_size, num_classes)), tf.float32)
            y_pred = tf.cast(tf.reshape(y_pred, (batch_size, num_classes)), tf.float32)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y1h,
                                                                             logits=y_pred))

            return loss, y_pred, y_true, batch_assign_ops, None
