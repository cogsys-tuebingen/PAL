import tensorflow as tf

from code_ import abstract_net_class as a
from code_.networks.res_net.resnet_model import Model


class ResNet(a.AbstractNetClass):

    @classmethod
    def get_name(cls):
        return "ResNet_101_ImageNet"

    @classmethod
    def _get_loss_y_pred_y_true_batch_assign_op(cls, iterator, inference_mode_var, batch_size):
        # Build model
        x = tf.Variable(tf.zeros([batch_size, 224, 224, 3]), dtype=tf.float32, trainable=False)
        y_true = tf.Variable(tf.zeros([batch_size, ], dtype=tf.int32), dtype=tf.int32, trainable=False)

        cx, cy = iterator.get_next()

        x_assign = x.assign(cx).op
        y_true_assign = y_true.assign(cy).op

        batch_assign_op = tf.group(x_assign, y_true_assign)

        # params from resnet paper https://arxiv.org/pdf/1512.03385.pdf
        res_model = Model(resnet_size=100, bottleneck=True, num_classes=1000, num_filters=64, kernel_size=7,
                          conv_stride=2, first_pool_size=3, first_pool_stride=2, block_sizes=[3, 4, 23, 3],
                          block_strides=[1, 2, 2, 2], resnet_version=2)
        logits = res_model(x, (tf.equal(inference_mode_var, tf.cast(a.InferenceMode.TRAIN, tf.uint8))))

        y_pred = logits
        y1h = tf.one_hot(y_true, 1000, on_value=1, name="oneHot")

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y1h,
                                                                         logits=y_pred))
        return loss, y_pred, y_true, batch_assign_op, None
