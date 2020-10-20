import tensorflow as tf

from code_ import abstract_net_class as a
import code_.networks.mobile_netv2.mobile_net_v2_model_IM as mobilenet_v2


class MobileNetV2(a.AbstractNetClass):

    @classmethod
    def get_name(cls):
        return "MobileNetV2 for Imagenet with depth_multiplier 1.0 "

    @classmethod
    def _get_loss_y_pred_y_true_batch_assign_op(cls, iterator, inference_mode_var, batch_size):
        """
        get  the MobileNetV2 with depth multiplier 1.0 model as defined in https://arxiv.org/pdf/1801.04381.pdf

        overrides :class:`abstract_net_class.get_model()`

        :param iterator:
        :param inference_mode_var: must hold an element of type :class:`code_.abstract_net_class.inferenceMode`
        :param batch_size:
        :return:  loss , __y_pred, acc_op, acc_update_op, batch_assign_ops
        """
        with tf.variable_scope("MobileNetV2", reuse=tf.AUTO_REUSE):
            # Build model
            num_classes = 1000
            x = tf.Variable(tf.zeros([batch_size, 224, 224, 3]), dtype=tf.float32, trainable=False)
            y_true = tf.Variable(tf.zeros([batch_size, ], dtype=tf.int32), dtype=tf.int32, trainable=False)

            cx, cy = iterator.get_next()
            x_assign = x.assign(cx).op
            y_true_assign = y_true.assign(cy).op

            batch_assign_ops = (x_assign, y_true_assign)

            with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(
                    is_training=(tf.equal(inference_mode_var, tf.cast(a.InferenceMode.TRAIN, tf.uint8))))):
                logits, endpoints, reuse_binary_tensor = mobilenet_v2.mobilenet(input_tensor=x, num_classes=num_classes,
                                                                                depth_multiplier=1.0)

            y_pred = logits
            y1h = tf.one_hot(y_true, num_classes, on_value=1, name="oneHot")
            y1h = tf.cast(tf.reshape(y1h, (batch_size, num_classes)), tf.float32)
            y_pred = tf.cast(tf.reshape(y_pred, (batch_size, num_classes)), tf.float32)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y1h,
                                                                             logits=y_pred))

            return loss, y_pred, y_true, batch_assign_ops, reuse_binary_tensor
