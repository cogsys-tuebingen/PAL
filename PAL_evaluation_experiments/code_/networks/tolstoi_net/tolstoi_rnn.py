import tensorflow as tf

from code_ import abstract_net_class as a
from code_.networks.tolstoi_net import tolstoi_char_rnn_model


class TolstoiRNN(a.AbstractNetClass):

    @classmethod
    def get_name(cls):
        return "Tolstoi_RNN"

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
        with tf.variable_scope("Tolstoi_RNN", reuse=tf.AUTO_REUSE):
            # Build model
            x = tf.Variable(tf.zeros([batch_size, 50], dtype=tf.int32), dtype=tf.int32, trainable=False)
            y_true = tf.Variable(tf.zeros([batch_size, 50], dtype=tf.int32), dtype=tf.int32, trainable=False)

            cx, cy = iterator.get_next()
            x_assign = x.assign(cx).op
            y_true_assign = y_true.assign(cy).op

            batch_assign_ops = (x_assign, y_true_assign)

            seq_length = 50

            logits = tolstoi_char_rnn_model.set_up(x=x, batch_size=batch_size, seq_length=seq_length, is_training=(
                tf.equal(inference_mode_var, tf.cast(a.InferenceMode.TRAIN, tf.uint8))))

            y_pred = logits

            loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
                y_pred,
                y_true,
                weights=tf.ones([batch_size, seq_length], dtype=tf.float32),
                average_across_timesteps=True,
                average_across_batch=False))

            y_pred = tf.reshape(y_pred, [batch_size * seq_length, -1])  # to be compatible with the following code
            y_true = tf.reshape(y_true, [batch_size * seq_length, -1])

            return loss, y_pred, y_true, batch_assign_ops, None
