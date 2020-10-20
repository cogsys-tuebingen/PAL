import abc

import tensorflow as tf
import numpy as np

class AbstractOptimizerClass(metaclass=abc.ABCMeta):
    """
    mandatory super class for all optimizers used in this framework
    """

    def __init__(self):
        """
        sets optimzer's internal state. But, doesn't add the optimzer to the graph.
        call optimizer :class:AbstractOptimizerClass.initialize'`to add the optimizer to the graph

        """

    @abc.abstractmethod
    def do_train_step(self, additional_ops=()):
        """
        do one train step
        :param additional_ops: some additional operators which results should be inferred from the graph
        :return:
        """

        return

    @abc.abstractmethod
    def initialize(self, sess, grad , loss, global_step=None, *args):
        """
        By calling the operator gets placed into the graph.
        This is separated from the class initialization, since the optimizer has to be set after the weight
        initialization to not change the local seeds.

        :param grad:
        :param sess:
        :param loss:
        :param global_step:
        :param args: params for capsuled classes
        :return:
        """
        return

    @abc.abstractmethod
    def __str__(self):
        return


# noinspection PyAttributeOutsideInit
class TfOptimizer(AbstractOptimizerClass):
    """
    capsules TF optimizers to use them with this framework.
    """

    def __init__(self, optimizer, decayf, params):
        super().__init__()
        self.params = params
        self.optimzerClass = optimizer
        self.decayf = decayf



    def initialize(self, sess,grad, loss=None, *args):
        """
        By calling the operator gets placed into the graph.
        This is separated from the class initialization, since the optimizer has to be set after the weight
        initialization to not change the local seeds.

        :param grad:
        :param sess:
        :param loss: not needed
        :param global_step:
        :param args: params for capsuled classes
        :return:
        """
        self.sess = sess
        sess.__enter__()  # don't know why i have to do this, but have to do it since otherwise elemnts
        #  are put in the wrong graph


        self.global_step = tf.Variable(1.0, trainable=False,dtype=tf.float32)
        if "learning_rate" in self.params:
            self.initial_learning_rate = self.params["learning_rate"]
            if self.decayf is not None:
                lr = self.decayf(self.global_step, self.params["learning_rate"])
                self.params["learning_rate"] = lr

        self.optimizer = self.optimzerClass(**self.params)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = self.optimizer.apply_gradients(grad, self.global_step, name="train_optimizer")
        with tf.control_dependencies(update_ops): # important for batch norm
            self.train_op = tf.group(train_op)

    def do_train_step(self, additional_ops=()):
        if "learning_rate" not in self.params:
            lr = np.nan
            _, ad_ops = self.sess.run((self.train_op, additional_ops))
        elif type(self.params["learning_rate"]) is float:
            lr = self.initial_learning_rate
            _, ad_ops = self.sess.run((self.train_op, additional_ops))
        else:
            _, ad_ops, lr = self.sess.run((self.train_op, additional_ops, self.params["learning_rate"]))
        return lr, ad_ops

    def __str__(self):
        nparams = self.params.copy()
        nparams["learning_rate"] = self.initial_learning_rate
        paramstring = ', '.join("{!s}={!r}".format(k, v) for (k, v) in nparams.items())
        return self.optimzerClass.__name__ + "_" + paramstring


class PAOptimizerSuper(AbstractOptimizerClass):
    """
    super class for all PA optimizer variants.
    Approximates the loss in the direction of the gradient with a parabolic function
    """

    norm_cases = 0  # approximation is positive parabola in direction of the gradient
    neg_jump_to_max_or_neg_line = 0  # approximation is negative parabola with negative jump to the maximum
    invalid_losses = 0  # loss at new position is larger than max_loss or nan
    jump_overs = 0  # jump to minimum is larger than max_step_size
    no_min_in_step_direction_cases = 0  # all other cases. In this case max_step_size

    measuring_step_size = None
    is_debug = False


