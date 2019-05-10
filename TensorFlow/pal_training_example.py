__author__ = "Maximus Mutschler"
__version__ = "1.0"
__email__ = "maximus.mutschler@uni-tuebingen.de"

import os
import tensorflow as tf
import sys
import numpy as np
import TensorFlow.cifar10_loader as cifar10_loader
from TensorFlow.cifar10_loader import InferenceMode
from TensorFlow.pal_optimizer import PalOptimizer
from TensorFlow.resnet_model import Model


class PalTrainingExample:
    """
    provides a simple training example that shows how to use PAL optimizer. Important parts of this code are marked
    with the !!!IMPORTANT!!! flag.
    """

    static_instance_counter = 0

    def __init__(self):

        self.batch_size = 100
        self.steps_per_train_epoch = 450
        self.steps_per_eval_epoch = 50
        self.steps_per_test_epoch = 100

        self.__sess = tf.Session()
        self.__sess.__enter__()

        np.random.seed(1)
        tf.set_random_seed(1)

        workpath = os.path.dirname(os.path.dirname(sys.argv[0])) + '/'
        self.check_and_create_path(workpath)
        self.__writer = tf.summary.FileWriter(workpath + "log/", filename_suffix=".event", flush_secs=10)

        iterator, self.inference_mode_var = cifar10_loader.get_cifar10_iterator(train_data_size=45000,
                                                                                batch_size=self.batch_size)

        # !!!IMPORTANT!!!   batch_assign_ops are needed to safe the batch to a variable
        # so that multiple inferences with the same batch are possible
        self.__loss_op, self.batch_assign_ops, self.__acc_op, self.__acc_update_op = \
            self._get_resnet34_model(iterator, self.inference_mode_var, self.batch_size)

        self.__sess.run(tf.global_variables_initializer())

        global_step = tf.Variable(1.0, trainable=False, name="global_step")

        # !!!IMPORTANT!!!  we got good results by using an exponential decay on measuring_step_size and
        # maximum_step_size
        measuring_step_dec = tf.train.exponential_decay(0.1,
                                                        global_step,
                                                        450.0,
                                                        0.85, staircase=False)
        maximum_step_size_dec = tf.train.exponential_decay(1.0,
                                                           global_step,
                                                           450.0, 0.85, staircase=False)

        # !!!IMPORTANT!!!  NEPAL works outside of the graph, therefore the optimizer does not return a graph based
        # training operation. An inside graph implementation is, as far as we know, not possible with tensorflow 1.12
        self.__pal = PalOptimizer(self.__loss_op, measuring_step_size=measuring_step_dec,
                                  max_step_size=maximum_step_size_dec, global_step=global_step, is_plot=False,
                                  plot_step_interval=200, save_dir=workpath + "lines/")

        self.metric_variables_initializer = [x.initializer for x in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)]

        self.__sess.run(tf.global_variables_initializer())

        print("\n" + "-" * 40)
        print("successfully loaded network")
        print("-" * 40)
        sys.stdout.flush()
        return

    def _load_next_batch(self):
        self.__sess.run(self.batch_assign_ops)

    def train(self, epochs):
        """
        trains the model for the given amount of epochs.
        after each epocj the average train accuracy, train loss, eval_loss and eval_accuracy are determined.
        Eventually, the test loss is calculated.

        :param epochs: epochs to train
        :return: statistic parameters: mean_train_losses, eval_accs,all_train_losses,all_step_sizes_to_min,
        all_angles,all_grad_norms, all_calc_times
        """
        print("\n" + "-" * 40)
        print("Start training: ")
        print("-" * 40)
        self.__sess.run(
            (self.metric_variables_initializer,  # reset all local variables (for accuracy determination)
             self.inference_mode_var.assign(InferenceMode.TRAIN)))  # set the inference mode to train)

        self._load_next_batch()
        initial_loss = self.__sess.run((self.__loss_op))
        print("initial loss: " + str(initial_loss))

        is_first_run = True

        # Train loop
        for epoch in range(epochs):
            print("--" * 40)
            print("starting epoch " + str(epoch))
            print("--" * 40)
            self.__sess.run((self.inference_mode_var.assign(InferenceMode.TRAIN),
                             self.metric_variables_initializer))
            sum_of_losses = 0

            for i in range(self.steps_per_train_epoch):

                if is_first_run is False:
                    # !!!IMPORTANT!!! a new batch has to be loaded by hand before each training step
                    self._load_next_batch()
                else:
                    is_first_run = False

                try:
                    # !!!IMPORTANT!!! no session.run is needed since Nepal does not provide a graph based training op
                    loss, _ = self.__pal.do_train_step(self.__acc_update_op)
                except BaseException as e:
                    print("~" * 40)
                    print("ERROR occurred: " + str(e) + " -> training stopped")
                    print("~" * 40)
                    raise e

                sum_of_losses += loss

            mean_train_loss = sum_of_losses / self.steps_per_train_epoch
            train_acc = self.__sess.run((self.__acc_op))
            current_step = epoch
            self.__log_scalar("Average Train Loss / Interval", mean_train_loss, current_step)
            self.__log_scalar("Average Train Accuracy / Interval", train_acc, current_step)
            print("average train loss: " + str(mean_train_loss))
            print("average train accuracy: " + str(train_acc))
            ms = self.__sess.run(self.__pal.measuring_step_size)
            print("measuring step size: " + str(ms))
            self.evaluate(current_step)
            sys.stdout.flush()

        self.test()
        return

    def evaluate(self, step):
        """
        evaluates the network
        :param step: only needed for logging
        :return: eval_acc, avg_eval_loss
        """

        self.__sess.run((self.inference_mode_var.assign(InferenceMode.EVAL),
                         self.metric_variables_initializer))

        eval_losses = []

        for j in range(self.steps_per_eval_epoch):
            self._load_next_batch()
            eval_loss, _ = self.__sess.run((self.__loss_op, self.__acc_update_op))
            eval_losses.append(eval_loss)

        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        eval_acc = self.__sess.run((self.__acc_op))

        # LogSummaries
        self.__log_scalar("Average Evaluation Accuracy / interval", eval_acc, step)
        self.__log_scalar("Average Evaluation Loss / interval", avg_eval_loss, step)

        # print stats
        print("average evaluation loss: {0:.6f}".format(avg_eval_loss))
        print("evaluation set accuracy: {0:.4f}".format(eval_acc))

        return eval_acc, avg_eval_loss

    def test(self):
        """
        tests the network
        :return: avg_test_acc, avg_test_loss
        """
        print("\n" + "-" * 40)
        print("Start testing: ")
        print("-" * 40)

        self.__sess.run((self.inference_mode_var.assign(InferenceMode.TEST),
                         self.metric_variables_initializer))

        test_losses = []

        for j in range(self.steps_per_test_epoch):
            self._load_next_batch()

            test_loss, _ = self.__sess.run((self.__loss_op, self.__acc_update_op))
            test_losses.append(test_loss)

        avg_test_loss = sum(test_losses) / len(test_losses)

        avg_test_acc = self.__sess.run((self.__acc_op))

        self.__log_scalar("Average Test Accuracy", avg_test_acc, 0)
        self.__log_scalar("Average Test Loss", avg_test_loss, 0)
        print(" average test loss: {0:.6f}".format(avg_test_loss))
        print(" test set accuracy: {0:.4f}".format(avg_test_acc))
        sys.stdout.flush()
        return avg_test_acc, avg_test_loss

    def __log_scalar(self, tag, value, step):
        """
        logs a scalar outside of the graph

        :param tag:  name of the scalar
        :param value: a scalar
        :param step:  training iteration
        source: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        # is not more expensive as creating summary in graph and logging it later
        self.__writer.add_summary(summary, step)

    @staticmethod
    def _get_resnet34_model(iterator, inference_mode_var, batch_size):
        """
        get the ResNet34 model as defined in  https://arxiv.org/pdf/1512.03385.pdf,
        ResNet V2 is used.        overrides :class:`abstract_net_class.get_model()`

        :param iterator:
        :param inference_mode_var: must hold an element of type :class:`simple_code.cifar10_loader.inferenceMode`
        :param batch_size:
        :return:  loss, batch_assign_ops, acc_op, acc_update_op,
        """
        with tf.variable_scope("Res_Net", reuse=tf.AUTO_REUSE):
            # Build model
            x = tf.Variable(tf.zeros([batch_size, 32, 32, 3]), dtype=tf.float32, trainable=False)
            y_true = tf.Variable(tf.zeros([batch_size, 1], dtype=tf.uint8), dtype=tf.uint8, trainable=False)

            cx, cy = iterator.get_next()
            # !!!IMPORTANT!!!  batches must be assigned to variables, so that multiple inferences over the same batch
            # are possible for each weight update step. Before each weight update step the new batch has to be loaded
            # with the batch_assign_ops
            x_assign = x.assign(cx).op
            y_true_assign = y_true.assign(cy).op

            batch_assign_ops = (x_assign, y_true_assign)

            # params from resnet paper https://arxiv.org/pdf/1512.03385.pdf
            res_model = Model(resnet_size=34, bottleneck=False, num_classes=10, num_filters=64, kernel_size=7,
                              conv_stride=2, first_pool_size=3, first_pool_stride=2, block_sizes=[3, 4, 6, 3],
                              block_strides=[1, 2, 2, 2], resnet_version=2)

            # !!!IMPORTANT!!!  Nepal does not support random operators like Dropout. It would support them if it
            # is possible to use the same random numbers for  at least 2 inferences.
            # tf.nn.dropout()

            logits = res_model(x, (tf.equal(inference_mode_var, tf.cast(InferenceMode.TRAIN, tf.uint8))))

            y_pred = logits
            y1h = tf.one_hot(y_true, 10, on_value=1, name="oneHot")

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y1h,
                                                                             logits=y_pred))

            predicted_class = tf.argmax(y_pred, axis=1)
            acc_op, acc_update_op = tf.metrics.accuracy(y_true, predicted_class)
            print("successfully loaded ResNet50 model")
            return loss, batch_assign_ops, acc_op, acc_update_op

    @staticmethod
    def check_and_create_path(path):
        if not os.path.exists(path):
            os.makedirs(path)
            return False
        return True


if __name__ == "__main__":
    example = PalTrainingExample()
    example.train(50000)
