import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import code_.framework_utils as u
from code_.abstract_net_class import AbstractNetClass
from code_.abstract_net_class import InferenceMode
from code_.dataset_loader import AbstractDatasetLoader
from code_.optimizers.optimizers import AbstractOptimizerClass
from code_.optimizers.optimizers import PAOptimizerSuper
from code_.optimizers.optimizers import TfOptimizer
from code_.optimizers.optimal_line_search import OptimalLineSearch


class NetFrame:
    """
    provides a frame where the network model, data set loader  and optimizer are easily exchangeable.
    Includes methods to  initialize, train, evaluate and test a network.
    """

    static_instance_counter = 0

    def __init__(self, net_class: AbstractNetClass, dataset_loader: AbstractDatasetLoader,
                 optimizer: AbstractOptimizerClass, num_gpus: int = 1, seed=1337,
                 train_data_size: int = 45000, batch_size: int = 100, dataset_path: str = "./Datasets/cifarDataset.npy",
                 work_path: str = "../", experiment_name: str = "model0",
                 is_calc_angle=False):

        self.optimizer = optimizer
        self.dataset_path = dataset_path
        self.model_dir = work_path + "./models/" + experiment_name + "_" + str(NetFrame.static_instance_counter) + "/"
        self.plot_dir = self.model_dir + "plots/" + experiment_name + "/"
        self.default_checkpoints_path = self.model_dir + "checkpoints/convNet.ckp"
        self.default_log_path = self.model_dir + "log/"
        self.experimentName = experiment_name
        self.batch_size = batch_size
        self.is_calc_angle = is_calc_angle

        u.check_and_create_path(self.model_dir)
        u.check_and_create_path(self.default_log_path)
        u.check_and_create_path(self.plot_dir)

        # Delete all existing plots and logs
        if u.check_and_create_path(self.plot_dir):
            for files in os.listdir(self.plot_dir):
                os.remove(os.path.join(self.plot_dir, files))

        if u.check_and_create_path(self.default_log_path):
            for files in os.listdir(self.default_log_path):
                os.remove(os.path.join(self.default_log_path, files))
                self.static_instance_counter += 1

        # Set random seeds
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.__sess = tf.Session(graph=tf.get_default_graph(), config=tf.ConfigProto(allow_soft_placement=True,
                                                                                     gpu_options=tf.GPUOptions(
                                                                                         allow_growth=True),
                                                                                     log_device_placement=False))

        self.__writer = tf.summary.FileWriter(self.default_log_path, filename_suffix=".event", flush_secs=10)

        self.__iterator, self.inference_mode_var, train_size, eval_size, test_size = dataset_loader.get_iterator(
            self.__sess, self.dataset_path, train_data_size, self.batch_size,
            num_gpus)

        # Determine number of inferences needed for one epoch
        self.__num_train_it_per_epoch = train_size // self.batch_size  # floor division
        self.__num_train_it_per_epoch += 1 if train_size % self.batch_size != 0 else 0
        self.__num_eval_it_per_epoch = eval_size // self.batch_size  # floor division
        self.__num_eval_it_per_epoch += 1 if eval_size % self.batch_size != 0 else 0
        self.__num_test_it_per_epoch = test_size // self.batch_size  # floor division
        self.__num_test_it_per_epoch += 1 if test_size % self.batch_size != 0 else 0

        # with tf.device('/cpu:0'):
        print("loading Network: " + net_class.get_name())

        self.__grad_op, self.__loss_op, _, self.__acc_op, self.__acc_update_op, self.batch_assign_ops, self.reuse_binary_tensor = net_class.get_model(
            self.__iterator, self.inference_mode_var, batch_size, num_gpus)

        # get gradient,  calc mean gradient, update gradient
        # build grad vars for angle determination
        if self.is_calc_angle:
            with tf.variable_scope("grad_vars"):
                grad_vars = []
                train_vars = [e[1] for e in self.__grad_op]
                gradient_tensors = [e[0] for e in self.__grad_op]
                for var in train_vars:
                    new_var = tf.Variable(tf.zeros(var.shape), trainable=False, name=var.name[0:-2])
                    grad_vars.append(new_var)
            # ass_old_step ops
            ass_grads = []
            for grad_var, grad in zip(grad_vars, gradient_tensors):
                assign = tf.assign(grad_var, grad)
                ass_grads.append(assign)
            with tf.control_dependencies(ass_grads):
                gradient_tensors = [tf.identity(g) for g in gradient_tensors]
                self.__grad_op = list(zip(gradient_tensors, train_vars))

        self.optimizer.initialize(self.__sess, self.__grad_op, self.__loss_op, None, self.plot_dir,
                                  self.reuse_binary_tensor)  # ,batch_assign_ops=self.batch_assign_ops)

        if self.is_calc_angle:
            if isinstance(self.optimizer, PAOptimizerSuper) or isinstance(self.optimizer, OptimalLineSearch):
                vars = self.optimizer.step_direction_variables
            elif isinstance(self.optimizer, TfOptimizer):
                if isinstance(self.optimizer.optimizer, tf.train.MomentumOptimizer):
                    vars = [self.optimizer.optimizer.get_slot(t_var, "momentum") for t_var in tf.trainable_variables()]
                    self.step_direction_norm_op = u.get_calc_norm_op(vars)
            self.step_direction_angle_op = u.get_calc_angel_op(vars, self.__grad_op)

        self.__sess.run(tf.global_variables_initializer())  # since parameter (weight) variables are added before
        #  optimizer variables all weights get the same g._last_id with different optimizers.
        # -> same weight initialization

        self.metric_variables_initializer = [x.initializer for x in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)]

        # get number of parameters
        sum_ = 0
        for train_var in tf.trainable_variables():
            prod = 1
            for e in train_var.get_shape():
                prod = e * prod
            sum_ += prod
        print("amount parameters: ", sum_)

        # saver has to be inizialized after model is build and variables are defined
        self.__saver = tf.train.Saver()

        # save graph for tensorboard
        # self.__writer.add_graph(self.__sess.graph)
        # self.__writer.flush()
        sys.stdout.flush()
        return

    def load_next_batch(self):
        self.__sess.run(self.batch_assign_ops)

    def train(self, training_steps):
        """
        trains the model for train_time minutes.
        after each minute the average train accuracy, train loss, eval_loss, eval_accuracy is determined.
        Eventually the test loss is calculated

        :param train_time: minutes to train
        :return: statistic parameters: mean_train_losses, eval_accs,all_train_losses,all_step_sizes_to_min,
        all_angles,all_grad_norms, all_calc_times
        """
        print("\n" + "-" * 40)
        print("Start training: ")
        print("-" * 40)
        self.__sess.run(
            (self.metric_variables_initializer,  # to reset all local variables (for accuracy determination)
             self.inference_mode_var.assign(InferenceMode.TRAIN)))  # to set the inference mode to train)

        # Statistic variables
        mean_train_losses = []
        all_train_losses = []
        sum_of_line_min_losses = 0
        all_stepsizes_on_line = []
        all_angles = []
        all_grad_norms = []
        all_calc_times = []
        eval_accs = []
        eval_losses = []
        train_accs = []
        all_first_derivatives = []
        all_second_derivatives = []

        self.load_next_batch()
        initial_loss = self.__sess.run((self.__loss_op))
        mean_train_losses.append(initial_loss)
        self.__log_scalar("Average Train Loss / Interval", initial_loss, 0)
        print("initial loss: " + str(initial_loss))

        # Logic variables
        first = True
        time_spend = 0
        epoch_counter = 0
        eval_interval_counter = 0
        inference_counter = 0
        steps_per_interval_counter = 0
        sum_of_losses = 0

        t0 = time.time()
        # Train loop
        for inference_counter in range(training_steps):
            steps_per_interval_counter += 1
            if first is False:
                self.load_next_batch()
            else:
                first = False

            try:

                loss, final_loss, stepsize_on_line, first_derivative, calc_time = self.__do_ol_training()

            except BaseException as e:
                print("~" * 40)
                print("ERROR occured: " + str(e) + " -> training stopped")
                print("~" * 40)
                raise e
            angle = 0

            # Call it here to get angle an same batch
            if self.is_calc_angle:
                angle = np.rad2deg(self.__sess.run(self.step_direction_angle_op))

            time_spend += calc_time
            all_calc_times.append(calc_time)
            sum_of_losses += loss
            all_train_losses.append(loss)
            sum_of_line_min_losses += final_loss
            all_stepsizes_on_line.append(stepsize_on_line)
            all_angles.append(angle)
            all_first_derivatives.append(first_derivative)
            # all_second_derivatives.append(second_derivative)

            self.__log_scalar("/data/angle per step", angle, inference_counter)
            self.__log_scalar("/data/first_derivative per step", first_derivative, inference_counter)
            self.__log_scalar("/data/step size to min  per step", stepsize_on_line, inference_counter)

            if (inference_counter + 1) % self.__num_train_it_per_epoch == 0:
                epoch_counter += 1

                print("steps per interval: " + str(steps_per_interval_counter))
                print("time spend: " + str(time_spend))

                mean_train_loss = sum_of_losses / steps_per_interval_counter
                mean_finaltrain_loss = sum_of_line_min_losses / steps_per_interval_counter
                steps_per_interval_counter = 0
                sum_of_losses = 0  # to determine mean loss per interval
                mean_train_losses.append(mean_train_loss)

                train_acc = self.__sess.run((self.__acc_op))

                self.__log_scalar("Average Train Loss / Interval", mean_train_loss, epoch_counter)
                self.__log_scalar("Average Final Line Loss / Interval", mean_finaltrain_loss, epoch_counter)
                self.__log_scalar("Average Train Accuracy / Interval", train_acc, epoch_counter)

                print("average train loss: " + str(mean_train_loss))
                print("average train accuracy: " + str(train_acc))
                train_accs.append(train_acc)

                eval_interval_counter += 1
                eval_acc, eval_loss = self.evaluate(eval_interval_counter)
                eval_accs.append(eval_acc)
                eval_losses.append(eval_loss)

                print("--" * 40)
                print("starting interval " + str(epoch_counter))
                print("--" * 40)

                # Set model to Train mode and reset accuracy vars
                self.__sess.run((self.inference_mode_var.assign(InferenceMode.TRAIN),
                                 self.metric_variables_initializer))

                print("total time: " + str(time.time() - t0))
                sys.stdout.flush()

        avg_test_acc, avg_test_loss = self.test()

        return mean_train_losses, eval_accs, all_train_losses, all_stepsizes_on_line, all_angles, all_grad_norms \
            , all_calc_times, train_accs, eval_losses, avg_test_acc, avg_test_loss, all_first_derivatives, all_second_derivatives

    def __do_ol_training(self):
        """
        performs a train step of an optimizer that is a subclass of
        :class:`code_.optimizers.optimizers.PAOptimizerSuper`

        :param step: the current iteration step
        :return: loss, 0, learning_rate, 0, calc_time
        """

        start_t = time.time()
        loss, final_loss, _total_step_distance, line_derivative_current_pos = self.optimizer.do_train_step(
            self.__acc_update_op)
        end_t = time.time()
        t_time = end_t - start_t

        return loss, final_loss, _total_step_distance, line_derivative_current_pos, t_time

    def evaluate_train_data(self):
        """
        evaluates the network
        :param interval: only needed for logging
        :return: eval_acc, avg_eval_loss
        """

        self.__sess.run((self.inference_mode_var.assign(InferenceMode.TRAIN),
                         self.metric_variables_initializer))  # use evaluation_Res_Net iterator

        eval_losses = []

        for j in range(self.__num_eval_it_per_epoch):  # is eval _it on purpose since otherwise to expensive!
            self.load_next_batch()

            eval_loss, _ = self.__sess.run((self.__loss_op, self.__acc_update_op))
            eval_losses.append(eval_loss)

        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        eval_acc = self.__sess.run((self.__acc_op))

        return eval_acc, avg_eval_loss

    def evaluate(self, interval):
        """
        evaluates the network
        :param interval: only needed for logging
        :return: eval_acc, avg_eval_loss
        """

        self.__sess.run((self.inference_mode_var.assign(InferenceMode.EVAL),
                         self.metric_variables_initializer))  # use evaluation_Res_Net iterator

        eval_losses = []

        for j in range(self.__num_eval_it_per_epoch):
            self.load_next_batch()

            eval_loss, _ = self.__sess.run((self.__loss_op, self.__acc_update_op))
            eval_losses.append(eval_loss)

        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        eval_acc = self.__sess.run((self.__acc_op))

        # LogSummaries
        self.__log_scalar("Average Evaluation Accuracy / interval", eval_acc, interval)
        self.__log_scalar("Average Evaluation Loss / interval", avg_eval_loss, interval)

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

        for j in range(self.__num_test_it_per_epoch):
            self.load_next_batch()

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

    def save_model(self, path="default"):
        """
        saves the model without the meta_graph
        """
        if path == "default":
            path = self.default_checkpoints_path
        dir_ = os.path.dirname(path)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        praefix = self.__saver.save(self.__sess, path, write_meta_graph=False)  # meta graph -> variable occupation,
        # operationens ...

        # checkpoint -> weights, biases, gradients
        print("Saved model to path: " + str(path))
        return praefix

    def restore_model(self, path="default"):
        """
        restores a model
        """
        if (path == "default"):
            path = self.default_checkpoints_path
        praefix = self.__saver.restore(self.__sess, path)
        print("Restored model from path: " + str(path))
        return praefix

    def __del__(self):
        """
        deconstructor
        """
        self.__sess.close()
        print("network closed")

    def __save_optimizer_evaluation_data(self):
        pass

    def __plot_and_save_data(self, mean_train_losses_per_min, evaluation_accuracies_per_min, train_losses,
                             stepsizes_on_line, angles_per_step, grad_norms, calc_times_per_step, train_acc):
        """
         saves plots and data of the the training process
        """
        name = self.experimentName
        plt.figure()

        # TODO put plotting in netframe
        plt.bar(list(range(len(train_losses))), train_losses)
        #   plt.show(block=False)
        u.save_fig("Train_Losses_" + name, train_losses, self.plot_dir)

        plt.figure()
        plt.bar(list(range(len(stepsizes_on_line))), stepsizes_on_line)
        #   plt.show(block=False)
        u.save_fig("Step_Sizes_to_min_" + name, stepsizes_on_line, self.plot_dir)

        plt.figure()
        plt.bar(list(range(len(angles_per_step))), angles_per_step)
        #   plt.show(block=False)
        u.save_fig("Angles_per_Step_" + name, angles_per_step, self.plot_dir)

        plt.figure()
        plt.bar(list(range(len(grad_norms))), grad_norms)
        #    plt.show(block=False)
        u.save_fig("Gradient_Norms_per_Step_" + name, grad_norms, self.plot_dir)

        plt.figure()
        gradmul = np.multiply(grad_norms, stepsizes_on_line)
        plt.bar(list(range(len(grad_norms))), gradmul)
        #   plt.show(block=False)
        u.save_fig("Grad_Norm_*_Stepsize_per_Step_" + name, gradmul, self.plot_dir)

        #
        u.plot_train_loss_wrt_time(calc_times_per_step, train_losses, name, self.plot_dir)
        #
        #

        u.plot_angles_path(angles_per_step, stepsizes_on_line, grad_norms, name, self.plot_dir)

        #    plt.show(block=False)
        plt.figure()
        u.save_fig("Mean_Train_Loss_per_Minute_" + name, mean_train_losses_per_min, self.plot_dir)

        plt.figure()
        plt.bar(list(range(len(grad_norms))), grad_norms)
        #    plt.show(block=False)
        u.save_fig("Evaluation_Accurarcy_per_Minute_ " + name, evaluation_accuracies_per_min, self.plot_dir)
