import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
general util methods needed by multiple classes in this framework
"""


def get_calc_angel_op(vars, grad_ops):
    """
    returns angle in radian
    :param vars: var containing the update direction (might not be normalized)
    :return:
    """

    gradient_tensors = [e[0] for e in grad_ops]
    epsilon = 10 ** (-8)

    # create angle obs:
    with tf.name_scope("Angle_Operator"):
        scalar_prod = None
        abs_old = None
        abs_new = None
        for direction, new_gradient in zip(vars, gradient_tensors):
            mul_scalar = tf.multiply(new_gradient, direction)
            mul_abs_old = tf.multiply(direction, direction)
            mul_abs_new = tf.multiply(new_gradient, new_gradient)
            if scalar_prod is not None:
                scalar_prod = tf.add(scalar_prod, tf.reduce_sum(mul_scalar))
                abs_old = tf.add(abs_old, tf.reduce_sum(mul_abs_old))
                abs_new = tf.add(abs_new, tf.reduce_sum(mul_abs_new))
            else:
                scalar_prod = tf.reduce_sum(mul_scalar)
                abs_old = tf.reduce_sum(mul_abs_old)
                abs_new = tf.reduce_sum(mul_abs_new)
        r = scalar_prod / tf.sqrt(abs_old * abs_new)
        a = tf.cond(tf.equal(0.0, r), lambda: epsilon, lambda: r)
    step_direction_angle = tf.acos(a)

    return step_direction_angle


def get_calc_norm_op(vars):
    norm = tf.constant(0.0)
    for var in vars:
        norm = tf.add(norm, tf.reduce_sum(tf.multiply(var, var)))
    norm = tf.sqrt(norm)
    return norm


def initialize_vars(sess):
    """
    initializes all uninitialized variables
    """
    initialize_uninitialized_variables(sess)


def plot_angles_path(angles, stepsize_on_line, grad_norm, name, dir_):
    """
     plots a 2D graph where a path of all consecutive steps on the loss function  are plotted.
     The length of a steps on the  loss function is given by  step_size_to_minimum*grad_norm

      step_size_to_minimum and the old gradients angle
    :param angles: each element of this list is the angle between the last gradient and the next gradient
    :param stepsize_on_line:
    :param grad_norm:  norm of the gradient
    :param name:
    :param dir_:
    """
    x_path = [0]
    y_path = [0]
    anglesum = 0
    for i in range(len(angles)):
        angle = angles[i]
        step = stepsize_on_line[i]
        norm = grad_norm[i]
        anglesum += angle
        xn = np.cos(anglesum) * step * norm + x_path[i]
        yn = np.sin(anglesum) * step * norm + y_path[i]
        x_path.append(xn)
        y_path.append(yn)

    plt.figure()
    plt.plot(x_path, y_path)
    # plt.show(block=False)
    save_fig("All_Angles_Path_(grad_norm_*_stepsize_on_line)_" + name, (x_path, y_path), dir_)


def save_fig(name, data, dir_):
    """
    saves a fig. In addition its data is saved as pickle files.
    :param name:
    :param data:
    :param dir_:

    """
    plt.title(name)
    plt.savefig(dir_ + name + ".png")
    pickle.dump(data, open(dir_ + name + ".data.pickle", 'wb'))
    plt.close(plt.gcf())


def plot_train_loss_wrt_time(times, losses, name, dir_):
    """
     plotting the loss with respect to time
    :param times:  calculating time for each loss
    :param losses:
    :param name:
    :param dir_:
    :return:
    """
    added_times = [0]
    for i in range(1, len(times)):
        added_times.append(added_times[i - 1] + times[i])
    plt.figure()
    plt.plot(added_times, losses)
    #   plt.show(block=False)
    save_fig("Loss_wrt_time_" + name, (added_times, losses), dir_)


def initialize_uninitialized_variables(sess=None):
    """
     initializes all uninitialized variables. Can lead to problems since ops might get collocated on a device
     thats later on not suitable anymore for these ops.
     """
    # with tf.name_scope("Initialize_Uninitialized_Vars"):
    # old_vars= set(old_vars)
    # sess.run(tf.initialize_variables(set(tf.global_variables()) - old_vars))
    # sess.run(tf.initialize_variables(uninitialized_variables))
    # sess.run([x.assign(x.initialized_value()) for x in tf.global_variables() if x not in exclude])
    sess.run([x.assign(x.initialized_value()) for x in tf.global_variables()])


def check_and_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    return True


def save_eval_data_wrapper(eval_data_wrapper, path):
    with  open(path + eval_data_wrapper.experiment_name + ".ew.pickle", 'wb') as output_file:
        pickle.dump(eval_data_wrapper, output_file)
    time.sleep(1200)


def save_data(data, path, experiment_name, sleeptime):
    with  open(path + experiment_name + ".ew.pickle", 'wb') as output_file:
        pickle.dump(data, output_file)
    if sleeptime != 0:
        time.sleep(sleeptime)


class EvalDataWrapper:
    """
    includes hyper paramerter and the loss per interval per network
    """

    def __init__(self, experiment_name, random_seed, optimizer, train_data_size, train_time, batchsize,
                 measuring_step_size, momentum, loose_approximation_factor, max_stepsize, decay, additional
                 , train_loss_per_interval=[], train_acc_per_interval=[], evaluation_accuracies=[], eval_losses=[],
                 avg_test_acc=None,
                 avg_test_loss=None, is_failed=False, angles_for_each_step=[], step_sizes_for_each_step=[],
                 grad_norms_for_each_step=[], all_first_derivatives=[], all_second_derivatives=[]):
        self.random_seed = random_seed
        self.train_data_size = train_data_size
        self.experiment_name = experiment_name
        self.train_time = train_time
        self.batchsize = batchsize
        self.measuring_step_size = measuring_step_size
        self.momentum = momentum
        self.optimizer = optimizer
        self.loose_approximation_factor = loose_approximation_factor
        self.max_stepsize = max_stepsize
        self.decay = decay
        self.additional = additional
        self.train_loss_per_interval = train_loss_per_interval
        self.train_acc_per_interval = train_acc_per_interval
        self.evaluation_accuracies = evaluation_accuracies
        self.eval_losses = eval_losses
        self.avg_test_acc = avg_test_acc
        self.avg_test_loss = avg_test_loss
        self.is_failed = is_failed
        self.angles_for_each_step = angles_for_each_step
        self.step_sizes_for_each_step = step_sizes_for_each_step
        self.grad_norms_for_each_step = grad_norms_for_each_step
        self.all_first_derivatives = all_first_derivatives
        self.all_second_derivatives = all_second_derivatives
