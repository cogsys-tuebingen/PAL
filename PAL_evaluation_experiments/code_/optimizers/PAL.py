import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.backend import is_tensor

from code_.optimizers.optimizers import PAOptimizerSuper


# noinspection PyChainedComparisons,PyAttributeOutsideInit,PyPep8
class PAL(PAOptimizerSuper):
    """
    Approximates the loss in the direction of the gradient with a parabolic function.
    Uses minimum of approximation for weight update.
    """

    # statistics:
    norm_cases = 0  # approximation is positive parabola in direction of the gradient
    neg_jump_to_max_or_neg_line = 0  # approximation is negative parabola with negative jump to the maximum
    jump_overs = 0  # jump to minimum is larger than max_step_size
    no_min_in_step_direction_cases = 0  # all other cases. In this case max_step_size


    # internal parameters
    step_to_target_point = 0
    _stepsize_on_line = 0
    _total_step_distance = 0
    is_debug = False
    second_divs = []

    # tunable parameters
    loose_approximation_factor = None  # TODO write getter and setter -> Properties
    momentum = None
    measuring_step_size = None
    max_stepsize = None
    max_stepsize_and_measurement_decay = None

    # line plot parameters
    plot_save_dir = None

    def __init__(self, decayf, measuring_step_size=0.01, momentum=0.4, loose_approximation_factor=1, max_step_size=0.2,
                 is_debug=False, is_plot=False, plot_step_interval=1):
        """
        :param decayf: possible decay for the maximal and measuring step size. Not used in the evaluation.
        :param measuring_step_size: step size of the step of the second point to measure
        :param momentum:
        :param loose_approximation_factor:  inverse update step adaptation factor
        :param max_step_size:
        :param is_debug: provides additional information
        :param is_plot: plots lines and the corresponding parabolic approximation
        :param plot_step_interval: plotting interval each plot_step_interval line is plotted
        """

        super().__init__()
        self.decayf = decayf
        self.loose_approximation_factor = loose_approximation_factor
        self.momentum = momentum
        self.measuring_step_size = measuring_step_size
        self.initial_measuring_step_size = measuring_step_size
        self.is_debug = is_debug
        self.max_step_size = max_step_size
        self.initiat_max_step_size = max_step_size

        self.epsilon = 1e-15
        self.plot_step_interval = plot_step_interval
        self.is_plot = is_plot
        self.calc_exact_directional_derivative = True

        # !!!!!!!!!!!!!!!!!!
        # DON'T INITIALIZE ANY  GRAPH PARTS HERE!! otherwise the seed changes
        # !!!!!!!!!!!!!!!!!!

    # region build methods

    # self,sess,loss,_global_step=None,*args
    def initialize(self, session, grads, loss_tensor, global_step=None, plot_save_dir="./lineplots",
                   switch_reuse_binary_tensor_var=None,batch_assign_ops=None):
        """
        By calling the operator gets placed into the graph.
        This is separated from the class initialization, since the optimizer has to be set after the weight
        initialization to not change the local seeds.
        All trainable variables have to be created and initialized before this step!
        :param session:
        :param loss_tensor: only scalar loss tensors are supported in the moment
        :param grads: (gradient,variable) tuple list
        :param global_step:
        :param plot_save_dir:
        :param switch_reuse_binary_tensor_var: tells dropout to keep the last sampled neuron bitmap
        :return: --
        """
        self.batch_assign_ops = batch_assign_ops
        self._sess = session
        self._loss_tensor = loss_tensor
        if global_step is None:
            self._global_step = tf.Variable(1.0, trainable=False, name="global_step")
        else:
            self._global_step = global_step
        self.plot_save_dir = plot_save_dir

        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)

        self._sess.__enter__()
        self._train_vars = [e[1] for e in grads]
        self._gradient_tensors = [e[0] for e in grads]  # same order holds with tf 1.10
        # assure that gradient tensors have the right shape it might happen that some don't have a shape at all:
        for train_var, grad_tensor in zip(self._train_vars, self._gradient_tensors):
            if not isinstance(grad_tensor,tf.IndexedSlices):
                grad_tensor.set_shape(train_var.get_shape())

        self._increase_global_step_op = tf.assign(self._global_step, self._global_step + 1)
        self._step_on_line_plh = tf.placeholder(tf.float32, shape=(), name="line_step_size")
        self._switch_reuse_binary_tensor_var = switch_reuse_binary_tensor_var
        # Has to be done here to ensure identical seeds
        if self.decayf is not None:
            self.measuring_step_size = self.decayf(self._global_step, self.initial_measuring_step_size)
            self.max_step_size = self.decayf(self._global_step, self.initiat_max_step_size)

        # Build additional graph variables an operators
        self._create_step_direction_variables()
        if self.is_debug:
            self._create_gradient_angle_op(self.momentum)
        self._create_momentum_norm_and_derivative_ops()

        self._create_weight_update_ops()

        return

    def _create_step_direction_variables(self):
        """
        creates variables where the current step direction (usually last gradient*momentum+gradient) is saved in
        :return: --
        """
        with tf.variable_scope("Gradient_Variables"):
            self.step_direction_variables = []
            for train_var in self._train_vars:
                # print(gradient_tensor.name,"\t", gradient_tensor.shape)
                new_var = tf.Variable(tf.zeros(train_var.shape), trainable=False, name=train_var.name[0:-2])
                self.step_direction_variables.append(new_var)

    def _create_momentum_norm_and_derivative_ops(self):
        """
        Creates ops that calculate, adapt and saves the gradient, to get the norm of the direction and
        to get the directional derivative.
        :return: --
        """
        with tf.name_scope("Momentum_Norm_Derivative_Operators"):
            self._gradient_vars_assign_ops = []
            directional_deriv = tf.constant(0.0)
            norm_grad_mom = tf.constant(0.0)
            self.norm_of_gradient_var = tf.Variable(0.0, trainable=False, name="norm_of_step_direction_var")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # important for batch normalization
            dependencies = [*update_ops]
            if self.is_debug:
                dependencies.append(self.step_direction_angle_op)
            if self.momentum != 0:
                for grad_tensor, gradient_var in zip(self._gradient_tensors, self.step_direction_variables):
                    # Update ops important for batch normalization
                    # since _conjugate_direction_var_assign_ops is always evaluated together with the loss op it
                    # is valid to put the dependency here
                    grad_mom = gradient_var * self.momentum + grad_tensor
                    norm_grad_mom = tf.add(norm_grad_mom, tf.reduce_sum(tf.multiply(grad_mom, grad_mom)))
                    if self.calc_exact_directional_derivative:
                        directional_deriv = tf.add(directional_deriv, tf.reduce_sum(tf.multiply(grad_tensor, grad_mom)))
                    with tf.control_dependencies(update_ops):
                        grad_var_ass_op = (gradient_var.assign(grad_mom)).op
                    self._gradient_vars_assign_ops.append(grad_var_ass_op)
                norm_grad_mom = tf.sqrt(norm_grad_mom)
                norm_grad_mom = tf.cond(tf.equal(norm_grad_mom, 0.0), lambda: self.epsilon, lambda: norm_grad_mom)
                if self.calc_exact_directional_derivative:
                    directional_deriv = - directional_deriv / norm_grad_mom
                else:
                    directional_deriv = - norm_grad_mom
            else:
                for grad_tensor, gradient_var in zip(self._gradient_tensors, self.step_direction_variables):
                    # Update ops important for batch normalization
                    # since _conjugate_direction_var_assign_ops is always evaluated together with the loss op it
                    # is valid to put the dependency here
                    # flat_grad = tf.reshape(grad_tensor, [-1])  # flatten
                    # norm_grad_mom = tf.add(norm_grad_mom, tf.tensordot(flat_grad, flat_grad, 1))
                    # Attention!! tf.tensordot is ver slow  -> use tf multiply!
                    norm_grad_mom = tf.add(norm_grad_mom, tf.reduce_sum(tf.multiply(grad_tensor, grad_tensor)))
                    with tf.control_dependencies(dependencies):
                        grad_var_ass_op = (gradient_var.assign(grad_tensor)).op
                    self._gradient_vars_assign_ops.append(grad_var_ass_op)
                norm_grad_mom = tf.sqrt(norm_grad_mom)
                norm_grad_mom = tf.cond(tf.equal(norm_grad_mom, 0.0), lambda: self.epsilon, lambda: norm_grad_mom)
                directional_deriv = -norm_grad_mom
            self.ass_norm_grad_var = tf.assign(self.norm_of_gradient_var, norm_grad_mom)
            self.directional_derivative = directional_deriv

    def _create_gradient_angle_op(self, momentum):
        """
        creates an op that calculates the angle between the last step direction saved in the step_direction_vars
        and the new step direction
        :param momentum:
        :return: --
        """
        with tf.name_scope("Gradient_Angle_Operator"):
            scalar_prod = None
            abs_old = None
            abs_new = None
            for grad_tensor, step_direction_var in zip(self._gradient_tensors, self.step_direction_variables):
                new_step_direction = step_direction_var * momentum + grad_tensor
                mul_scalar = tf.multiply(new_step_direction, step_direction_var)
                mul_abs_old = tf.multiply(step_direction_var, step_direction_var)
                mul_abs_new = tf.multiply(new_step_direction, new_step_direction)
                if scalar_prod is not None:
                    scalar_prod = tf.add(scalar_prod, tf.reduce_sum(mul_scalar))
                    abs_old = tf.add(abs_old, tf.reduce_sum(mul_abs_old))
                    abs_new = tf.add(abs_new, tf.reduce_sum(mul_abs_new))
                else:
                    scalar_prod = tf.reduce_sum(mul_scalar)
                    abs_old = tf.reduce_sum(mul_abs_old)
                    abs_new = tf.reduce_sum(mul_abs_new)

            self.step_direction_angle_op = tf.acos(scalar_prod / tf.sqrt(abs_old * abs_new))


    def _create_weight_update_ops(self):
        """
        updates weights by  step_on line * -step_dir / norm of step direction)
        :return: --
        """
        with tf.name_scope("Weight_Update_Operators"):
            self.weight_vars_assign_ops = []
            for weight_matrix, grad in zip(self._train_vars, self.step_direction_variables):
                self.weight_vars_assign_ops.append(
                    tf.assign_add(weight_matrix, self._step_on_line_plh * -grad / self.norm_of_gradient_var).op)

    def _get_loss_directional_deriv_and_save_gradient(self, additional_ops):
        """
        does a step on the line (weight update), calculates loss and gradients at the new position. Sets the new step
        direction vars.
        :param step_size: step done in the current step direction.
        :param additional_ops: additional operators that will get inferred from the graph
        :return:  loss, inter_step_direction_angle, results_of_additional_ops, current_step_direction_vars_norm
        """
        if self.is_debug:
            loss, inter_step_direction_angle, directional_deriv, _, _, results_of_additional_ops = self._sess.run(
                (self._loss_tensor, self.step_direction_angle_op, self.directional_derivative,
                 self._gradient_vars_assign_ops, self.ass_norm_grad_var,
                 additional_ops))
        else:
            loss, directional_deriv, _, _, results_of_additional_ops = self._sess.run(
                (self._loss_tensor, self.directional_derivative, self._gradient_vars_assign_ops, self.ass_norm_grad_var,
                 additional_ops))
            inter_step_direction_angle = 0
        current_step_direction_vars_norm = self._sess.run(self.norm_of_gradient_var)
        self._total_step_distance = 0  # distance already walked on line. Set 0 since new line.
        return loss, directional_deriv, inter_step_direction_angle, results_of_additional_ops, current_step_direction_vars_norm

    def _do_line_step(self, step_size, reuse_dropout_binaries=True):
        """
        moves all weights in negative gradient direction by step_size
        :param step_size: in negative gradient direction
        :return: loss at new position
        """
        if step_size != 0:
            self._sess.run(self.weight_vars_assign_ops, feed_dict={self._step_on_line_plh: step_size})
        if reuse_dropout_binaries == True and self._switch_reuse_binary_tensor_var is not None:
            self._sess.run(self._switch_reuse_binary_tensor_var)  # switching
        loss = self._sess.run((self._loss_tensor))
        if reuse_dropout_binaries == True and self._switch_reuse_binary_tensor_var is not None:
            self._sess.run(self._switch_reuse_binary_tensor_var)  # switching
        self._total_step_distance += step_size
        return loss

    # endregion

    # region training

    def do_train_step(self, additional_ops):
        """
        approximates 1 dimensional parabolic function along the gradient direction using newton's method.
        if the approximation is a line or has no minimum in the direction of the step direction no step is done.
        :param additional_ops: additional operators that will get inferred from the graph
        :return: measuring_step, min_at_current_position, self._stepsize_on_line, [a, b, c, t],
         inter_step_direction_angle, norm_of_step_direction, additional_ops_results
        """
        max_step_size = self._sess.run(self.max_step_size) if is_tensor(self.max_step_size) else self.max_step_size
        loose_approximation_factor = self._sess.run(self.loose_approximation_factor) if \
            is_tensor(self.loose_approximation_factor) else self.loose_approximation_factor
        measuring_step = self._sess.run(self.measuring_step_size) if \
            is_tensor(self.measuring_step_size) else self.measuring_step_size

        # does step to position on line, which got inferred in the last call of this function
        loss_at_current_position, line_derivative_current_pos, inter_step_direction_angle, additional_ops_results, norm_of_step_direction \
            = self._get_loss_directional_deriv_and_save_gradient(additional_ops)

        ###
        # endregion

        loss1 = loss_at_current_position
        loss2 = self._do_line_step(measuring_step, True)


        line_derivative_current_pos_plus_one_half = (loss2 - loss1) / measuring_step
        line_second_derivative_current_pos = loose_approximation_factor * (
                line_derivative_current_pos_plus_one_half - line_derivative_current_pos) / \
                                             (measuring_step / 2)
        #print(line_second_derivative_current_pos)
        if np.isnan(line_second_derivative_current_pos) or np.isnan(line_derivative_current_pos) \
                or np.isinf(line_second_derivative_current_pos) or np.isinf(line_derivative_current_pos):
            return measuring_step, loss_at_current_position, self._stepsize_on_line, \
                   [1, 1, 1, 1], inter_step_direction_angle, norm_of_step_direction, additional_ops_results
        # raise ValueError("first or second derivative is nan or inf -> training is stopped")

        if line_second_derivative_current_pos > 0 and line_derivative_current_pos < 0:  # approximation is positive (convex) square function.
            #  Minimum is located in positive line direction. Should be the primary case.
            self._stepsize_on_line = - line_derivative_current_pos / line_second_derivative_current_pos
            self.norm_cases += 1

        elif line_second_derivative_current_pos <= 0 and line_derivative_current_pos < 0:
            # l''<0, l'<0 approximation is negative (concave) square function.
            # maximum is located in negative line direction.
            # l''==0, l'<0  approximation is negative line
            # Second step was more negarive. so we jump there.
            self._stepsize_on_line = measuring_step
            self.neg_jump_to_max_or_neg_line += 1
        # self.measuring_step_size = self.measuring_step_size * decay
        # self.max_step_size = self.max_step_size * decay

        else:
            #  l''<0, l'>0   approximation is negative (concave) square function
            #     maximum is located in positive line direction
            #  l''>0, l'>0 approximation is positive (convex) square function
            #     minimum is located in negative line direction

            #     Ignoring momentum, this should not happen since the minimum has to be in the direction
            #     of the gradient.
            #     With a reasonable chosen momentum this should still be the case.

            #     l''==0, l'>0
            #     approximation is a positive line
            #     l'==0
            #     the current position is already an extremum
            self._stepsize_on_line = 0
            self.no_min_in_step_direction_cases += 1

        if self._stepsize_on_line > max_step_size:
            self._stepsize_on_line = max_step_size
            self.jump_overs += 1

        self.step_to_target_point = self._stepsize_on_line - measuring_step

        # get parameters of parabolic approximation: f(x)= a(x-t)^2+b(x-t)+c
        a = line_second_derivative_current_pos / 2
        b = line_derivative_current_pos
        c = loss1
        t = 0

        #self.second_divs.append(line_second_derivative_current_pos)

        if self.is_plot:
            global_step = self._sess.run(self._global_step)
            if (global_step-1) % self.plot_step_interval == 0:
                self.save_plot_data(measuring_step / 25, self.step_to_target_point,
                                    measuring_step, line_second_derivative_current_pos, line_derivative_current_pos,
                                    loss1, loss2, self.plot_save_dir)

        if self.step_to_target_point != 0:
            self._sess.run(self.weight_vars_assign_ops, feed_dict={self._step_on_line_plh: self.step_to_target_point})

        self._sess.run(self._increase_global_step_op)

        return measuring_step, loss_at_current_position, self._stepsize_on_line, \
               [a, b, c, t], inter_step_direction_angle, norm_of_step_direction, additional_ops_results




    # region plotting
    @staticmethod
    def parabolic_function(x, a, b, c, t):
        """
        :return:  value of f(x)= a(x-t)^2+b(x-t)+c
        """
        return a * (x - t) ** 2 + b * (x - t) + c

    def save_plot_data(self, resolution, a_min, mu, loss_d2, loss_d1_0, loss_0, loss_mu, save_dir):
        real_a_min = a_min + mu
        line_losses = []
        resolution=resolution*2
        max_step = max(real_a_min * 2, mu)
        max_step = 2#max(real_a_min * 2, mu)
        min_step = 1
        #measure_range= 2*resolution-max_step
        interval = list(np.arange(-2*resolution-min_step, max_step + 2 * resolution*2, resolution))
        line_losses.append(self._do_line_step(-mu - 2*resolution-min_step))

        for i in range(len(interval) - 1):
            line_losses.append(self._do_line_step(resolution))

        # parabola parameters:
        a = loss_d2 / 2
        b = loss_d1_0
        c = loss_0

        def parabolic_function(x, a, b, c):
            """
            :return:  value of f(x)= a(x-t)^2+b(x-t)+c
            """
            return a * x ** 2 + b * x + c

        x = interval
        x2 = list(np.arange(-resolution, 1.1 * resolution, resolution))

        approx_values = [parabolic_function(x_i, a, b, c) for x_i in x]
        grad_values = [b * x2_i + c for x2_i in x2]
        global_step = int(self._sess.run(self._global_step))
        data = (
        x, x2, line_losses, approx_values, grad_values, real_a_min, mu, loss_0, loss_d1_0, loss_d2, loss_mu, resolution,
        global_step)
        pickle.dump(data, open("{0}line{1:d}.data.pickle".format(save_dir, global_step), 'wb'))
        print("saved data  {0}line{1:d}".format(save_dir, global_step))
        positive_steps = sum(i > 0 for i in interval)
        self._do_line_step(- positive_steps * resolution + mu)

    def plot_loss_line_and_approximation(self, resolution, a_min, mu, loss_d2, loss_d1_0, loss_0, loss_mu, save_dir):
        real_a_min = a_min + mu
        line_losses = []
        resolution=resolution*2
        max_step = max(real_a_min * 2, mu)
        max_step = 10#max(real_a_min * 2, mu)
        min_step = 1
        #measure_range= 2*resolution-max_step
        interval = list(np.arange(-2*resolution-min_step, max_step + 2 * resolution*2, resolution))
        line_losses.append(self._do_line_step(-mu - 2*resolution-min_step))

        for i in range(len(interval) - 1):
            # load new batch
            if self.batch_assign_ops is not None:
                self._sess.run(self.batch_assign_ops)
            line_losses.append(self._do_line_step(resolution))


        # parabola parameters:
        a = loss_d2 / 2
        b = loss_d1_0
        c = loss_0

        def parabolic_function(x, a, b, c):
            """
            :return:  value of f(x)= a(x-t)^2+b(x-t)+c
            """
            return a * x ** 2 + b * x + c

        x = interval
        x2 = list(np.arange(-resolution, 1.1 * resolution, resolution))

        plt.rc('text', usetex=True)
        plt.rc('font', serif="Times")
        scale_factor = 1
        tick_size = 23 * scale_factor
        labelsize = 23 * scale_factor
        headingsize = 26 * scale_factor
        fig_sizes = np.array([10, 8]) * scale_factor

        fig = plt.figure(0)
        fig.set_size_inches(fig_sizes)
        plt.plot(x, line_losses, linewidth=3.0)
        approx_values = [parabolic_function(x_i, a, b, c) for x_i in x]
        plt.plot(x, approx_values, linewidth=3.0)
        grad_values = [b * x2_i + c for x2_i in x2]
        plt.plot(x2, grad_values, linewidth=3.0)
        plt.axvline(real_a_min, color="red", linewidth=3.0)
        y_max = max(line_losses)
        y_min = min(min(approx_values), min(line_losses))
        plt.ylim([y_min, y_max])
        plt.legend(["loss", "approximation", "derivative", r"$a_{min}$"], fontsize=labelsize)
        plt.xlabel("step on line", fontsize=labelsize)
        plt.ylabel("loss in line direction", fontsize=labelsize)
        plt.plot(0, loss_0, 'x')
        plt.plot(mu, loss_mu, 'x')

        global_step = int(self._sess.run(self._global_step))
        plt.title("Loss line of step {0:d}".format(global_step), fontsize=headingsize)

        plt.gca().tick_params(
            axis='both',
            which='both',
            labelsize=tick_size)

        plt.savefig("{0}line{1:d}.png".format(save_dir, global_step))
        data = (x, line_losses, approx_values, grad_values, real_a_min, mu, loss_0, loss_d1_0, loss_d2, resolution)
        pickle.dump(data, open("{0}line{1:d}.data.pickle".format(save_dir, global_step), 'wb'))
        print("plottet line {0}line{1:d}.png".format(save_dir, global_step))
        # plt.show(block=True)
        plt.close(0)
        positive_steps = sum(i > 0 for i in interval)
        self._do_line_step(- positive_steps * resolution + mu)



    def __str__(self):
        dict_ = {"measuring_step_size": self.measuring_step_size, "momentum": self.momentum,
                 "loose_approximation_factor": self.loose_approximation_factor, "max_step_size": self.max_step_size,
                 "max_stepsize_and_measurement_decay": self.max_stepsize_and_measurement_decay,
                 "is_debug": self.is_debug}
        param_string = ', '.join("{!s}={!r}".format(k, v) for (k, v) in dict_.items())
        return self.__class__.__name__ + "_" + param_string
