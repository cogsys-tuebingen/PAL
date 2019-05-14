__author__ = "Maximus Mutschler"
__version__ = "1.0"
__email__ = "maximus.mutschler@uni-tuebingen.de"

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import is_tensor
import matplotlib.pyplot as plt
import os



class PalOptimizer:
    """
    Approximates the loss in negative gradient direction with a parabolic function.
    Uses minimum of approximation for weight update.
    !!!!!
    IMPORTANT, READ THE INFORMATION BELOW!!!!
    !!!!!
    This optimizer can't be used like usual TensorFlow optimizers. The reason is that one train step of PAL
    needs multiple graph inferences over the same input data.
    1. Use Variables for your input data. Load a new batch each time before you call PAL's do_train_step method.
    2. Exclude all random operators from  your graph. (like Dropout , ShakeDrop or Shake-Shake).
       In general, they are not supported by PAL, since, if used, the loss function changes with each inference.
       However, random operators are supported, if they are implemented in a way that they can reuse random chosen
       numbers for multiple inferences.
    """

    # tunable parameters
    loose_approximation_factor = None
    momentum = None
    measuring_step_size = None
    max_step_size = None

    # region initialization
    def __init__(self, loss_tensor, measuring_step_size=0.1, momentum=0.6, loose_approximation_factor=0.6,
                 max_step_size=1, gradients=None, global_step=None, calc_exact_directional_derivative=False,
                 is_plot=False, plot_step_interval=10, save_dir="/tmp/pt.lineopt/lines/"):
        """
        :param loss_tensor: only scalar loss tensors are supported in the moment
        :param measuring_step_size: python scalar or tf.tensor are accepted. Should have the same decay as max_step_size
        Good values are between 0.01 and 0.1
        :param momentum: python scalar or tf.tensor are accepted
        Good values are either 0 or 0.6
        :param loose_approximation_factor: intentionally approximates the function with less or more curvature
        Values between 0 and inf (inf exclusive) are accepted. Values <1 lead to approximations with less curvature.
        python scalar or tf.tensor are accepted. Good values are between 0.4 and 0.6
        :param max_step_size: python scalar or tf.tensor are accepted. Same decay as for measuring_step_size should be
        applied
        Good values are between 0.1 and 1.
        :param gradients: (grad,corresponding variable) tuple list
        :param global_step: step counter
        :param calc_exact_directional_derivative: more exact approximation but more time consuming (not recommended)
        :param is_plot: plot lines in gradient direction as well as the parabolic approximation. Latex has to be
        installed for plotting.
        :param plot_step_interval: defines how often a line is plotted.
        :save_dir: plot save location
        """
        if is_plot is True and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self._sess = tf.get_default_session()

        self.loose_approximation_factor = loose_approximation_factor
        self.momentum = momentum
        self.measuring_step_size = measuring_step_size
        self.max_step_size = max_step_size

        self.epsilon = 1e-15

        self._loss_tensor = loss_tensor
        self.calc_exact_directional_derivative = calc_exact_directional_derivative
        self.is_plot = is_plot
        self.plot_step_interval = plot_step_interval
        self.save_dir = save_dir

        if global_step is None:
            self._global_step = tf.Variable(0.0, trainable=False, name="global_step")
        else:
            self._global_step = global_step

        if gradients is None:
            self._train_vars = tf.trainable_variables()
            self._gradient_tensors = list(tf.gradients(loss_tensor, tf.trainable_variables()))

        else:
            self._train_vars = [e[1] for e in gradients]
            self._gradient_tensors = [e[0] for e in gradients]  # same order holds with tf >= 1.10

        self._increase_global_step_op = tf.assign(self._global_step, self._global_step + 1)
        self._step_on_line_plh = tf.placeholder(tf.float32, shape=(), name="line_step_size")

        # Build additional graph variables and operators
        self._create_line_gradient_variables()
        self._create_momentum_norm_and_derivative_ops()
        self._create_weight_update_ops()
        print("successfully initialized PAL optimizer")
        return

    def _create_line_gradient_variables(self):
        """
        Creates variables in which the current adapted gradient (= negative line direction)
        (usually -(last_gradient*momentum+gradient)) is saved.
        :return: --
        """
        with tf.variable_scope("Line_Direction_Variables"):
            self.gradient_vars = []
            for gradient_tensor in self._gradient_tensors:
                new_var = tf.Variable(tf.zeros(gradient_tensor.shape), trainable=False, name=gradient_tensor.name[0:-2])
                self.gradient_vars.append(new_var)

    def _create_momentum_norm_and_derivative_ops(self):
        """
        Creates ops that calculate, adapt and saves the gradient, to get the norm of the gradient and
        to get the directional derivative.
        :return: --
        """
        with tf.name_scope("Momentum_Norm_Derivative_Operators"):
            self._gradient_vars_assign_ops = []
            directional_deriv = tf.constant(0.0)
            norm_grad_mom = tf.constant(0.0)
            self.norm_of_gradient_var = tf.Variable(0.0, trainable=False, name="norm_of_gradient_var")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # important for batch normalization
            if self.momentum != 0:
                for grad_tensor, gradient_var in zip(self._gradient_tensors, self.gradient_vars):
                    # Update ops important for batch normalization
                    # since _gradient_vars_assign_ops is always evaluated together with the loss op it
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
                    directional_deriv = -norm_grad_mom
            else:
                for grad_tensor, gradient_var in zip(self._gradient_tensors, self.gradient_vars):
                    # Update ops important for batch normalization
                    # since _gradient_vars_assign_ops is always evaluated together with the loss op it
                    # is valid to put the dependency here
                    norm_grad_mom = tf.add(norm_grad_mom, tf.reduce_sum(tf.multiply(grad_tensor, grad_tensor)))
                    with tf.control_dependencies(update_ops):
                        grad_var_ass_op = (gradient_var.assign(grad_tensor)).op
                    self._gradient_vars_assign_ops.append(grad_var_ass_op)
                norm_grad_mom = tf.sqrt(norm_grad_mom)
                norm_grad_mom = tf.cond(tf.equal(norm_grad_mom, 0.0), lambda: self.epsilon, lambda: norm_grad_mom)
                directional_deriv = -norm_grad_mom
            self.ass_norm_grad_var = tf.assign(self.norm_of_gradient_var, norm_grad_mom)
            self.directional_derivative = directional_deriv

    def _create_weight_update_ops(self):
        """
        Updates weights by step_on line * -grad / norm of step direction)
        :return: --
        """
        with tf.name_scope("Weight_Update_Operators"):
            self.weight_vars_assign_ops = []
            for weight_matrix, grad in zip(self._train_vars, self.gradient_vars):
                self.weight_vars_assign_ops.append(
                    tf.assign_add(weight_matrix, self._step_on_line_plh * -grad / self.norm_of_gradient_var).op)

    def _get_loss_directional_deriv_and_save_gradient(self, additional_ops):
        """
        Calculates loss and gradient at the current position. Saves the (adapted)
        gradient to gradient vars.
        :param additional_ops: additional operators that will get inferred from the graph
        :return: loss, results_of_additional_ops,
        """
        loss, directional_deriv, _, _, results_of_additional_ops = self._sess.run(
            (self._loss_tensor, self.directional_derivative, self._gradient_vars_assign_ops, self.ass_norm_grad_var,
             additional_ops))
        return loss, directional_deriv, results_of_additional_ops

    def _do_line_step(self, step_size):
        """
        moves all weights in negative gradient direction by step_size
        :param step_size: in negative gradient direction
        :return: loss at new position
        """
        if step_size != 0:
            self._sess.run(self.weight_vars_assign_ops, feed_dict={self._step_on_line_plh: step_size})
        loss = self._sess.run((self._loss_tensor))
        return loss

    # endregion

    # region training

    def do_train_step(self, additional_ops):
        """
        Does one training step. Look at the class documentation to get the requirements needed for a successful
        update step. Approximates a 1 dimensional parabolic function along the negative gradient direction.
        If the approximation is a negative line, a step of measuring_step_size is done in line direction.
        If the approximation is an other, unsuited parabola, no update step is done.
        :param additional_ops: additional operations that infer information from the graph
        :return: loss (before parameter update), additional_ops_results
        """
        max_step_size = self._sess.run(self.max_step_size) if is_tensor(self.max_step_size) else self.max_step_size
        loose_approximation_factor = self._sess.run(self.loose_approximation_factor) if \
            is_tensor(self.loose_approximation_factor) else self.loose_approximation_factor
        measuring_step = self._sess.run(self.measuring_step_size) if \
            is_tensor(self.measuring_step_size) else self.measuring_step_size

        # does step to position on line, which got inferred in the last call of this function
        loss_at_current_position, line_derivative_current_pos, additional_ops_results = \
            self._get_loss_directional_deriv_and_save_gradient(additional_ops)

        loss1 = loss_at_current_position
        loss2 = self._do_line_step(measuring_step)

        first_derivative_current_line_pos_plus_one_half = (loss2 - loss1) / measuring_step
        second_derivative_current_line_pos = loose_approximation_factor * (
                first_derivative_current_line_pos_plus_one_half - line_derivative_current_pos) / \
                (measuring_step / 2)

        if np.isnan(second_derivative_current_line_pos) or np.isnan(line_derivative_current_pos) \
                or np.isinf(second_derivative_current_line_pos) or np.isinf(line_derivative_current_pos):
                return loss1, additional_ops_results


        if second_derivative_current_line_pos > 0 and line_derivative_current_pos < 0:
            # approximation is positive (convex) square function.
            # Minimum is located in positive line direction. Should be the primary case.
            step_size_on_line = - line_derivative_current_pos / second_derivative_current_line_pos

        elif second_derivative_current_line_pos <= 0 and line_derivative_current_pos < 0:
            # l''<0, l'<0 approximation is negative (concave) square function.
            # maximum is located in negative line direction.
            # l''==0, l'<0  approximation is negative line
            # Second step was more negative. so we jump there.
            step_size_on_line = measuring_step
        else:
            #  l'>0  can't happen since the first derivative is the norm of the gradient
            #  l'==0
            #  the current position is already an optimum
            step_size_on_line = 0

        if step_size_on_line > max_step_size:
            step_size_on_line = max_step_size

        step_to_target_point = step_size_on_line - measuring_step

        # plotting
        if self.is_plot:
            global_step = self._sess.run(self._global_step)
            if global_step % self.plot_step_interval == 1:
                self.plot_loss_line_and_approximation(measuring_step / 10, step_to_target_point,
                                                      measuring_step, second_derivative_current_line_pos,
                                                      line_derivative_current_pos, loss1, loss2, self.save_dir)

        if step_to_target_point != 0:
            self._sess.run(self.weight_vars_assign_ops, feed_dict={self._step_on_line_plh: step_to_target_point})

        self._sess.run(self._increase_global_step_op)

        return loss1, additional_ops_results

    def __str__(self):
        dict_ = {"measuring_step_size": self.measuring_step_size, "momentum": self.momentum,
                 "loose_approximation_factor": self.loose_approximation_factor, "max_step_size": self.max_step_size}
        param_string = ', '.join("{!s}={!r}".format(k, v) for (k, v) in dict_.items())
        return self.__class__.__name__ + "_" + param_string

    # region plotting
    def plot_loss_line_and_approximation(self, resolution, a_min, mu, loss_d2, loss_d1_0, loss_0, loss_mu,
                                         save_dir):
        real_a_min = a_min + mu
        line_losses = []
        max_step = max(real_a_min * 2, mu)
        interval = list(np.arange(-resolution, max_step + 2 * resolution, resolution))
        line_losses.append(self._do_line_step(-mu - resolution))

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

        plt.rc('text', usetex=True)
        plt.rc('font', serif="Times")
        scale_factor = 1
        tick_size = 21 * scale_factor
        labelsize = 23 * scale_factor
        headingsize = 26 * scale_factor
        fig_sizes = np.array([10, 8]) * scale_factor
        linewidth = 4.0

        fig = plt.figure(0)
        fig.set_size_inches(fig_sizes)
        plt.plot(x, line_losses, linewidth=linewidth)
        plt.plot(x, approx_values, linewidth=linewidth)
        plt.plot(x2, grad_values, linewidth=linewidth)
        plt.axvline(real_a_min, color="red", linewidth=linewidth)
        y_max = max(line_losses)
        y_min = min(min(approx_values), min(line_losses))
        plt.ylim([y_min, y_max])
        plt.scatter(0, loss_0, color="black", marker='x', s=100, zorder=10, linewidth=linewidth)
        plt.scatter(mu, loss_mu, color="black", marker='x', s=100, zorder=10, linewidth=linewidth)
        plt.legend(["loss", "approximation", "derivative", "update step", "loss measurements"], fontsize=labelsize,
                   loc="upper center")
        plt.xlabel(r"step on line", fontsize=labelsize)
        plt.ylabel("loss in line direction", fontsize=labelsize)

        plt.title("update step {0:d}".format(global_step), fontsize=headingsize)

        plt.gca().tick_params(
            axis='both',
            which='both',
            labelsize=tick_size
        )
        plt.gca().ticklabel_format(style='sci')
        plt.gca().yaxis.get_offset_text().set_size(tick_size)

        plt.savefig("{0}line{1:d}.png".format(save_dir, global_step))
        print("plottet line {0}line{1:d}.png".format(save_dir, global_step))
        # plt.show(block=True)
        plt.close(0)
        self._do_line_step(-(len(interval) - 1) * resolution + mu)

    #endregion
