__author__ = "Maximus Mutschler"
__version__ = "1.0"
__email__ = "maximus.mutschler@uni-tuebingen.de"

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import is_tensor


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
    2. Replace all random operators from your graph. (like Dropout , ShakeDrop or Shake-Shake) with implementations
       that can reuse random values multiple times. For Dropout we provide such an impplementation.
       in dropout_with_random_value_reuse.py.
       If there is no alternative available exclude the random operators from your graph.
    """

    # tunable parameters
    update_step_adaptation = None
    conjugate_gradient_factor = None
    measuring_step_size = None
    max_step_size = None

    # region initialization
    def __init__(self, loss_tensor, measuring_step_size=1, conjugate_gradient_factor=0, update_step_adaptation=1 / 0.6,
                 # 1/0.6,
                 max_step_size=10, gradients=None, global_step=None, calc_exact_directional_derivative=True,
                 is_plot=False, plot_step_interval=10, save_dir="/tmp/lines/"):
        """
        :param loss_tensor: only scalar loss tensors are supported in the moment
        :param measuring_step_size: python scalar or tf.tensor are accepted. Should have the same decay as max_step_size
        Good values are between 0.1 and 1
        :param conjugate_gradient_factor: python scalar or tf.tensor are accepted
        Good values are either 0 or 0.4
        :param update_step_adaptation: intentionally increase or decrease the update step by a factor. Good values are between 1.2 and 1.7
        :param max_step_size: python scalar or tf.tensor are accepted. Same decay as for measuring_step_size should be
        applied
        Good values are between 1 and 10.
        :param gradients: (grad,corresponding variable) tuple list
        :param global_step: step counter
        :param calc_exact_directional_derivative: more exact approximation but more time consuming
        :param is_plot: plot lines in gradient direction as well as the parabolic approximation. Latex has to be
        installed for plotting.
        :param plot_step_interval: defines how often a line is plotted.
        :save_dir: plot save location
        """
        if is_plot is True and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self._sess = tf.get_default_session()

        self.update_step_adaptation = float(update_step_adaptation)
        self.conjugate_gradient_factor = float(conjugate_gradient_factor)
        self.measuring_step_size = float(measuring_step_size)
        self.max_step_size = float(max_step_size)

        self.epsilon = 1e-10

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
        Creates variables in which the current line direction is saved.
        (usually -(last_gradient*conjugate_gradient_factor+gradient))
        :return: --
        """
        with tf.variable_scope("Line_Direction_Variables"):
            self.step_direction_variables = []
            for train_var in self._train_vars:
                new_var = tf.Variable(tf.zeros(train_var.shape), trainable=False, name=train_var.name[0:-2])
                self.step_direction_variables.append(new_var)

    def _create_momentum_norm_and_derivative_ops(self):
        """
        Creates ops that calculate, adapt and saves the line direction, to get the norm of the direction and
        to get the directional derivative.
        :return: --
        """
        with tf.name_scope("Momentum_Norm_Derivative_Operators"):
            self._step_direction_vars_assign_ops = []
            directional_deriv = tf.constant(0.0)
            norm_of_step_direction = tf.constant(0.0)
            self.norm_of_gradient_var = tf.Variable(0.0, trainable=False, name="norm_of_step_direction_var")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # Update ops important for batch normalization
            # since _step_direction_vars_assign_ops is always evaluated together with the loss op it
            # is valid to put the dependency here
            if self.conjugate_gradient_factor != 0:
                for grad_tensor, last_step_direction in zip(self._gradient_tensors, self.step_direction_variables):
                    step_direction = last_step_direction * self.conjugate_gradient_factor + grad_tensor
                    norm_of_step_direction = tf.add(norm_of_step_direction,
                                                    tf.reduce_sum(tf.multiply(step_direction, step_direction)))
                    if self.calc_exact_directional_derivative:
                        directional_deriv = tf.add(directional_deriv,
                                                   tf.reduce_sum(tf.multiply(grad_tensor, step_direction)))
                    with tf.control_dependencies(update_ops):
                        grad_var_ass_op = (last_step_direction.assign(step_direction)).op
                    self._step_direction_vars_assign_ops.append(grad_var_ass_op)
                norm_of_step_direction = tf.sqrt(norm_of_step_direction)
                norm_of_step_direction = tf.cond(tf.equal(norm_of_step_direction, 0.0), lambda: self.epsilon,
                                                 lambda: norm_of_step_direction)
                if self.calc_exact_directional_derivative:
                    directional_deriv = - directional_deriv / norm_of_step_direction
                else:
                    directional_deriv = -norm_of_step_direction
            else:
                for grad_tensor, last_step_direction in zip(self._gradient_tensors, self.step_direction_variables):
                    norm_of_step_direction = tf.add(norm_of_step_direction,
                                                    tf.reduce_sum(tf.multiply(grad_tensor, grad_tensor)))
                    with tf.control_dependencies(update_ops):
                        grad_var_ass_op = (last_step_direction.assign(grad_tensor)).op
                    self._step_direction_vars_assign_ops.append(grad_var_ass_op)
                norm_of_step_direction = tf.sqrt(norm_of_step_direction)
                norm_of_step_direction = tf.cond(tf.equal(norm_of_step_direction, 0.0), lambda: self.epsilon,
                                                 lambda: norm_of_step_direction)
                directional_deriv = -norm_of_step_direction
            self.ass_norm_grad_var = tf.assign(self.norm_of_gradient_var, norm_of_step_direction)
            self.directional_derivative = directional_deriv

    def _create_weight_update_ops(self):
        """
        Updates weights by step_on line * -grad / norm of step direction)
        :return: --
        """
        with tf.name_scope("Weight_Update_Operators"):
            self.weight_vars_assign_ops = []
            for weight_matrix, grad in zip(self._train_vars, self.step_direction_variables):
                self.weight_vars_assign_ops.append(
                    tf.assign_add(weight_matrix, self._step_on_line_plh * -grad / self.norm_of_gradient_var).op)

    def _get_loss_directional_deriv_and_save_gradient(self, additional_ops):
        """
        Calculates loss and directionsl derivative at the current position. Saves the (adapted)
        line direction to  line direction vars.
        :param additional_ops: additional operators that will get inferred from the graph
        :return: loss, results_of_additional_ops,
        """
        loss, directional_deriv, _, _, results_of_additional_ops = self._sess.run(
            (self._loss_tensor, self.directional_derivative, self._step_direction_vars_assign_ops,
             self.ass_norm_grad_var,
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
        update_step_adaption = self._sess.run(self.update_step_adaptation) if \
            is_tensor(self.update_step_adaptation) else self.update_step_adaptation
        measuring_step = self._sess.run(self.measuring_step_size) if \
            is_tensor(self.measuring_step_size) else self.measuring_step_size

        # does step to position on line, which got inferred in the last call of this function
        loss_at_current_position, directional_derivative, additional_ops_results = \
            self._get_loss_directional_deriv_and_save_gradient(additional_ops)

        loss_0 = loss_at_current_position
        loss_mu = self._do_line_step(measuring_step)

        b = directional_derivative
        a = (loss_mu - loss_0 - directional_derivative * measuring_step) / (measuring_step ** 2)

        if np.isnan(a) or np.isnan(a) \
                or np.isinf(b) or np.isinf(b):
            return loss_0, additional_ops_results

        if a > 0 and b < 0:
            # approximation is positive (convex) square function.
            # Minimum is located in positive line direction. Should be the primary case.
            update_step = - b / (2 * a) * update_step_adaption

        elif a <= 0 and b < 0:
            # l''<0, l'<0 approximation is negative (concave) square function.
            # maximum is located in negative line direction.
            # l''==0, l'<0  approximation is negative line
            # Second step was more negative. Thus we jump there.
            update_step = measuring_step
        else:
            #  l'>0  can't happen since the first derivative is the norm of the gradient
            #  l'==0
            #  the current position is already an optimum
            update_step = 0

        if update_step > max_step_size:
            update_step = max_step_size

        step_to_target_point = update_step - measuring_step

        # plotting
        if self.is_plot:
            global_step = self._sess.run(self._global_step)
            if global_step % self.plot_step_interval == 1:
                self.plot_loss_line_and_approximation(measuring_step / 10, step_to_target_point,
                                                      measuring_step, a,
                                                      b, loss_0, loss_mu, self.save_dir)

        if step_to_target_point != 0:
            self._sess.run(self.weight_vars_assign_ops, feed_dict={self._step_on_line_plh: step_to_target_point})

        self._sess.run(self._increase_global_step_op)

        return loss_0, additional_ops_results

    def __str__(self):
        dict_ = {"measuring_step_size": self.measuring_step_size,
                 "conjugate_gradient_factor": self.conjugate_gradient_factor,
                 "update_step_adaptation": self.update_step_adaptation, "max_step_size": self.max_step_size}
        param_string = ', '.join("{!s}={!r}".format(k, v) for (k, v) in dict_.items())
        return self.__class__.__name__ + "_" + param_string

    # region plotting
    def plot_loss_line_and_approximation(self, resolution, a_min, mu, a, b, loss_0, loss_mu,
                                         save_dir):
        real_a_min = a_min + mu
        line_losses = []
        resolution = resolution * 2
        #max_step = max(real_a_min * 2, mu)
        max_step = 2  # max(real_a_min * 2, mu)
        min_step = 1
        # measure_range= 2*resolution-max_step
        interval = list(np.arange(-2 * resolution - min_step, max_step + 2 * resolution * 2, resolution))
        line_losses.append(self._do_line_step(-mu - 2 * resolution - min_step))

        for i in range(len(interval) - 1):
            line_losses.append(self._do_line_step(resolution))

        # parabola parameters:
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
        print("plotted line {0}line{1:d}.png".format(save_dir, global_step))
        # plt.show(block=True)
        plt.close(0)
        positive_steps = sum(i > 0 for i in interval)
        self._do_line_step(- positive_steps * resolution + mu)

    # endregion
