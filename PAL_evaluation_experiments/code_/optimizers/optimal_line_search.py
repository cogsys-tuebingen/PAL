import os
import tensorflow as tf
from keras.backend import is_tensor


class OptimalLineSearch:
    """
    A binary line search, which estimates the exact location of the minimum on a line
    """
    # internal parameters
    step_to_target_point = 0
    _stepsize_on_line = 0
    _total_step_distance = 0
    is_debug = False
    second_divs = []

    # line plot parameters
    plot_save_dir = None

    def __init__(self, initial_search_step, max_num_of_steps,momentum):
        """
        :param initial_search_step: initial step size for extrapolation
        :param max_num_of_steps: maximal number of search steps
        :param momentum: new search direction is combination of old ones
        """
        super().__init__()

        self.initial_search_step = initial_search_step
        self.max_num_of_search_steps = max_num_of_steps
        self.epsilon = 1e-15
        self.momentum = momentum
        self.calc_exact_directional_derivative = True


        # !!!!!!!!!!!!!!!!!!
        # DON'T INITIALIZE ANY  GRAPH PARTS HERE!! otherwise the seed changes
        # !!!!!!!!!!!!!!!!!!

    # region build methods

    # self,sess,loss,_global_step=None,*args
    def initialize(self, session, grads, loss_tensor, global_step=None, plot_save_dir="./lineplots",switch_reuse_binary_tensor_var=None):
        """
        By calling the operator gets placed into the graph.
        This is separated from the class initialization, since the optimizer has to be set after the weight
        initialization to not change the local seeds.
        All trainable variables have to be created and initialized before this step!
        :param session:
        :param loss_tensor: only scalar loss tensors are supported in the moment
        :param grads: grad_var tuple list
        :param global_step:
        :param plot_save_dir:
        :switch_reuse_binary_tensor_var: tells dropout to keep the last sampled neuron bitmap
        :return: --
        """
        self._switch_reuse_binary_tensor_var = switch_reuse_binary_tensor_var
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
        # Has to be done here to ensure identical seeds
        # Build additional graph variables an operators
        self._create_step_direction_variables()

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
        Creates ops that calculate, adapt and saves the gradient, to get the norm of the gradient and
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
                    # norm = tf.add(norm,tf.reduce_sum(tf.multiply(grad_m, grad_m)))  # element wise
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


    def _create_weight_update_ops(self):
        """
        updates weights by  step_on line * -grad / norm of step direction)
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
        :param additional_ops: additional operators that will get inferred from the graph
        :return:  loss, inter_step_direction_angle, results_of_additional_ops, current_step_direction_vars_norm
        """

        loss, directional_deriv, _, _, results_of_additional_ops = self._sess.run(
            (self._loss_tensor, self.directional_derivative, self._gradient_vars_assign_ops, self.ass_norm_grad_var,
             additional_ops))
        current_step_direction_vars_norm = self._sess.run(self.norm_of_gradient_var)
        self._total_step_distance = 0  # distance already walked on line. Set 0 since new line.
        return loss, directional_deriv, results_of_additional_ops, current_step_direction_vars_norm

    def _do_line_step(self, step_size,get_loss=True, reuse_dropout_binaries=True):
        """
        moves all weights in negative gradient direction by step_size
        :param step_size: in negative gradient direction
        :return: loss at new position
        """
        loss=0
        if step_size != 0:
            self._sess.run(self.weight_vars_assign_ops, feed_dict={self._step_on_line_plh: step_size})
        if get_loss:
            if reuse_dropout_binaries == True and self._switch_reuse_binary_tensor_var is not None:
                self._sess.run(self._switch_reuse_binary_tensor_var)  # switching
            loss = self._sess.run((self._loss_tensor))
            if reuse_dropout_binaries == True and self._switch_reuse_binary_tensor_var is not None:
                self._sess.run(self._switch_reuse_binary_tensor_var)  # switching
        self._total_step_distance += step_size
        return loss




    # endregion

    # region training
    def binary_line_search(self, last_loss, step,counter, is_extrapolate):
       # print("search_step: \t",counter,"\t loss: \t",last_loss,"\t step: \t",step)
        if counter == self.max_num_of_search_steps:
            return last_loss
        counter += 1
        if is_extrapolate:
            current_loss = self._do_line_step(step)
            if current_loss < last_loss:
                return self.binary_line_search(current_loss, step,counter, is_extrapolate)
            else:
                is_extrapolate = False
                self._do_line_step(-step,get_loss=False) # initial position
        if not is_extrapolate: # no else since is_extrapolate may change
            loss_right = self._do_line_step(0.5*step, True)
            if loss_right < last_loss:
                return self.binary_line_search(loss_right, 0.5*step,counter, is_extrapolate)
            loss_left = self._do_line_step(-1*step, True)
            if loss_left < last_loss:
                return self.binary_line_search(loss_left, 0.5*step,counter, is_extrapolate)
            self._do_line_step(0.5 * step, get_loss=False)  # initial position
            if loss_right >= last_loss and loss_left >= last_loss:
                return self.binary_line_search(loss_left, 0.5 * step, counter,is_extrapolate)
        raise Exception("this state should not be possible")



    def do_train_step(self, additional_ops):
        """
        performs the minimum search on one line
        :param additional_ops: additional operators that will get inferred from the graph
        """
        self._total_step_distance = 0
        initial_search_step = self._sess.run(self.initial_search_step) if \
            is_tensor(self.initial_search_step) else self.initial_search_step

        # does step to position on line, which got inferred in the last call of this function
        loss_at_current_position, line_derivative_current_pos, additional_ops_results, norm_of_step_direction \
            = self._get_loss_directional_deriv_and_save_gradient(additional_ops)

        ###
        # endregion
        final_loss  =self.binary_line_search(loss_at_current_position, initial_search_step,0, True)


        self._sess.run(self._increase_global_step_op)

        return loss_at_current_position, final_loss, self._total_step_distance, line_derivative_current_pos





