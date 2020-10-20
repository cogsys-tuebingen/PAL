import tensorflow as tf

from code_.optimizers.optimizers import PAOptimizerSuper


# noinspection PyChainedComparisons,PyAttributeOutsideInit,PyPep8
class SLS():
    def __init__(self, n_batches_per_epoch=351.0,
                 init_step_size=1.0,
                 c=0.1,
                 beta_b=0.9,
                 gamma=2.0):
        """Implements stochastic line search
        `paper <https://arxiv.org/abs/1905.09997>`_.
        Arguments:
            n_batches_per_epoch (int, recommended):: the number batches in an epoch
            init_step_size (float, optional): initial step size (default: 1)
            c (float, optional): armijo condition constant (default: 0.1)
            beta_b (float, optional): multiplicative factor for decreasing the step-size (default: 0.9)
            gamma (float, optional): factor used by Armijo for scaling the step-size at each line-search step (default: 2.0)
        """

        super().__init__()
        self.n_batches_per_epoch = n_batches_per_epoch
        self.init_step_size = init_step_size
        self.c = c
        self.beta_b = beta_b
        self.gamma = gamma
        self.epsilon = 1e-8

    # region build methods
    def reset_learning_rate(self):
        lr_update = self.current_learning_rate * self.gamma ** (1.0 / self.n_batches_per_epoch)
        self.current_learning_rate = min(10.0, lr_update)

    def is_armijo_condition_satisfied(self, first_loss, current_loss, grad_norm):
        lr = self.current_learning_rate
        break_condition = current_loss - (first_loss - self.c * lr * grad_norm ** 2)
        # print("break condition \t", break_condition)
        is_armijo = break_condition <= 0
        return is_armijo

    def initialize(self, session, grads, loss_tensor, global_step=None, plot_dir=None,
                   switch_reuse_binary_tensor_var=None, *args):
        self._sess = session
        self._loss_tensor = loss_tensor
        self.plot_dir = plot_dir  # unused
        if global_step is None:
            self._global_step = tf.Variable(1.0, trainable=False, name="global_step")
        else:
            self._global_step = global_step
        if switch_reuse_binary_tensor_var != None:
            self._switch_reuse_binary_tensor_var = switch_reuse_binary_tensor_var
        else:
            self._switch_reuse_binary_tensor_var = tf.zeros(0)
        # reuse binary mask for neurons of Dropout layers, since fixing the random seed is not possible in tf


        self.current_learning_rate = self.init_step_size

        self._sess.__enter__()
        self._train_vars = [e[1] for e in grads]
        self._gradient_tensors = [e[0] for e in grads]
        for train_var, grad_tensor in zip(self._train_vars, self._gradient_tensors):
            if not isinstance(grad_tensor, tf.IndexedSlices):
                grad_tensor.set_shape(train_var.get_shape())

        self._increase_global_step_op = tf.assign(self._global_step, self._global_step + 1)
        self._create_copy_weights_ops()
        self._create_save_gradients_ops()
        self._create_gradient_norm_ops()
        self._create_weight_update_ops()
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self._loss_tensor = tf.identity(self._loss_tensor)
        # self._create_weight_norm_ops()

        return

    def _create_weight_update_ops(self):
        self.current_learning_rate_plh = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        assign_ops = []
        for var, initial_var, grad in zip(self._train_vars, self.train_var_copies, self.gradient_vars):
            ass_op = tf.assign(var, initial_var - self.current_learning_rate_plh * grad)
            assign_ops.append(ass_op)
        self.update_step = tf.group(assign_ops)

    def _create_gradient_norm_ops(self):
        with tf.name_scope("gradient_norm"):
            norm_grad = tf.constant(0.0)
            for grad_var in self.gradient_vars:
                norm_grad = tf.add(norm_grad, tf.reduce_sum(tf.multiply(grad_var, grad_var)))
            norm_grad = tf.sqrt(norm_grad)
            self.norm_grad_op = norm_grad

    def _create_copy_weights_ops(self):
        with tf.variable_scope("train_var_copy"):
            self.train_var_copies = []
            for train_var in self._train_vars:
                new_var = tf.Variable(tf.zeros(train_var.shape), trainable=False, name=train_var.name[0:-2])
                self.train_var_copies.append(new_var)
            self.copy_train_var_values_ops = []
            for train_var, train_var_copy in zip(self._train_vars, self.train_var_copies):
                self.copy_train_var_values_ops.append(train_var_copy.assign(train_var))

    def _create_save_gradients_ops(self):
        with tf.variable_scope("train_var_copy"):
            self.gradient_vars = []
            for train_var in self._train_vars:
                new_var = tf.Variable(tf.zeros(train_var.shape), trainable=False, name=train_var.name[0:-2])
                self.gradient_vars.append(new_var)
            self.gradient_save_ops = []
            for gradient, gradient_var in zip(self._gradient_tensors, self.gradient_vars):
                self.gradient_save_ops.append(gradient_var.assign(gradient))

    def do_train_step(self, additional_ops):
        self._sess.run([self.copy_train_var_values_ops])
        self.reset_learning_rate()
        first_loss, additional_res, _ = self._sess.run([self._loss_tensor, additional_ops, self.gradient_save_ops])
        grad_norm = self._sess.run((self.norm_grad_op))

        if grad_norm < self.epsilon:
            return first_loss, additional_res, 0, grad_norm, 0
        found = False
        num_steps = 0
        self._sess.run(self._switch_reuse_binary_tensor_var)
        for e in range(100):
            num_steps += 1

            self._sess.run([self.update_step], feed_dict={self.current_learning_rate_plh: self.current_learning_rate})
            current_loss = self._sess.run((self._loss_tensor))
            if self.is_armijo_condition_satisfied(first_loss, current_loss, grad_norm):
                found = True
                break
            self.current_learning_rate *= self.beta_b  # like in their code but not like in the paper

        if not found:
            self._sess.run([self.update_step],
                           feed_dict={self.current_learning_rate_plh: 1e-6})

        self._sess.run(self._increase_global_step_op)
        self._sess.run(self._switch_reuse_binary_tensor_var)

        return first_loss, additional_res, num_steps, grad_norm, self.current_learning_rate
