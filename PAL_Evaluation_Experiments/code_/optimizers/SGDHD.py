import tensorflow as tf

from tensorflow.python.training import optimizer


# experiment: adapt

# noinspection PyChainedComparisons,PyAttributeOutsideInit,PyPep8
class SGDHD(optimizer.Optimizer):

    def __init__(self, learning_rate=0.01, hyper_gradient_learning_rate=0.001, global_step=None,use_locking=False,name="SGD-HD"):
        super().__init__(use_locking,name)
        self.initial_learning_rate = learning_rate
        self.hyper_gradient_learning_rate = hyper_gradient_learning_rate
        self.epsilon = 1e-8
        if global_step is None:
            self._global_step = tf.Variable(0, trainable=False, name="global_step")
        else:
            self._global_step = global_step
        self.learning_rate_var = tf.Variable([self.initial_learning_rate], trainable=False, name="learning_rate_var")
        #self._sess.__enter__()
        self._increase_global_step_op = tf.assign(self._global_step, self._global_step + 1)

    def apply_gradients(self,grad_var_tuples, global_step, name="train_optimizer"):

        self._train_vars = [x[1] for x in grad_var_tuples]
        self._gradient_tensors = [x[0] for x in grad_var_tuples]
        return self._initialize_train_ops(global_step)


    def _initialize_train_ops(self, global_step):

        if global_step is None:
            self._global_step = tf.Variable(0.0, trainable=False, name="global_step",dtype=tf.float32)
        else:
            self._global_step = global_step

        with tf.variable_scope("u_Variables"):
            self._u_vars = []
            self._u_vars_assign_ops = []
            for gradient_tensor in self._gradient_tensors:
                new_var_1 = tf.Variable(tf.zeros_like(gradient_tensor), trainable=False,
                                        name=gradient_tensor.name[0:-2] + "m_1")
                self._u_vars.append(new_var_1)
                u_op = new_var_1.assign(-gradient_tensor)
                # u_cond= tf.cond(tf.equal(self._global_step,0), lambda: new_var_1.assign(tf.zeros_like(gradient_tensor)),lambda: u_op )
                self._u_vars_assign_ops.append(u_op)

        with tf.name_scope("learning_rate_update"):
            h = tf.constant([0.0])
            for gradient, u_var in zip(self._gradient_tensors, self._u_vars):
                h = tf.add(h, tf.reduce_sum(tf.multiply(gradient, u_var)))
            new_learning_rate = self.learning_rate_var - self.hyper_gradient_learning_rate * h
            #print_new_learning_rate = tf.print(new_learning_rate, [])
            learning_rate_update_op = tf.assign(self.learning_rate_var, new_learning_rate)

        with tf.name_scope("SGD_Update_Operators"):
            self.weight_vars_assign_ops = []
            for weight_matrix, gradient in zip(self._train_vars, self._gradient_tensors):
                update_op = tf.assign_add(weight_matrix, -new_learning_rate * gradient)
                self.weight_vars_assign_ops.append(update_op)
        with tf.control_dependencies([self._increase_global_step_op, learning_rate_update_op]+self._u_vars_assign_ops):
            a = tf.group(self.weight_vars_assign_ops)
        return a


    def minimize(self,loss_tensor,global_step=None):
        self._train_vars = tf.trainable_variables()
        self._gradient_tensors = tf.gradients(loss_tensor, self._train_vars, colocate_gradients_with_ops=True)

        return self._initialize_train_ops(global_step)
