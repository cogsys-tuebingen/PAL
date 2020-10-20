import tensorflow as tf





class AliGwithMomentum(tf.train.MomentumOptimizer):
    """Optimizer that implements the AliG algorithm.
    """

    #def __init__(self, max_lr=None, momentum=0, use_locking=False, name="AliG", eps=1e-5):
    def __init__(self, max_lr=None, momentum=0, use_locking=False, name="AliG", eps=1e-5):
        super(AliGwithMomentum, self).__init__(
            learning_rate=0.0, momentum=momentum, use_locking=use_locking,
            name=name, use_nesterov=True)
        self._max_lr = max_lr
        self.eps = eps
        self.learning_rate_var = tf.Variable(0.0, trainable=False, name="learning_rate_var")

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=tf.train.Optimizer.GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None):
        self._train_vars = tf.trainable_variables()
        self._gradient_tensors = tf.gradients(loss, self._train_vars, colocate_gradients_with_ops=True)

        return self.update(global_step)

    def apply_gradients(self,grad_var_tuples, global_step, name="alig_optimizer"):
        self.grads_and_vars = grad_var_tuples
        #self._train_vars = [x[1] for x in grad_var_tuples]
        #self._gradient_tensors = [x[0] for x in grad_var_tuples]
        return self.update(global_step)

    def update(self, loss, global_step=None, name=None):
        """
        Re-write of tf.train.Optimizer.minimize
        """
        # first part of method is identical to tf
        # grads_and_vars = optimizer.compute_gradients(
        #     loss, var_list=var_list, gate_gradients=gate_gradients,
        #     aggregation_method=aggregation_method,
        #     colocate_gradients_with_ops=colocate_gradients_with_ops,
        #     grad_loss=grad_loss)
        grads_and_vars = self.grads_and_vars

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars], loss))

        # compute step-size here
        grad_sqrd_norm = sum(tf.norm(grad) ** 2 for grad, _ in grads_and_vars)
        self._learning_rate = loss / (grad_sqrd_norm + self.eps)
        if self._max_lr is not None:
            self._learning_rate = tf.clip_by_value(self._learning_rate, clip_value_min=0,
                                                        clip_value_max=self._max_lr)
            learning_rate_save_op = tf.assign(self.learning_rate_var, self._learning_rate)
        with tf.control_dependencies([ learning_rate_save_op]):
            return super(tf.train.MomentumOptimizer,self).apply_gradients(grads_and_vars, global_step=global_step,
                                             name=name)
#
# class AliGwithoutMomentum(tf.train.GradientDescentOptimizer):
#     """Optimizer that implements the AliG algorithm.
#     """
#
#     def __init__(self, max_lr=None, use_locking=False, name="AliG", eps=1e-5):
#         super(AliGwithoutMomentum, self).__init__(
#             learning_rate=None, use_locking=use_locking, name=name)
#         self._max_lr = max_lr
#         self.eps = eps
#
#     def minimize(self, loss, global_step=None, var_list=None,
#                  gate_gradients=tf.train.Optimizer.GATE_OP, aggregation_method=None,
#                  colocate_gradients_with_ops=False, name=None,
#                  grad_loss=None):
#         return minimize(self, loss, global_step=global_step, var_list=var_list,
#                         gate_gradients=gate_gradients, aggregation_method=aggregation_method,
#                         colocate_gradients_with_ops=colocate_gradients_with_ops, name=name,
#                         grad_loss=grad_loss)
#
# def AliG(max_lr=None, momentum=0, use_locking=False, name="AliG", eps=1e-5):
#     if momentum < 0:
#         raise ValueError("Momentum cannot be negative ({})".format(momentum))
#     elif momentum > 0:
#         return AliGwithMomentum(max_lr=max_lr, momentum=momentum,
#                                 use_locking=use_locking, name=name, eps=eps)
#     else:
#         return AliGwithoutMomentum(max_lr=max_lr, use_locking=use_locking, name=name, eps=eps)