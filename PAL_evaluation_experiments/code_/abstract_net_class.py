import abc
import tensorflow as tf

class AbstractNetClass(metaclass=abc.ABCMeta):
    """
    All network model used within this framework must inherit from this class
    """

    @classmethod
    def get_model(cls,iterator, inference_mode_var, batch_size, num_gpus):
        """
         returns the network model
        :param iterator:  to load a new data element
        :param inference_mode_var:  holds a value defining the  inference mode of a network model.
        Hold value must be of type :class:`InferenceMode`.
        :param batch_size: will be divided on num_gpu gpus
        :param num_gpus:
        :return: loss , __y_pred, acc_op, acc_update_op, batch_assign_ops
        """
        if batch_size % num_gpus !=0:
            raise ValueError("batch_size of {0:d} not multiple of num_gpu {1:d}".format(batch_size, num_gpus))

        div_batch_size = batch_size // num_gpus

        losses=[]
        acc_ops=[]
        acc_update_ops=[]
        batch_assign_ops=[]
        gradients=[]
        if num_gpus >1:
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(num_gpus):
                    with tf.device("/device:GPU:{:d}".format(i)):
                        with tf.name_scope("gpu_{:d}/model".format(i)) :

                            loss , y_pred, y_true,batch_assign_op,reuse_binary_ops= cls._get_loss_y_pred_y_true_batch_assign_op(iterator, inference_mode_var, div_batch_size)

                            predicted_class = tf.argmax(y_pred, axis=1)
                            acc_op, acc_update_op = tf.metrics.accuracy(y_true, predicted_class)


                            # Weight decay!
                            train_vars = tf.trainable_variables()
                            reg_loss = tf.constant(0.0)
                            for train_var in train_vars:
                                reg_loss += tf.nn.l2_loss(train_var)
                            reg_constant = 0.0001
                            loss_with_reg = loss + reg_constant * reg_loss

                            gradient_tensors = tf.gradients(loss_with_reg, train_vars,colocate_gradients_with_ops=True)
                            gradients_and_vars = list(zip(gradient_tensors,train_vars))
                            gradients.append(gradients_and_vars)
                            losses.append(loss_with_reg)
                            acc_ops.append(acc_op)
                            acc_update_ops.append(acc_update_op)
                            batch_assign_ops.append(batch_assign_op)

                            #reuse_binary_ops are shared so we just need one of the retuned lists
                            tf.get_variable_scope().reuse_variables() # use same weights on each gpu

                average_grads= AbstractNetClass.__average_gradients(gradients)
                stacked_losses= tf.stack(losses)
                averaged_loss= tf.reduce_mean(stacked_losses) # TODO check




                average_acc_ops= tf.reduce_mean(tf.stack(acc_ops))
                combined_acc_update_ops= tf.group(*acc_update_ops)
                combined_batch_assign_ops= tf.group(*batch_assign_ops)
                return average_grads ,averaged_loss,loss_with_reg, average_acc_ops, combined_acc_update_ops, combined_batch_assign_ops,reuse_binary_ops
        else:
            loss, y_pred, y_true, batch_assign_op, reuse_binary_ops = cls._get_loss_y_pred_y_true_batch_assign_op(iterator,
                                                                                                inference_mode_var,
                                                                                            div_batch_size)
            y_true = tf.reshape(y_true, [batch_size, ])

            predicted_class = tf.argmax(y_pred, axis=1)
            acc_op, acc_update_op = tf.metrics.accuracy(y_true, predicted_class)



            #Weight decay!
            train_vars = tf.trainable_variables()
            reg_loss=tf.constant(0.0)
            for train_var in train_vars:
                reg_loss+=tf.nn.l2_loss(train_var)
            reg_constant = 0.0001
            loss_with_reg = loss + reg_constant * reg_loss

            gradient_tensors = tf.gradients(loss_with_reg, train_vars, colocate_gradients_with_ops=True)
            gradients_and_vars = list(zip(gradient_tensors, train_vars))

            return gradients_and_vars, loss_with_reg,loss, acc_op, acc_update_op, batch_assign_op, reuse_binary_ops





    @classmethod
    @abc.abstractmethod
    def _get_loss_y_pred_y_true_batch_assign_op(cls,iterator, inference_mode_var, batch_size):
        """
        Has to be implemented by every subclass. Is defining the model.
        :return: loss,y_pred,y_true,batch_assign_op
        """
        return None,None,None,None

    @classmethod
    @abc.abstractmethod
    def get_name(cls):
        """

        :return: the name of this network
        """
        return None


    @staticmethod
    @abc.abstractmethod
    def __average_gradients(tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads




class InferenceMode:
    """
    Has to be used to define the inference mode of a network model
    """
    TRAIN = 0
    EVAL = 1
    TEST = 2
    PREDICT = 3
