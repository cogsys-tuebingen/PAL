__author__ = "Maximus Mutschler"
__version__ = "1.0"
__email__ = "maximus.mutschler@uni-tuebingen.de"

import numpy as np
import tensorflow as tf


def get_cifar10_iterator(train_data_size=40000, batch_size=100):
    """
     Creates an iterator that iterates over CIFAR-images and the corresponding labels.
     From all pixels the pixel_mean is substracted. No shuffling is done. The iterator repeats forever.
     Returns:
     iterator, inference_mode_var (must hold an element of type :class:`~code_.abstract_net_class.inferenceMode` )
     ,train_data_size, eval_data_Size, test_data_size
    """
    with tf.variable_scope("Iterator", reuse=tf.AUTO_REUSE):
        with tf.device("device:CPU:0"):
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

            channels = 3
            spatial_resolution = 32
            label_size = 1

            pixel_mean = np.mean(x_train, axis=0)

            x_eval = x_train[train_data_size:, ...]
            x_train = x_train[0:train_data_size, ...]
            y_eval = y_train[train_data_size:, ...]
            y_train = y_train[0:train_data_size, ...]

            eval_data_size = len(y_eval)
            test_data_size = len(y_test)

            x_train = x_train.reshape(train_data_size, spatial_resolution, spatial_resolution, channels).astype('uint8')
            x_eval = x_eval.reshape(eval_data_size, spatial_resolution, spatial_resolution, channels).astype('uint8')
            x_test = x_test.reshape(test_data_size, spatial_resolution, spatial_resolution, channels).astype('uint8')
            y_train = y_train.reshape(train_data_size, label_size).astype('uint8')
            y_eval = y_eval.reshape(eval_data_size, label_size).astype('uint8')
            y_test = y_test.reshape(test_data_size, label_size).astype('uint8')

            num_aug_threads = 8

            def mapf(x, y): return tf.cast(x, tf.float32) - pixel_mean, y

            test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            test_dataset = test_dataset.map(mapf, num_parallel_calls=num_aug_threads)
            test_dataset = test_dataset.repeat()  # repeats for ever
            test_dataset = test_dataset.batch(batch_size)
            test_dataset = test_dataset.prefetch(8)

            eval_dataset = tf.data.Dataset.from_tensor_slices((x_eval, y_eval))
            eval_dataset = eval_dataset.map(mapf, num_parallel_calls=num_aug_threads)
            eval_dataset = eval_dataset.repeat()  # repeats for ever
            eval_dataset = eval_dataset.batch(batch_size)
            eval_dataset = eval_dataset.prefetch(8)

            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_dataset = train_dataset.map(mapf, num_parallel_calls=num_aug_threads)
            train_dataset = train_dataset.repeat()
            train_dataset = train_dataset.batch(batch_size)
            train_dataset = train_dataset.prefetch(8)

            sess=tf.get_default_session()
            iterator, inference_mode_var = _create_handle_iterator(train_dataset, eval_dataset, test_dataset, sess)

            sess.run(tf.global_variables_initializer())
            print("successfully loaded CIFAR data")

            return iterator, inference_mode_var


def _create_handle_iterator(train_dataset, eval_dataset, test_dataset, sess):
    """
    Creates iterator that is dependent on the value of inference_mode_var. Depending on its inference_mode_var's value
    data is loaded either from the train or eval or test dataset.
    :return:  iterator, inference_mode_var
    """
    train_iterator_handle = sess.run((train_dataset.make_one_shot_iterator().string_handle()))
    eval_iterator_handle = sess.run(eval_dataset.make_one_shot_iterator().string_handle())
    test_iterator_handle = sess.run(test_dataset.make_one_shot_iterator().string_handle())

    inference_mode_var = tf.get_variable("mode", shape=(),
                                         initializer=tf.constant_initializer(value=InferenceMode.TRAIN,
                                                                             dtype=tf.uint8),
                                         trainable=False, dtype=tf.uint8)
    cases = tf.case(
        [(tf.equal(inference_mode_var, tf.cast(InferenceMode.TRAIN, tf.uint8)),
          lambda: train_iterator_handle),
         (
             tf.equal(inference_mode_var, tf.cast(InferenceMode.EVAL, tf.uint8)), lambda: eval_iterator_handle),
         (
             tf.equal(inference_mode_var, tf.cast(InferenceMode.TEST, tf.uint8)), lambda: test_iterator_handle)]
    )
    iterator = tf.data.Iterator.from_string_handle(cases,
                                                   train_dataset.output_types,
                                                   train_dataset.output_shapes)
    return iterator, inference_mode_var


class InferenceMode:
    """
    Has to be used to define the inference mode of a network model
    """
    TRAIN = 0
    EVAL = 1
    TEST = 2
    PREDICT = 3
