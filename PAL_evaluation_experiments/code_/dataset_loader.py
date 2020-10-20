import numpy as np
import tensorflow as tf

from code_ import framework_utils as u
from code_.abstract_net_class import InferenceMode
from glob import glob
import os
import abc
from code_.data_augment import cifar10_augment
from code_.data_augment import cifar100_augment
from pathlib import Path


class AbstractDatasetLoader(metaclass=abc.ABCMeta):
    """
    All Dataset loaders  used with this framework must inherit from this class
    """

    @staticmethod
    @abc.abstractmethod
    def get_iterator(sess, dataset_dir, train_data_size, batch_size, num_gpus):
        """
         creates an iterator that iterates over traindate images and the corresponding labels.
        :param sess:
        :param dataset_dir: path to a directory containing traindata
        :param train_data_size:
        :param batch_size: must be mutiple of num_gpus
        _param num_gpus
        Returns:
         iterator, inference_mode_var (must hold an element of type :class:`~code_.abstract_net_class.inferenceMode` )
         ,train_data_size, eval_data_Size, test_data_size
        """
        pass

    @staticmethod
    def _create_handleiterator(train_dataset, eval_dataset, test_dataset, sess):
        """
        creates iterator that is dependend on the value of inference_mode_var. Depending on its inference_mode_var's value
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
            [(tf.equal(inference_mode_var, tf.cast(InferenceMode.TRAIN, tf.uint8)), lambda: train_iterator_handle),
             (tf.equal(inference_mode_var, tf.cast(InferenceMode.EVAL, tf.uint8)), lambda: eval_iterator_handle),
             (tf.equal(inference_mode_var, tf.cast(InferenceMode.TEST, tf.uint8)), lambda: test_iterator_handle)]
        )
        iterator = tf.data.Iterator.from_string_handle(cases,
                                                       train_dataset.output_types,
                                                       train_dataset.output_shapes)
        return iterator, inference_mode_var


class ImageNetLoader(AbstractDatasetLoader):
    @staticmethod
    def _resize_image(image):
        shape = tf.shape(image)
        height = tf.cast(shape[0], tf.float32)
        width = tf.cast(shape[1], tf.float32)
        new_shorter_edge = tf.constant(226, dtype=tf.float32)
        height_smaller_than_width = tf.less_equal(height, width)
        new_height, new_width = tf.cond(
            height_smaller_than_width,
            lambda: (new_shorter_edge, (new_shorter_edge / height) * width),
            lambda: ((new_shorter_edge / width) * height, new_shorter_edge))
        image = tf.image.resize_images(image,
                                       [tf.cast(new_height, dtype=tf.int32), tf.cast(new_width, dtype=tf.int32)])
        return image

    @staticmethod
    def get_iterator(sess, dataset_dir="", train_data_size=0,
                     batch_size=100, num_gpus=1):
        """
         creates an iterator that iterates over imagenet images and the corresponding labels.
         No shuffling is done (data is alreaday shuffled). The iterator repeats forever. The test and eval iterator are
         iterating over the same dataset in this case.
        :param sess:
        :param dataset_dir: path to a directory containing train and eval TFRecords
        :param train_data_size: unused
        :param batch_size: must be multiple of num_gpus
        _param num_gpus
        Returns:
         iterator, inference_mode_var (must hold an element of type :class:`~code_.abstract_net_class.inferenceMode` )
         ,train_data_size, eval_data_Size, test_data_size
        """

        if batch_size % num_gpus != 0:
            raise ValueError("batsite of {0:d} not multiple of num_gpu {1:d}".format(batch_size, num_gpus))

        div_batch_size = batch_size // num_gpus

        with tf.variable_scope("Iterator", reuse=tf.AUTO_REUSE):
            with tf.device("/cpu:0"):

                files = glob(dataset_dir + "/*")  # TODO check wheter dir exists!!
                if len(files) != 1153:
                    raise ValueError("No Imagenet Dataset at this location: " + dataset_dir)

                train_files = []
                eval_files = []

                for f in files:
                    if "train" in os.path.basename(f):
                        train_files.append(f)
                    elif "validation" in os.path.basename(f):
                        eval_files.append(f)

                num_aug_threads = 8

                buffer_size = 10 ** 6

                prefetch = 8

                decoder = ImageNetLoader.__get__image_net__t_f_record__decoder()

                def _parse_function(example_proto):
                    return (decoder.decode(example_proto, ["image", "label"]))

                eigval = tf.reshape(tf.constant((0.2175, 0.0188, 0.0045)), (3, 1))
                eigvec = tf.constant([[-0.5675, 0.7192, 0.4009],  # eigv are columns
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]])

                def _train_preprocess_function(image, label):
                    image = ImageNetLoader._resize_image(image)
                    image = tf.random_crop(image, [224, 224, 3])
                    image = image / 255.0
                    random_alpha = tf.random_normal((3, 1), -2, 0.1)

                    mul = tf.multiply(random_alpha, eigval)
                    lighting = tf.matmul((eigvec), mul)
                    lighting = tf.reshape(lighting, (3,))

                    image = image + lighting

                    image = tf.cast(image, tf.float32)
                    label = tf.cast(label, tf.int32)

                    return [image, label]

                def _eval_preprocess_function(image, label):
                    image = ImageNetLoader._resize_image(image)
                    image = image / 255.0
                    image = tf.random_crop(image, [224, 224, 3])
                    image = tf.cast(image, tf.float32)
                    label = tf.cast(label, tf.int32)

                    return [image, label]

                eval_dataset = tf.data.TFRecordDataset(eval_files, buffer_size=buffer_size,
                                                       num_parallel_reads=num_aug_threads)
                eval_dataset = eval_dataset.map(_parse_function, num_parallel_calls=num_aug_threads)
                eval_dataset = eval_dataset.map(_eval_preprocess_function, num_parallel_calls=num_aug_threads)
                eval_dataset = eval_dataset.repeat()  # repeats for ever
                eval_dataset = eval_dataset.batch(div_batch_size, drop_remainder=True)
                eval_dataset = eval_dataset.prefetch(prefetch)

                train_dataset = tf.data.TFRecordDataset(train_files, buffer_size=buffer_size,
                                                        num_parallel_reads=num_aug_threads)
                train_dataset = train_dataset.map(_parse_function, num_parallel_calls=num_aug_threads)
                train_dataset = train_dataset.map(_train_preprocess_function, num_parallel_calls=num_aug_threads)
                train_dataset = train_dataset.repeat()
                train_dataset = train_dataset.batch(div_batch_size, drop_remainder=True)
                # batch size shape known if using this method
                train_dataset = train_dataset.prefetch(prefetch)

                iterator, inference_mode_var = super(ImageNetLoader, ImageNetLoader)._create_handleiterator(
                    train_dataset, eval_dataset, eval_dataset, sess)

                u.initialize_vars(sess)
                print("sucessfully loaded ImageNet iterator")

                return iterator, inference_mode_var, 1281167, 50000, 50000

    @staticmethod
    def __get__image_net__t_f_record__decoder():
        slim = tf.contrib.slim
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature(
                (), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature(
                (), tf.string, default_value='jpeg'),
            'image/class/label': tf.FixedLenFeature(
                [], dtype=tf.int64, default_value=-1),
            'image/class/text': tf.FixedLenFeature(
                [], dtype=tf.string, default_value=''),
            'image/object/bbox/xmin': tf.VarLenFeature(
                dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(
                dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(
                dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(
                dtype=tf.float32),
            'image/object/class/label': tf.VarLenFeature(
                dtype=tf.int64),
        }

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
            'label_text': slim.tfexample_decoder.Tensor('image/class/text'),
            'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
            'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)
        return decoder


class Cifar10Loader(AbstractDatasetLoader):
    is_augment = False

    @staticmethod
    def get_iterator(sess, dataset_path="../Datasets/cifar10Dataset.npy", train_data_size=40000, batch_size=100,
                     num_gpus=1):
        """
         creates an iterator that iterates over cifar images and the corresponding labels.
         Data Augementation and shuffling is done. The iterator repeats forever.
        :param sess:
        :param dataset_path: path to a numpy file containing the cifar dataset.
        :param train_data_size: eval data size will be determined out of this.
        :param batch_size: must be multiple of num_gpus
        _param num_gpus
        Returns:
         iterator, inference_mode_var (must hold an element of type :class:`~code_.abstract_net_class.inferenceMode` )
         ,train_data_size, eval_data_Size, test_data_size
        """

        if batch_size % num_gpus != 0:
            raise ValueError("batch_size of {0:d} not multiple of num_gpu {1:d}".format(batch_size, num_gpus))

        div_batch_size = batch_size // num_gpus

        with tf.variable_scope("Iterator", reuse=tf.AUTO_REUSE):
            with tf.device("/cpu:0"):
                # a = np.load("Datasets/cifarDataset.npy")

                if not os.path.exists(dataset_path):
                    from keras.datasets import cifar10
                    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
                    os.makedirs(Path(dataset_path).parent, exist_ok=True)
                    np.save(dataset_path, ((x_train, y_train), (x_test, y_test)))
                (x_train, y_train), (x_test, y_test) = np.load(dataset_path, allow_pickle=True)

                print("loaded data from " + dataset_path)

                channels = 3
                spatial_resolution = 32
                label_size = 1

                mean = np.array([0.4914, 0.4822, 0.4465]) * 255
                std = np.array([0.2023, 0.1994, 0.2010]) * 255

                x_eval = x_train[train_data_size:, ...]
                x_train = x_train[0:train_data_size, ...]
                y_eval = y_train[train_data_size:, ...]
                y_train = y_train[0:train_data_size, ...]

                eval_data_size = len(y_eval)
                test_data_size = len(y_test)

                x_train = x_train.reshape(train_data_size, spatial_resolution, spatial_resolution, channels).astype(
                    'uint8')
                x_eval = x_eval.reshape(eval_data_size, spatial_resolution, spatial_resolution, channels).astype(
                    'uint8')
                x_test = x_test.reshape(test_data_size, spatial_resolution, spatial_resolution, channels).astype(
                    'uint8')
                y_train = y_train.reshape(train_data_size, label_size).astype('uint8')
                y_eval = y_eval.reshape(eval_data_size, label_size).astype('uint8')
                y_test = y_test.reshape(test_data_size, label_size).astype('uint8')

                num_aug_threads = 8
                prefetch = 8

                def mapf_normalize(x, y): return (x - mean) / std, y

                def mapf_cast(x, y): return tf.cast(x, tf.float32), y

                test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
                test_dataset = test_dataset.map(mapf_cast, num_parallel_calls=num_aug_threads)
                test_dataset = test_dataset.map(mapf_normalize, num_parallel_calls=num_aug_threads)
                test_dataset = test_dataset.repeat()  # repeats for ever
                test_dataset = test_dataset.batch(div_batch_size)
                test_dataset = test_dataset.prefetch(prefetch)

                eval_dataset = tf.data.Dataset.from_tensor_slices((x_eval, y_eval))
                eval_dataset = eval_dataset.map(mapf_cast, num_parallel_calls=num_aug_threads)
                eval_dataset = eval_dataset.map(mapf_normalize, num_parallel_calls=num_aug_threads)
                eval_dataset = eval_dataset.repeat()  # repeats for ever
                eval_dataset = eval_dataset.batch(div_batch_size)
                eval_dataset = eval_dataset.prefetch(prefetch)

                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                train_dataset = train_dataset.map(mapf_cast, num_parallel_calls=num_aug_threads)
                if Cifar10Loader.is_augment:
                    train_dataset = train_dataset.map(cifar10_augment, num_parallel_calls=num_aug_threads)
                train_dataset = train_dataset.map(mapf_normalize, num_parallel_calls=num_aug_threads)
                train_dataset = train_dataset.repeat()
                train_dataset = train_dataset.batch(div_batch_size)
                train_dataset = train_dataset.prefetch(prefetch)

                iterator, inference_mode_var = super(Cifar10Loader, Cifar10Loader)._create_handleiterator(train_dataset,
                                                                                                          eval_dataset,
                                                                                                          test_dataset,
                                                                                                          sess)
                u.initialize_vars(sess)
                print("sucessfully loaded CIFAR-10 data")

                return iterator, inference_mode_var, train_data_size, eval_data_size, test_data_size


class Cifar100Loader(AbstractDatasetLoader):
    @staticmethod
    def get_iterator(sess, dataset_path="../Datasets/cifar100Dataset.npy", train_data_size=40000, batch_size=100,
                     num_gpus=1):
        """
         creates an iterator that iterates over cifar images and the corresponding labels.
         Data Augementation and shuffling is done. The iterator repeats forever.
        :param sess:
        :param dataset_path: path to a numpy file containing the cifar dataset.
        :param train_data_size: eval data size will be determined out of this.
        :param batch_size: must be multiple of num_gpus
        _param num_gpus
        Returns:
         iterator, inference_mode_var (must hold an element of type :class:`~code_.abstract_net_class.inferenceMode` )
         ,train_data_size, eval_data_Size, test_data_size
        """

        if batch_size % num_gpus != 0:
            raise ValueError("batch_size of {0:d} not multiple of num_gpu {1:d}".format(batch_size, num_gpus))

        div_batch_size = batch_size // num_gpus

        with tf.variable_scope("Iterator", reuse=tf.AUTO_REUSE):
            with tf.device("/cpu:0"):
                if not os.path.exists(dataset_path):
                    from keras.datasets import cifar100
                    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
                    os.makedirs(Path(dataset_path).parent, exist_ok=True)
                    np.save(dataset_path, (x_train, y_train, x_test, y_test))
                else:
                    x_train, y_train, x_test, y_test = np.load(dataset_path, allow_pickle=True)

                print("loaded data from " + dataset_path)

                channels = 3
                spatial_resolution = 32
                label_size = 1

                mean = np.array([0.4914, 0.4822, 0.4465]) * 255
                std = np.array([0.2023, 0.1994, 0.2010]) * 255

                x_eval = x_train[train_data_size:, ...]
                x_train = x_train[0:train_data_size, ...]
                y_eval = y_train[train_data_size:, ...]
                y_train = y_train[0:train_data_size, ...]

                eval_data_size = len(y_eval)
                test_data_size = len(y_test)

                x_train = x_train.reshape(train_data_size, spatial_resolution, spatial_resolution, channels).astype(
                    'uint8')
                x_eval = x_eval.reshape(eval_data_size, spatial_resolution, spatial_resolution, channels).astype(
                    'uint8')
                x_test = x_test.reshape(test_data_size, spatial_resolution, spatial_resolution, channels).astype(
                    'uint8')
                y_train = y_train.reshape(train_data_size, label_size).astype('uint8')
                y_eval = y_eval.reshape(eval_data_size, label_size).astype('uint8')
                y_test = y_test.reshape(test_data_size, label_size).astype('uint8')

                num_aug_threads = 8
                prefetch = 8

                def mapf_normalize(x, y):
                    return (x - mean) / std, y

                def mapf_cast(x, y):
                    return tf.cast(x, tf.float32), y  #

                test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
                test_dataset = test_dataset.map(mapf_cast, num_parallel_calls=num_aug_threads)
                test_dataset = test_dataset.map(mapf_normalize, num_parallel_calls=num_aug_threads)
                test_dataset = test_dataset.repeat()
                test_dataset = test_dataset.batch(div_batch_size)
                test_dataset = test_dataset.prefetch(prefetch)

                eval_dataset = tf.data.Dataset.from_tensor_slices((x_eval, y_eval))
                eval_dataset = eval_dataset.map(mapf_cast, num_parallel_calls=num_aug_threads)
                eval_dataset = eval_dataset.map(mapf_normalize, num_parallel_calls=num_aug_threads)
                eval_dataset = eval_dataset.repeat()
                eval_dataset = eval_dataset.batch(div_batch_size)
                eval_dataset = eval_dataset.prefetch(prefetch)

                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                train_dataset = train_dataset.map(mapf_cast, num_parallel_calls=num_aug_threads)
                train_dataset = train_dataset.map(cifar100_augment, num_parallel_calls=num_aug_threads)
                train_dataset = train_dataset.map(mapf_normalize, num_parallel_calls=num_aug_threads)
                train_dataset = train_dataset.repeat()
                train_dataset = train_dataset.shuffle(buffer_size=train_data_size // 10)

                train_dataset = train_dataset.batch(div_batch_size)
                train_dataset = train_dataset.prefetch(prefetch)

                iterator, inference_mode_var = super(Cifar100Loader, Cifar100Loader)._create_handleiterator(
                    train_dataset, eval_dataset, test_dataset, sess)

                u.initialize_vars(sess)
                print("successfully loaded CIFAR100 data")

                return iterator, inference_mode_var, train_data_size, eval_data_size, test_data_size


class TolstoiLoader(AbstractDatasetLoader):
    @staticmethod
    def get_iterator(sess, dataset_path="../Datasets/tolstoi.npy", train_data_size=40000, batch_size=100, num_gpus=1):
        """
         creates an iterator that iterates over cifar images and the corresponding labels.
         Data Augementation and shuffling is done. The iterator repeats forever.
        :param sess:
        :param dataset_path: path to a numpy file containing the cifar dataset.
        :param train_data_size: eval data size will be determined out of this.
        :param batch_size: must be multiple of num_gpus
        _param num_gpus
        Returns:
         iterator, inference_mode_var (must hold an element of type :class:`~code_.abstract_net_class.inferenceMode` )
         ,train_data_size, eval_data_Size, test_data_size
        """

        if batch_size % num_gpus != 0:
            raise ValueError("batch_size of {0:d} not multiple of num_gpu {1:d}".format(batch_size, num_gpus))

        div_batch_size = batch_size // num_gpus

        seq_length = 50  # (as in deepobs)

        train_data, test_data = np.load(dataset_path, allow_pickle=True)

        print("loaded data from " + dataset_path)

        num_batches_train = int(
            np.floor(
                (np.size(train_data) - 1) / (div_batch_size * seq_length)))
        if num_batches_train == 0:
            raise ValueError(
                "This dataset is to small to use with this batch size "
                "and sequence length.")

        num_batches_test = int(
            np.floor(
                (np.size(test_data) - 1) / (div_batch_size * seq_length)))
        if num_batches_test == 0:
            raise ValueError(
                "This dataset is to small to use with this batch size "
                "and sequence length.")

        x_train = train_data[:num_batches_train * batch_size * seq_length]
        y_train = train_data[1:num_batches_train * batch_size * seq_length + 1]

        x_test = test_data[:num_batches_test * batch_size * seq_length]
        y_test = test_data[1:num_batches_test * batch_size * seq_length + 1]

        X_train = x_train.reshape(-1, seq_length)
        Y_train = y_train.reshape(-1, seq_length)

        X_test = x_test.reshape(-1, seq_length)
        Y_test = y_test.reshape(-1, seq_length)
        train_eval_size = 653237

        train_data_size = len(X_train) - train_eval_size // seq_length

        X_eval = X_train[train_data_size:, :]
        X_train = X_train[0:train_data_size, :]
        Y_eval = Y_train[train_data_size:, :]
        Y_train = Y_train[0:train_data_size, :]

        num_aug_threads = 8
        prefetch = 8

        def mapf_cast(x, y):
            return tf.cast(x, tf.int32), tf.cast(y, tf.int32)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        test_dataset = test_dataset.map(mapf_cast, num_parallel_calls=num_aug_threads)
        test_dataset = test_dataset.repeat()
        test_dataset = test_dataset.batch(div_batch_size)
        test_dataset = test_dataset.prefetch(prefetch)

        eval_dataset = tf.data.Dataset.from_tensor_slices((X_eval, Y_eval))
        eval_dataset = eval_dataset.map(mapf_cast, num_parallel_calls=num_aug_threads)
        eval_dataset = eval_dataset.repeat()
        eval_dataset = eval_dataset.batch(div_batch_size)
        eval_dataset = eval_dataset.prefetch(prefetch)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.map(mapf_cast, num_parallel_calls=num_aug_threads)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.shuffle(buffer_size=train_data_size // 10)
        train_dataset = train_dataset.batch(div_batch_size)
        train_dataset = train_dataset.prefetch(prefetch)

        iterator, inference_mode_var = super(TolstoiLoader, TolstoiLoader)._create_handleiterator(train_dataset,
                                                                                                  eval_dataset,
                                                                                                  test_dataset, sess)

        u.initialize_vars(sess)
        print("successfully loaded Tolstoi War and Peace data")

        return iterator, inference_mode_var, train_data_size, np.shape(X_eval)[0], np.shape(X_test)[0]


class MNISTLoader(AbstractDatasetLoader):
    @staticmethod
    def get_iterator(sess, dataset_path="NOT USED", train_data_size=40000, batch_size=100, num_gpus=1):
        # TODO add augmenter field
        """
         creates iterator that iterates over MNIST images and  corresponding labels.
         No shuffling is done. The iterator repeats forever.

        :param sess:
        :param dataset_path: path to a numpy file containing the cifar dataset.
        :param train_data_size: eval data size will be determined out of this.
        :param batch_size: must be multiple of num_gpus
        _param num_gpus
        Returns:
         iterator, inference_mode_var (must hold element of type :class:`~code_.abstract_net_class.inferenceMode` ),
          train_data_size, eval_data_Size, test_data_size
        """

        if batch_size % num_gpus != 0:
            raise ValueError("batsite of {0:d} not multiple of num_gpu {1:d}".format(batch_size, num_gpus))

        div_batch_size = batch_size // num_gpus

        with tf.variable_scope("Iterator", reuse=tf.AUTO_REUSE):
            with tf.device("/cpu:0"):
                from keras.datasets import mnist
                (x_train, y_train), (x_test, y_test) = mnist.load_data()

                x_eval = x_train[train_data_size:, ...]
                x_train = x_train[0:train_data_size, ...]
                y_eval = y_train[train_data_size:, ...]
                y_train = y_train[0:train_data_size, ...]

                eval_data_size = len(y_eval)
                test_data_size = len(y_test)

                x_train = x_train.reshape(train_data_size, 28, 28, 1).astype('uint8')
                x_eval = x_eval.reshape(eval_data_size, 28, 28, 1).astype('uint8')
                x_test = x_test.reshape(test_data_size, 28, 28, 1).astype('uint8')
                y_train = y_train.reshape(train_data_size, 1)
                y_eval = y_eval.reshape(eval_data_size, 1)
                y_test = y_test.reshape(test_data_size, 1)

                def mapf(x, y):
                    return tf.cast(x, tf.float32), y

                num_aug_threads = 8

                test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
                test_dataset = test_dataset.map(mapf, num_parallel_calls=num_aug_threads)
                test_dataset = test_dataset.prefetch(buffer_size=test_data_size // 4)
                test_dataset = test_dataset.repeat()
                test_dataset = test_dataset.batch(div_batch_size)

                eval_dataset = tf.data.Dataset.from_tensor_slices((x_eval, y_eval))
                eval_dataset = eval_dataset.map(mapf, num_parallel_calls=num_aug_threads)
                eval_dataset = eval_dataset.prefetch(buffer_size=eval_data_size // 4)
                eval_dataset = eval_dataset.repeat()
                eval_dataset = eval_dataset.batch(div_batch_size)

                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                train_dataset = train_dataset.prefetch(buffer_size=train_data_size // 4)
                train_dataset = train_dataset.map(mapf, num_parallel_calls=num_aug_threads)
                train_dataset = train_dataset.repeat()
                train_dataset = train_dataset.batch(div_batch_size)
                iterator, inference_mode_var = super(MNISTLoader, MNISTLoader)._create_handleiterator(train_dataset,
                                                                                                      eval_dataset,
                                                                                                      test_dataset,
                                                                                                      sess)
                print("sucessfully loaded MNIST data")

                return iterator, inference_mode_var, train_data_size, eval_data_size, test_data_size
