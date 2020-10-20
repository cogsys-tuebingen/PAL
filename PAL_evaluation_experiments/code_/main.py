import argparse
import sys

from code_.networks.tolstoi_net import tolstoi_rnn
from code_.networks.efficient_net import efficient_net_cifar10
from code_.networks.efficient_net import efficient_net_cifar100
from code_.networks.efficient_net import efficient_net_IM
from code_.networks.mobile_netv2 import mobile_net_v2_cifar10
from code_.networks.mobile_netv2 import mobile_net_v2_cifar100
from code_.networks.mobile_netv2 import mobile_net_v2_IM
from code_.networks.res_net import resnet_32_cifar10
from code_.networks.res_net import resnet_34_IN_style_cifar
from code_.networks.res_net import resnet_32_cifar100
from code_.networks.res_net import resnet_101_IM
from code_.networks.res_net import resnet_50_IM
from code_.networks.dense_net import dense_net_cifar10
from code_.networks.dense_net import dense_net_cifar100
from code_.networks.dense_net import dense_net_IM
from code_.networks.mnist_simple import simple_mnist_net

from code_ import net_frame
from code_ import net_frame_ol

from code_.optimizers.optimal_line_search import OptimalLineSearch
from code_.optimizers.optimizers import TfOptimizer
from code_.optimizers.PAL import PAL
from code_.optimizers.SLS import SLS
from code_.optimizers.SGDHD import SGDHD
from code_.optimizers.cocob import COCOB
from code_.optimizers.alig import AliGwithMomentum

from code_ import framework_utils as fu
from code_.dataset_loader import *
from tensorflow.python.client import device_lib


def main():
    """
    Entry point to the experiment. Parameters are read and applied to the network model.
    The network model to use has to uncommented in this file (see below).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',
                        type=str,
                        default='../Datasets2/cifar100Dataset.npy',
                        help='location of the dataset in numpy format', )
    parser.add_argument('--train_steps',
                        type=int,
                        default=150000,
                        help='training steps', )
    parser.add_argument('--measuring_step_size',
                        type=float,
                        default=0.1,
                        help='step size to where a second loss is determined to approximate the loss function '
                             'in the direction of the gradient by a parabola', )
    parser.add_argument('--momentum',
                        type=float,
                        default=0.4,
                        help='momentum term', )
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='batch_size', )
    parser.add_argument('--experiment_name',
                        type=str,
                        default="testmodel",
                        help='the name of the experiment', )
    parser.add_argument('--loose_approximation_factor',
                        type=float,
                        default=1.0,
                        help='intentionally approximate the function with less or more curvature. = 1/ step size adaptation '
                             'less curvature <1 more curvature >1', )
    parser.add_argument('--train_data_size',
                        type=int,
                        default=45000,
                        help='train data size,remaining elements define the evaluation_Res_Net set', )
    parser.add_argument('--random_seed',
                        type=int,
                        default=1,
                        help='random number seed for numpy and tensorflow to get same results for multiple runs', )
    parser.add_argument('--max_stepsize',
                        type=float,
                        default=3.6,
                        help='max stepsize in direction of the gradient', )
    parser.add_argument('--decay',
                        type=float,
                        default=1,
                        help='max stepsize and measurment stepsize decay rate', )
    parser.add_argument('--additional',
                        type=float,
                        default=100,
                        help='additional parameter', )
    parser.add_argument('--num_gpus',
                        type=int,
                        default=1,
                        help='num gpus to train on', )
    parser.add_argument('--optimizer',
                        type=str,
                        default="SLS",
                        help='the optimizer to use', )

    FLAGS, unparsed = parser.parse_known_args()
    for k, v in vars(FLAGS).items():
        k, v = str(k), str(v)
        print('%s: %s' % (k, v))
    FLAGS.dataset_path = os.path.expanduser(FLAGS.dataset_path)
    print("DatasetPath: " + str(FLAGS.dataset_path))

    workpath = os.path.dirname(os.path.dirname(sys.argv[0])) + '/'  # double dir name to get parent

    print("workpath: " + workpath)

    # check gpus
    local_device_protos = device_lib.list_local_devices()
    num_available_gpus = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
    assert num_available_gpus >= FLAGS.num_gpus
    print("GPUs available: {1:d}  \t GPUs used: {1:d}".format(num_available_gpus, FLAGS.num_gpus))

    learning_rate_pf = lambda global_step, learning_rate: tf.train.piecewise_constant(global_step,
                                                                                      [75000.0, 112500.0],
                                                                                      [float(learning_rate),
                                                                                       float(learning_rate / 10),
                                                                                       float(learning_rate / 100),
                                                                                       ])

    if FLAGS.optimizer == "PAL":
        optimizer = PAL(None, FLAGS.measuring_step_size, FLAGS.momentum, FLAGS.loose_approximation_factor,
                        FLAGS.max_stepsize, False)
    elif FLAGS.optimizer == "SLS":
        optimizer = SLS(n_batches_per_epoch=FLAGS.train_data_size // FLAGS.batch_size,
                        init_step_size=FLAGS.measuring_step_size, c=FLAGS.momentum,
                        beta_b=FLAGS.loose_approximation_factor, gamma=FLAGS.max_stepsize)
    elif FLAGS.optimizer == "OL":
        #optimizer = OptimalLineSearch(initial_search_step=FLAGS.measuring_step_size,
        #                                max_num_of_steps=FLAGS.max_stepsize, momentum=FLAGS.momentum)
        optimizer = OptimalLineSearch(initial_search_step=1.0,
                                        max_num_of_steps=20.0, momentum=0.0)

    elif FLAGS.optimizer == "RMSP":
        optimizer = TfOptimizer(tf.train.RMSPropOptimizer, learning_rate_pf,
                                {"learning_rate": FLAGS.measuring_step_size, "decay": FLAGS.momentum,
                                 "epsilon": FLAGS.loose_approximation_factor})
    elif FLAGS.optimizer == "ADAM":
        optimizer = TfOptimizer(tf.train.AdamOptimizer, learning_rate_pf,
                                {"learning_rate": FLAGS.measuring_step_size, "beta1": FLAGS.momentum,
                                 "beta2": FLAGS.loose_approximation_factor, "epsilon": FLAGS.max_stepsize})
    elif FLAGS.optimizer == "SGD":
        optimizer = TfOptimizer(tf.train.MomentumOptimizer, learning_rate_pf,
                                {"learning_rate": FLAGS.measuring_step_size, "momentum": FLAGS.momentum,
                                 "use_nesterov": True})
    elif FLAGS.optimizer == "SGDHD":
        optimizer = TfOptimizer(SGDHD, None,
                                {"learning_rate": FLAGS.measuring_step_size, "hyper_gradient_learning_rate": FLAGS.momentum})
    elif FLAGS.optimizer == "ALIG":
        optimizer = TfOptimizer(AliGwithMomentum, None,
                                {"max_lr": FLAGS.measuring_step_size, "momentum": FLAGS.momentum})
    elif FLAGS.optimizer == "COCOB":
        optimizer = TfOptimizer(COCOB, None,
                                {"alpha": FLAGS.measuring_step_size})

    else:
        raise ValueError("unknown optimizer flag:" + FLAGS.optimizer)

    # Uncomment the network and dataset to use!

    # net_type= tolstoi_rnn.TolstoiRNN

    # net_type= simple_mnist_net.SimpleMnistNet

    # net_type=efficient_net_cifar10.EfficientNet
    # net_type=mobile_net_v2_cifar10.MobileNetV2
    net_type = resnet_32_cifar10.ResNet
    # net_type=dense_net_cifar10.DenseNet
    # net_type=resnet_34_IN_style_cifar.ResNet

    # net_type=efficient_net_cifar100.EfficientNet
    # net_type=mobile_net_v2_cifar100.MobileNetV2
    # net_type=resnet_32_cifar100.ResNet
    # net_type=dense_net_cifar100.DenseNet

    # net_type=efficient_net_IM.EfficientNet
    # net_type=mobile_net_v2_IM.MobileNetV2
    # net_type=resnet_101_IM.ResNet
    # net_type=resnet_50_IM.ResNet
    # net_type=dense_net_IM.DenseNet

    # data_set_loader = ImageNetLoader
    #data_set_loader = Cifar10Loader  # also uncomment is_augment
    #data_set_loader.is_augment = True
    data_set_loader = Cifar100Loader
    # data_set_loader = TolstoiLoader
    # data_set_loader= MNISTLoader

    sys.stdout.flush()

    if FLAGS.optimizer == "OL":
        net = net_frame_ol.NetFrame(net_type, data_set_loader, optimizer, FLAGS.num_gpus, FLAGS.random_seed,
                                         FLAGS.train_data_size,
                                         FLAGS.batch_size, FLAGS.dataset_path, workpath, FLAGS.experiment_name,
                                         is_calc_angle=False)
    else:
        net = net_frame.NetFrame(net_type, data_set_loader, optimizer, FLAGS.num_gpus, FLAGS.random_seed,
                                      FLAGS.train_data_size,
                                      FLAGS.batch_size, FLAGS.dataset_path, workpath, FLAGS.experiment_name,
                                      is_calc_angle=False)  # 100. 0.001  # problem 1,1  or 20,1 -> very steep descent!

    is_failed = False
    try:
        mean_train_losses_per_interval, evaluation_accuracies, train_losses_for_each_step, step_sizes_for_each_step, \
        angles_for_each_step, grad_norms_for_each_step, train_time_for_each_step, tran_acc_per_interval, \
        eval_losses, avg_test_acc, avg_test_loss, all_first_derivatives, all_second_derivatives \
            = net.train(FLAGS.train_steps)
    except Exception as e:
        print(e.__doc__)
        is_failed = True
        print("FAILED")

    if is_failed:
        eval_data_wrapper = fu.EvalDataWrapper(FLAGS.experiment_name, FLAGS.random_seed, FLAGS.optimizer,
                                               FLAGS.train_data_size,
                                               FLAGS.train_steps, FLAGS.batch_size, FLAGS.measuring_step_size,
                                               FLAGS.momentum,
                                               FLAGS.loose_approximation_factor, FLAGS.max_stepsize, FLAGS.decay,
                                               FLAGS.additional, [], [], [], [], None, None, is_failed)
    else:
        eval_data_wrapper = fu.EvalDataWrapper(FLAGS.experiment_name, FLAGS.random_seed, FLAGS.optimizer,
                                               FLAGS.train_data_size,
                                               FLAGS.train_steps, FLAGS.batch_size, FLAGS.measuring_step_size,
                                               FLAGS.momentum,
                                               FLAGS.loose_approximation_factor, FLAGS.max_stepsize, FLAGS.decay,
                                               FLAGS.additional, mean_train_losses_per_interval, tran_acc_per_interval,
                                               evaluation_accuracies, eval_losses, avg_test_acc, avg_test_loss, is_failed,
                                               angles_for_each_step, step_sizes_for_each_step, grad_norms_for_each_step,
                                               all_first_derivatives, all_second_derivatives
                                               )

    fu.save_eval_data_wrapper(eval_data_wrapper, net.model_dir)

if __name__ == "__main__":
    main()
