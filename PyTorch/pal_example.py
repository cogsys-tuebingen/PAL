__author__ = "Maximus Mutschler, Kevin Laube"
__version__ = "1.0"
__email__ = "maximus.mutschler@uni-tuebingen.de"

"""
Implementation adapted from https://github.com/kuangliu/pytorch-cifar
"""
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter


from PyTorch.pal_optimizer import PalOptimizer
from PyTorch.resnet import ResNet34
from PyTorch.resnet import ResNet18
import sys
import time
import os
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

parser = argparse.ArgumentParser(description='PAL Pytorch CIFAR10 Training')
parser.add_argument('--mu', default=1.0, type=float, help='sample distance, multiple of norm gradient')
parser.add_argument('--mss', default=10.0, type=float, help='max step size')
parser.add_argument('--conjugate_gradient_factor', default=0.4, type=float, help='conjugate_gradient_factor')
parser.add_argument('--update_step_adaptation', default=1, type=float, help='update_step_adaptation')
#parser.add_argument('--decay_rate', default=0.95, type=float, help='decay rate for exponential decay')
#parser.add_argument('--decay_steps', default=450, type=float, help='decay steps for exponential decay')
parser.add_argument('--is_plot', default=False, type=bool, help='visualize lines in negative line direction,'
                                                        'the coresponding parabolic approximation and update step')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--batch_size_test', type=int, default=100)
parser.add_argument('--data_dir', type=str, default='~/Data/Datasets/cifar10_data/')
parser.add_argument('--cp_dir', type=str, default='/tmp/pt.lineopt/')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
update_bar_prints_train, update_bar_prints_test = 1, 1

# Data
logger.info('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                           shuffle=True, num_workers=2)
steps_per_train_epoch = len(train_loader.dataset) / train_loader.batch_size


test_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size_test,
                                          shuffle=False, num_workers=2)

# Tensorboard, logging
tb_dir = args.cp_dir + 'tb/'
writer = SummaryWriter(log_dir=tb_dir)

logger.info('\n--- Args: ---')
for k, v in vars(args).items():
    k, v = str(k), str(v)
    writer.add_text('args', '%s: %s' % (k, v))
    logger.info('%s: %s' % (k, v))
writer.add_text('device', device)
logger.info('device: %s\n' % device)

# Seed
torch.manual_seed(1)

# Model
logger.info('==> Building model..')
net = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
}.get(args.model.lower())()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = PalOptimizer(net.parameters(), writer, measuring_step_size=args.mu, max_step_size=args.mss,
                         update_step_adaptation=args.update_step_adaption, conjugate_gradient_factor=args.conjugate_gradient_factor, is_plot=args.is_plot, plot_step_interval=100, save_dir="lines/")
time_start = time.time()


def formatted_str(prefix, epoch_, loss, accuracy):
    return '{p} Epoch: {e:3}, Loss: {l:0.4E}, {p}Acc.: {a:5.2f}%'.format(
        **{
            'p': prefix,
            'e': epoch_,
            'l': loss,
            'a': accuracy*100,
        })


# Training
def train(epoch_):
    net.train()
    train_loss = 0
    correct = 0
    total = 0


    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        #if batch_idx==1:
            #intitial_loss = criterion(net(inputs), targets)
            #print("Initial Loss ", intitial_loss)

        def loss_fn(backward=True):
            out_ = net(inputs)
            loss_ = criterion(out_, targets)
            if backward:
                loss_.backward()
            return loss_, out_

        # decay if wanted:

        # d_mu = args.mu * args.decay_rate ** (((epoch_ + 1) * steps_per_train_epoch + batch_idx) / args.decay_steps)
        # d_mss = args.mss * args.decay_rate ** (((epoch_ + 1) * steps_per_train_epoch + batch_idx) / args.decay_steps)
        #
        # for param_group in optimizer.param_groups:
        #     param_group['mu'] = d_mu
        #     param_group['mss'] = d_mss

        loss, outputs = optimizer.step(loss_fn)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    cur_time = int((time.time() - time_start))
    logger.debug('train time: {:4.2f} min'.format(cur_time / 60))
    logger.info(formatted_str('TRAIN:', epoch_, train_loss/(batch_idx+1), correct/total))
    # log some info, via epoch and time[ms]

    for s, t in [('time', cur_time), ('epoch', epoch_)]:
        writer.add_scalar('train-%s/accuracy' % s, correct / total, t)
        writer.add_scalar('train-%s/train_loss' % s, train_loss/(batch_idx+1), t)


def test(epoch_):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    logger.info(formatted_str('TEST:', epoch_, loss, correct/total))

    # log some info, via epoch and time[ms]
    cur_time = int((time.time() - time_start) * 1000)
    for s, t in [('time', cur_time), ('epoch', epoch_)]:
        writer.add_scalar('test-%s/accuracy' % s, correct / total, t)
        writer.add_scalar('test-%s/loss_sum' % s, test_loss, t)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        os.makedirs(args.cp_dir, exist_ok=True)
        best_acc = acc
        if args.save:
            logger.info('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch_,
            }
            torch.save(state, args.cp_dir + 'ckpt.t7')


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(epoch)
        test(epoch)
