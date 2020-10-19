__author__ = "Maximus Mutschler, Kevin Laube"
__version__ = "1.1"
__email__ = "maximus.mutschler@uni-tuebingen.de"

"""
Implementation adapted from https://github.com/kuangliu/pytorch-cifar
"""
import argparse
import logging
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from PyTorch.pal_optimizer import PalOptimizer
from PyTorch.networks import ResNet18
from PyTorch.networks import ResNet34
from PyTorch.networks import DenseNet_Cifar

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

parser = argparse.ArgumentParser(description='PAL Pytorch CIFAR10 Training')
parser.add_argument('--mu', default=1.0, type=float, help='sample distance, multiple of norm gradient')
parser.add_argument('--mss', default=3.6, type=float, help='max step size')
parser.add_argument('--direction_adaptation_factor', default=0.4, type=float, help='direction_adaptation_factor')
parser.add_argument('--update_step_adaptation', default=1, type=float, help='update_step_adaptation')
parser.add_argument('--is_plot', default=False, type=bool, help='visualize lines in negative line direction,'
                                                                'the coresponding parabolic approximation and the update step')
parser.add_argument('--model', type=str, default='resnet34')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--batch_size_test', type=int, default=128)
parser.add_argument('--data_dir', type=str, default='~/Data/Datasets/cifar10_data/')
parser.add_argument('--cp_dir', type=str, default='/tmp/pt.lineopt/')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
update_bar_prints_train, update_bar_prints_test = 1, 1

# Data
logger.info('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
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

# Model
logger.info('==> Building model..')
net = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'densenet': DenseNet_Cifar,
}.get(args.model.lower())()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = PalOptimizer(net.parameters(), writer, measuring_step_size=args.mu, max_step_size=args.mss,
                         update_step_adaptation=args.update_step_adaptation,
                         direction_adaptation_factor=args.direction_adaptation_factor, is_plot=args.is_plot,
                         plot_step_interval=100, save_dir="./lines/")
time_start = time.time()


def formatted_str(prefix, epoch_, loss, accuracy):
    return '{p} Epoch: {e:3}, Loss: {l:0.4E}, {p}Acc.: {a:5.2f}%'.format(
        **{
            'p': prefix,
            'e': epoch_,
            'l': loss,
            'a': accuracy * 100,
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

        if batch_idx == 0 and epoch_ == 0:
            intitial_loss = criterion(net(inputs), targets)
            print("Initial Loss ", intitial_loss)

        def loss_fn(backward=True):
            out_ = net(inputs)
            loss_ = criterion(out_, targets)
            if backward:
                loss_.backward()
            return loss_, out_

        loss, outputs, _ = optimizer.step(loss_fn)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    cur_time = int((time.time() - time_start))
    logger.debug('train time: {:4.2f} min'.format(cur_time / 60))
    logger.info(formatted_str('TRAIN:', epoch_, train_loss / (batch_idx + 1), correct / total))

    for s, t in [('time', cur_time), ('epoch', epoch_)]:
        writer.add_scalar('train-%s/accuracy' % s, correct / total, t)
        writer.add_scalar('train-%s/train_loss' % s, train_loss / (batch_idx + 1), t)


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

    logger.info(formatted_str('TEST:', epoch_, loss, correct / total))

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


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(epoch)
        test(epoch)
