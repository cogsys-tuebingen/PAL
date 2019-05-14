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

import models.resnet as resnet
import models.preact_resnet as presnet
from PyTorch.pal_optimizer import PalOptimizer
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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--mu', default=0.1, type=float, help='sample distance, multiple of norm gradient')
parser.add_argument('--mss', default=1.0, type=float, help='max step size')
parser.add_argument('--momentum', default=0.6, type=float, help='lambda, momentum')
parser.add_argument('--lambda_', default=0.6, type=float, help='lambda, loose approximation')
parser.add_argument('--decay_rate', default=0.95, type=float, help='decay rate for exponential decay')
parser.add_argument('--decay_steps', default=450, type=float, help='decay steps for exponential decay')
parser.add_argument('--is_plot', default=False, type=bool, help='visualize lines in negative gradient direction,'
                                                        'the coresponding parabolic approximation and update step')
parser.add_argument('--save', '-s', action='store_true', help='save to checkpoint')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--log_detailed', '-d', action='store_true', help='detailed TB logs on a per-batch base',
                    default=True)
parser.add_argument('--test', '-t', action='store_true', help='check the test error after every epoch')
parser.add_argument('--data_dir', type=str, default='~/Data/Datasets/cifar10_data/')
parser.add_argument('--cp_dir', type=str, default='/tmp/pt.lineopt/')
parser.add_argument('--model', type=str, default='resnet34')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=2, help='num workers for data loading/augmenting')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--batch_size_test', type=int, default=100)
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
                                           shuffle=True, num_workers=args.num_workers)
steps_per_train_epoch = len(train_loader.dataset) / train_loader.batch_size


test_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size_test,
                                          shuffle=False, num_workers=args.num_workers)

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
    'preactresnet18': presnet.PreActResNet18,
    'resnet34': resnet.ResNet34,
}.get(args.model.lower())()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    logger.info('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
w = writer if args.log_detailed else None
optimizer = PalOptimizer(net.parameters(), w, mu=args.mu, s_max=args.mss,
                         lambda_=args.lambda_, mom=args.momentum, is_plot=args.is_plot, plot_step_interval=200)
time_start = time.time()


def formatted_str(prefix, epoch_, batch_idx, batch_count, loss, correct, total):
    return '{p} Epoch: {e:3}, Batch: {b:5}/{tb:5}, Loss: {l:0.4E}, Acc: {a:5.2f}%, Correct: {c:5}, Total: {t:5}'.format(
        **{
            'p': prefix,
            'e': epoch_,
            'b': batch_idx,
            'tb': batch_count,
            'l': loss,
            'a': 100. * correct / total,
            'c': correct,
            't': total
        })


# Training
def train(epoch_):
    logger.info('\nEpoch: %d' % epoch_)
    logger.info(optimizer)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        def loss_fn(backward=True):
            out_ = net(inputs)
            loss_ = criterion(out_, targets)
            if backward:
                loss_.backward()
            return loss_, out_

        # decay

        d_mu = args.mu * args.decay_rate ** (((epoch_ + 1) * steps_per_train_epoch + batch_idx) / args.decay_steps)
        d_mss = args.mss * args.decay_rate ** (((epoch_ + 1) * steps_per_train_epoch + batch_idx) / args.decay_steps)

        for param_group in optimizer.param_groups:
            param_group['mu'] = d_mu
            param_group['mss'] = d_mss

        loss, outputs = optimizer.step(loss_fn)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % (len(train_loader) // update_bar_prints_train) == 0 or (batch_idx + 1) == len(
                train_loader):
            logger.info(formatted_str('TRAIN:', epoch_, batch_idx, len(train_loader), loss, correct, total))

    # log some info, via epoch and time[ms]
    cur_time = int((time.time() - time_start))
    logger.debug('train time: {:4.2f} min'.format(cur_time / 60))
    for s, t in [('time', cur_time), ('epoch', epoch_)]:
        logger.debug('writing training train-%s with %s' % (s, writer))
        writer.add_scalar('train-%s/accuracy' % s, correct / total, t)
        writer.add_scalar('train-%s/loss_sum' % s, train_loss, t)


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

            if (batch_idx + 1) % (len(test_loader) // update_bar_prints_test) == 0 or (batch_idx + 1) == len(
                    test_loader):
                logger.info(formatted_str(' TEST:', epoch_, batch_idx, len(test_loader), loss, correct, total))

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
    test(start_epoch + args.epochs+1)
