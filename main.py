import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import sys
import time
import math
import os
import argparse

#Logging Of Model
import wandb
wandb.init(project="game-theorectic-pruning")

#models
import efficientnet
from resnet import ResNet50
from vgg import VGG
from lenet import LeNet
from dpn import DPN
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

""" Prune stuff
new_model = LeNet()
for name, module in new_model.named_modules():
    # prune 20% of connections in all 2D-conv layers 
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.2)
    # prune 40% of connections in all linear layers 
    elif isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.4)
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)
class FooBarPruningMethod(prune.BasePruningMethod):


    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0 
        return mask
def foobar_unstructured(module, name):

    FooBarPruningMethod.apply(module, name)
    return module
foobar_unstructured(model.fc3, name='bias')



"""
def load_cifar_10(target_batch_size, target_num_workers): 
    print('Loading CIFAR 10 Data')
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2675, 0.2565, 0.2761)),])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=target_batch_size, shuffle=True, num_workers=target_num_workers)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=target_num_workers)
    return trainloader, testloader

def load_cifar_100(target_batch_size, target_num_workers): 
    print('Loading CIFAR 100 Data')
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=target_batch_size, shuffle=True, num_workers=target_num_workers)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=target_num_workers)
    return trainloader, testloader

def train(model, epoch, trainloader, device, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(model, epoch, testloader, device, criterion, args, best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % args.log_interval == 0:
                wandb.log({"Test Accuracy": correct / total, "Test Loss": loss})
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, args.save_name)
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, args.save_name))

def run(args):
    if args.arch == 'VGG':
        model = VGG('VGG16')
    elif args.arch == 'LENET':
        model = LeNet()
    elif args.arch == 'RESNET':
        model = ResNet50()
    elif args.arch == 'DPN':
        model = DPN92()
    elif args.arch == 'EFFICIENTNET':
        model = EfficientNet()
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    model = model.to(device)
    wandb.watch(model)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    best_acc = 0
    start_epoch = 0
    trainloader, testloader = load_cifar_10(args.batch_size, args.workers)
    if args.load_name != None:
        print("Loading previous model:{}".format(args.save_name))
        checkpoint = torch.load(args.save_name)
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(start_epoch, args.epochs):
        train(model, epoch, trainloader, device, optimizer, criterion)
        test(model, epoch, testloader, device, criterion, args, best_acc)
        
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments in Using Game Theory For Model Pruning')
    parser.add_argument('--log_interval', default=100)
    parser.add_argument('--arch', default='RESNET', choices=['VGG','LENET','RESNET','DPN'],help='model architectures: VGG16, LENET, RESNET, DPN92')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int,  help='mini-batch size (default: 128)')
    parser.add_argument('--learning_rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='M',help='Weight Decay')
    parser.add_argument('--train', default=True, help='Train network from scratch')
    parser.add_argument('--eval', default=True, help='Evaluate network')
    parser.add_argument('--prune', default=True, help='Prune Network')
    parser.add_argument('--prune_method', default='GMP')
    parser.add_argument('--load_name', default=None, metavar='PATH',help='path to resume training from')
    parser.add_argument('--cuda', default=True, help='Use CPU or CUDA GPU' )
    parser.add_argument('--save_name', help='The path used to save the trained models',default='model', type=str)
    run(parser.parse_args())
