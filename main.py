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
import numpy as np
import random
from collections import OrderedDict

#Logging Of Model
import wandb
#wandb.init(project="game-theorectic-pruning")

#models
from resnet import ResNet50
from vgg import VGG16
from dpn import DPN92
    
def print_sparse_model_params(model):
    params = 0
    #for module in model.module.features: # for VGG
    for module in model.module.children():
        for submodule in module.children():
            for subsubmodule in submodule.children():
                if isinstance(subsubmodule, nn.Conv2d):
                    params +=  float(subsubmodule.weight.nelement())
            """for p in module.parameters():
                if p.requires_grad:
                    params += float(p.data.nelement())   * (1-get_param_sparcity(p))"""
    print('  + Number of params: %.2fM' % (params / 1e6)) 
    exit(-1)
    
def get_module_weight_sparcity(module):
    return 100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())
def get_param_sparcity(p):
    return 100. * float(torch.sum(p.data == 0)) / float(p.data.nelement())  
# Data Loaders
def load_imagenet(target_batch_size, target_num_workers):
    print("Loading imagenet Data")
    trainset = torchvision.datasets.ImageNet(root='./data',split='train',download=True,transform = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=target_batch_size, shuffle=True, num_workers=target_num_workers)    
    testset = torchvision.datasets.ImageNet(root='./data',split='val',download=True,transform = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=target_num_workers)
    return trainloader, testloader

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
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=target_num_workers)
    return trainloader, testloader

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

def test(model, epoch, testloader, device, criterion, args, best_acc, prune_flag=False):
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
                pass #wandb.log({"Test Accuracy": correct / total, "Test Loss": loss})
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    if acc > best_acc and prune_flag == False: 
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, args.save_name)
        best_acc = acc
        #torch.save(model.state_dict(), os.path.join(wandb.run.dir, args.save_name))
    if prune_flag == True:
        return acc
    return best_acc

def L1RandomPrune(module, prune_percentage):
    if random.randint(0, 1)== 0:
        prune.random_unstructured(module, name='weight', amount = prune_percentage/2)
        prune.l1_unstructured(module, name='weight', amount = prune_percentage/2)
    else:
        prune.l1_unstructured(module, name='weight', amount = prune_percentage/2)
        prune.random_unstructured(module, name='weight', amount = prune_percentage/2)

def PositiveMagnitudePrune(module,prune_percentage):
    prune_percentage = max(1/module.weight.nelement(),prune_percentage)
    PositiveMagnitude.apply(module, name='weight', percentage=prune_percentage)

def NegativeMagnitudePrune(module,prune_percentage):
    prune_percentage = max(1/module.weight.nelement(),prune_percentage)
    NegativeMagnitude.apply(module, name='weight', percentage=prune_percentage)

def RandomPrune(module, prune_percentage):
    prune_percentage = max(1/module.weight.nelement(),prune_percentage)
    prune.random_unstructured(module, name='weight', amount = prune_percentage)

def L1Prune(module, prune_percentage):
    prune_percentage = max(1/module.weight.nelement(),prune_percentage)
    prune.l1_unstructured(module, name='weight', amount = prune_percentage)

def MagnitudeL1Prune(module,prune_percentage):
    if random.randint(0, 1) == 0:
        prune.l1_unstructured(module, name='weight', amount = prune_percentage/2)
        if random.randint(0, 1) == 0:
            PositiveMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
            NegativeMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
        else:
            NegativeMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
            PositiveMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
    else:
        if random.randint(0, 1) == 0:
            NegativeMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
            PositiveMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
            prune.l1_unstructured(module, name='weight', amount = prune_percentage/2)
        else:
            PositiveMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
            NegativeMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
            prune.l1_unstructured(module, name='weight', amount = prune_percentage/2)

def MagnitudeRandomPrune(module,prune_percentage):
    if random.randint(0, 1) == 0:
        prune.random_unstructured(module, name='weight', amount = prune_percentage/3)
        if random.randint(0, 1) == 0:
            PositiveMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
            NegativeMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
        else:
            NegativeMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
            PositiveMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
    else:
        if random.randint(0, 1) == 0:
            NegativeMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
            PositiveMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
            prune.random_unstructured(module, name='weight', amount = prune_percentage/3)
        else:
            PositiveMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
            NegativeMagnitude.apply(module, name='weight', percentage=prune_percentage/3)
            prune.random_unstructured(module, name='weight', amount = prune_percentage/3) 

def MagnitudePrune(module,prune_percentage):
    if random.randint(0, 1) == 0:
        PositiveMagnitude.apply(module, name='weight', percentage=prune_percentage/2)
        NegativeMagnitude.apply(module, name='weight', percentage=prune_percentage/2)
    else:
        NegativeMagnitude.apply(module, name='weight', percentage=prune_percentage/2)
        PositiveMagnitude.apply(module, name='weight', percentage=prune_percentage/2)

class PositiveMagnitude(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    def __init__(self, percentage):
        super(PositiveMagnitude, self).__init__()
        self.percentage = percentage
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        total_weights = len(t.flatten())
        b = t.flatten()
        m_sort, idx_sort = b.sort()
        target_idx = 0
        for i in range(1,total_weights):
            if m_sort[i].item() > 0 and m_sort[i-1].item() <= 0:
                target_idx = i
                break
        for i in range(max(1,int(self.percentage*total_weights))):
            mask.view(-1)[idx_sort[target_idx + i]] = 0
        return mask

class NegativeMagnitude(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    def __init__(self, percentage):
        super(NegativeMagnitude, self).__init__()
        self.percentage = percentage
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        total_weights = len(t.flatten())
        b = t.flatten()
        m_sort, idx_sort = b.sort()
        target_idx = 0
        for i in range(total_weights-1,0):
            if m_sort[i].item() < 0 and m_sort[i-1].item() >= 0:
                target_idx = i
                break
        for i in range(max(1,int(self.percentage*total_weights))):
            mask.view(-1)[idx_sort[target_idx - i]] = 0
        return mask

def get_module_weight_sparcity(module):
    return 100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())

def main(args):
    if args.dataset == 'cifar10':
        trainloader, testloader = load_cifar_10(args.batch_size, args.workers)
        num_classes = 10
    elif args.dataset == 'cifar100':
        trainloader, testloader = load_cifar_100(args.batch_size, args.workers)
        num_classes = 100
    elif args.dataset == 'imagenet':
        print("NOT YET IMPLEMENTED! Give me a sec")
        trainloader, testloader = load_imagenet(args.batch_size, args.workers)
        num_classes = 1000
    if args.arch == 'VGG16':
        print("Experimenting with VGG16")
        model = VGG16(num_classes)
    elif args.arch == 'RESNET50':
        print("Experimenting with ResNet50")
        model = ResNet50(num_classes)
    elif args.arch == 'DPN92':
        print("Experimenting with DPN92")
        model = DPN92(num_classes)
    if args.cuda:
        device = 'cuda'
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    else:
        device = 'cpu'
    
    model = model.to(device)
    #wandb.watch(model)  
    
    best_acc = 0
    start_epoch = 0

    if args.load_name != None:
        print("Loading previous model:{}".format(args.load_name))
        checkpoint = torch.load(args.load_name)
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print("Model is from epoch:{}".format(start_epoch))
        model.load_state_dict(checkpoint['model'])
        args.save_name = args.load_name
    
    print_sparse_model_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    epoch = start_epoch
    if args.train == True:
        args.load_name = args.save_name
        print("Training the model!")
        for epoch in range(start_epoch, args.epochs):
            best_acc = test(model, epoch, testloader, device, criterion, args, best_acc)
            train(model, epoch, trainloader, device, optimizer, criterion)

    
    if args.eval == True:
        print("Testing the model {}!".format(args.load_name))
        best_acc = test(model, epoch, testloader, device, criterion, args, best_acc)

    print("Best Accuracy of model is {}".format(best_acc))
    if args.prune == True:
        """
        model = LeNet()
        General formula. 
        1. Train a model Regulary. 
        2. Stabilize for 2 epochs at a learning rate of 0.1
        3. Prune model 5% of weights via heuristic
        4. Retrain for an epoch with a learning rate of 0.01
        5. Repear 3 and 4 until 90% of weights are prunned
        6. Train prune model for another set of N epochs with a LR of 0.001 to fine tune network. 
        """
        prune_epochs = int(args.target_sparcity/ args.prune_speed)
        print("Model will be pruned by {} percent each epoch for a total of {} epochs using the {} method. Model will be train for one epoch with learning rate of {} between pruning steps.".format(args.prune_speed,prune_epochs, args.prune_method, args.prune_learning_rate))
        print("Stabilizing Model")
        optimizer = optim.SGD(model.parameters(), lr=args.prune_learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        args.prune_learning_rate  *= .1
        args.save_name = args.save_name + "_prune"
        accuracies = []
        for epoch in range(start_epoch, start_epoch + 0):
            acc = test(model, epoch, testloader, device, criterion, args, best_acc, prune_flag=True)
            accuracies.append((0, acc))
            train(model, epoch, trainloader, device, optimizer, criterion)
        print("Model Stabilized with accuracy:{}.\n Moving on to Prunning\n".format(best_acc))
        print_sparse_model_params(model)
        acc = 0
        optimizer = optim.SGD(model.parameters(), lr=args.prune_learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        print('Prune using Gradual Magnitude Pruning for using {}'.format(args.prune_method))
        args.save_name = args.save_name + args.prune_method
        #acc = test(model, start_epoch, testloader, device, criterion, args, best_acc, prune_flag=True)
        accuracies.append((0, acc))
        weight_sparcity = 0
        for i in range(1, prune_epochs+1):
            print_sparse_model_params(model)
            target = float(args.prune_speed * i)
            sparcities = [[],[]]
            for module in model.module.features: # for VGG
            #for module in model.module.children():
                    pruneable = False
                    if isinstance(module, nn.Conv2d):
                        for p in module.parameters():
                            if p.requires_grad:
                                pruneable = True
                                break
                    if pruneable:
                        sparcities[0].append(get_module_weight_sparcity(module))  
            weight_sparcity = np.average(sparcities[0]) 
            while weight_sparcity <= target:
                sparcities[1] = []
                for module in model.module.features: # for VGG
                #for module in model.module.children():
                    pruneable = False
                    if isinstance(module, nn.Conv2d):
                        for p in module.parameters():
                            if p.requires_grad:
                                pruneable = True
                                break
                    if pruneable and get_module_weight_sparcity(module) <= target:
                        prune_percentage = args.nim_stride_size
                        if args.prune_method == 'L1': 
                            L1Prune(module, prune_percentage)
                        elif args.prune_method == 'RANDOM':
                            RandomPrune(module, prune_percentage)
                        elif args.prune_method == 'L1+RANDOM':
                            L1RandomPrune(module, prune_percentage)
                        elif args.prune_method == 'POSITIVE':
                            PositiveMagnitudePrune(module, prune_percentage)
                        elif args.prune_method == 'NEGATIVE':
                            NegativeMagnitudePrune(module, prune_percentage)
                        elif args.prune_method == 'MAGNITUDE':
                            MagnitudePrune(module, prune_percentage)
                        elif args.prune_method == 'MAGNITUDE+RANDOM':
                            MagnitudeRandomPrune(module, prune_percentage)
                        elif args.prune_method == 'MAGNITUDE+L1':
                            MagnitudeL1Prune(module, prune_percentage)
                        sparcities[1].append(get_module_weight_sparcity(module))
                weight_sparcity = np.average(sparcities[1])
            sparcities[1] = []   
            for module in model.module.features: # for VGG
            #for module in model.module.children():
                pruneable = False
                if isinstance(module, nn.Conv2d):
                    for p in module.parameters():
                        if p.requires_grad:
                            pruneable = True
                            break
                if pruneable:
                    sparcities[1].append(get_module_weight_sparcity(module))  
            print("Weight Sparcity Before:{}.\nWeight Sparcity After{}.\n".format(np.average(sparcities[0]),np.average(sparcities[1])))
            #accuracies.append((weight_sparcity, test(model, i, testloader, device, criterion, args, best_acc)))
            #train(model, i, trainloader, device, optimizer, criterion)
            #accuracies.append((weight_sparcity, test(model, i, testloader, device, criterion, args, best_acc)))
        

        print("Done Prunning. Now Stabilizing for 10 epochs")
        print_sparse_model_params(model)
        args.prune_learning_rate  *= .1
        args.save_name = args.save_name + "_stabilized"
        optimizer = optim.SGD(model.parameters(), lr=args.prune_learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        for epoch in range(10):
            #train(model, epoch, trainloader, device, optimizer, criterion)
            acc = test(model, epoch, testloader, device, criterion, args, best_acc)
            accuracies.append((weight_sparcity, acc))
        print("Model Accuracy with regards to spacity:{}".format(accuracies))

    
        
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments in Using Game Theory For Model Pruning')
    # Basic Params
    parser.add_argument('--log_interval', default=100)
    parser.add_argument('--load_name', default=None, metavar='PATH',help='path to resume training from')
    parser.add_argument('--save_name', help='The path used to save the trained models',default='model', type=str)
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CPU or CUDA GPU' )
    parser.add_argument('--dataset', default='cifar10',help='Data set used for testing and training', choices=['cifar10','cifar100','imagenet'])

    # Model Train Params
    parser.add_argument('--arch', default='RESNET50', choices=['VGG16','RESNET50','DPN92'],help='model architectures: VGG16, LENET, RESNET, DPN92')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int,  help='mini-batch size (default: 128)')
    parser.add_argument('--learning_rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='M',help='Weight Decay')
    parser.add_argument('--train', action='store_true', help='Train network from scratch')
    parser.add_argument('--eval', action='store_true', help='Evaluate network')
    
    # Pruning Params
    parser.add_argument('--prune', action='store_true', help='Prune Network')
    parser.add_argument('--prune_method', default='L1', choices=['L1','RANDOM','L1+RANDOM','POSITIVE','NEGATIVE','MAGNITUDE','MAGNITUDE+RANDOM','MAGNITUDE+L1'])
    parser.add_argument('--prune_learning_rate', default=0.1, help='Learning rate for prunning. Stabilization starts, prunning continues with 1/10th, and then final stabilization with 1/100th')
    parser.add_argument('--target_sparcity', default=.95, help='Target Sparcity of the model [0,1]')
    parser.add_argument('--prune_speed', default=.05, help='Target percentage of weights to be pruned each step')
    parser.add_argument('--nim_stride_size', default=0.001, help='how much to prune each step. Will minimize to at least one weight for small layers')
    main(parser.parse_args())
