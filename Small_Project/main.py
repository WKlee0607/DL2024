import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
from auto_augment import AutoAugment, Cutout

import os
import argparse

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy on valid dataset
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

m_name = "ResNet20" 
print(f"Train Model {m_name}")

# Data
print('==> Preparing data..') 
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10), # 추가
    #transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)), # 추가
    AutoAugment(),
    # Cutout(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='/data/datasets/cifar10/', train=True, download=False, transform=transform_train)

validset = torchvision.datasets.CIFAR10(
    root='/data/datasets/cifar10/', train=True, download=False, transform=transform_valid)

valid_size = 0.1
import time 
random_seed = int(time.time())
num_workers = 2 
batch_size = 128
shuffle = True
pin_memory = False

num_train = len(trainset) # 50k
indices = list(range(num_train)) # 0 ~ 49,999 
split = int(np.floor(valid_size * num_train))


if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, sampler=train_sampler,
    num_workers=num_workers, pin_memory=pin_memory,
)
validloader = torch.utils.data.DataLoader(
    validset, batch_size=100, sampler=valid_sampler,
    num_workers=num_workers, pin_memory=pin_memory,
)
print("# of train Batch:", len(trainloader))
print("# of valid Batch:", len(validloader))

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet20() 
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume: # fine-tuning 등 재훈련 시킬 때 사용.
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f"./checkpoint/ckpt_{m_name}.pth")
    net.load_state_dict(checkpoint['net'], strict=False)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(f"resum !!!!, best_acc: {best_acc}, start_epoch:{start_epoch}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) 

# Training
def train(epoch):
    print(f'\nEpoch:{epoch}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device) 
        optimizer.zero_grad() 
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward() 
        optimizer.step() 

        train_loss += loss.item() 
        _, predicted = outputs.max(1) 
        total += targets.size(0) # 
        correct += predicted.eq(targets).sum().item() 

        progress_bar(batch_idx, len(trainloader), 
                    'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 
                    100.*correct/total, correct, total))
        
def val_test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(): # gradient(모델 파라미터 업뎃 X, 평가 모드)
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(validloader), 'Val Loss: %.3f | Val Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/ckpt_{m_name}.pth')
        best_acc = acc


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch+220):
        train(epoch)
        val_test(epoch)
        scheduler.step()
    print(f"{m_name}'s Best Valid Accuracy for Valid Dataset: {best_acc}")
    
    
    
