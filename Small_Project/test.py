import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models import *
from utils import progress_bar

# params
m_name = "ResNet20" 
print(f"Test Model {m_name}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 2
best_test_acc = 0


# Datasets
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='/data/datasets/cifar10/', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=True, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# models
net = ResNet20()
net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load(f"./checkpoint/ckpt_{m_name}.pth")
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
saved_epoch = checkpoint['epoch']
print(f"best_val_acc: {best_acc}, saved_epoch:{saved_epoch+1}")

# Test Settings
criterion = nn.CrossEntropyLoss()

# Test
def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(): # gradient(모델 파라미터 업뎃 X, 평가 모드)
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    print(f"{m_name}'s Final Test Accuracy for Test Dataset: {acc}, Correct: {correct}, Total: {total}")
    
test()