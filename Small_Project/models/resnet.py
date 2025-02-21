import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, negative_slope=0.1)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1) 
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2) 
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2) 
        self.linear = nn.Linear(64*block.expansion, num_classes) 

        # weigt init
        self.weight_init()

    def _make_layer(self,block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu') 
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8) 
        out = out.view(out.size(0), -1) 
        logit = self.linear(out) 
        return logit

def ResNet20():
    return ResNet(BasicBlock,[3,3,3])

def ResNet32():
    return ResNet(BasicBlock,[5,5,5]) # 0.46M

def ResNet44():
    return ResNet(BasicBlock,[7,7,7]) # 0.66M

def ResNet56():
    return ResNet(BasicBlock,[9,9,9]) # 0.86M

def ResNet110():
    return ResNet(BasicBlock,[18,18,18]) # 1.7M

def test():
    net = ResNet20()
    #import torchsummary
    #torchsummary.summary(net, (3, 32, 32)) 

#test()