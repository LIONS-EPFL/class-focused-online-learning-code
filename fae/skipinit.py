'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_activ=True, **kwargs):
        super(BasicBlock, self).__init__()
        self.use_activ = use_activ
        self.activ = partial(nn.ReLU(inplace=True)) if self.use_activ else lambda x: x
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.res_multiplier = torch.nn.Parameter(torch.tensor(0.0))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.activ(self.conv1(x))
        out = self.conv2(out)
        out = self.res_multiplier * out
        out += self.shortcut(x)
        out = self.activ(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_activ=True):
        super(Bottleneck, self).__init__()
        self.use_activ = use_activ
        self.activ = partial(nn.ReLU(inplace=True)) if self.use_activ else lambda x: x
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.res_multiplier = torch.nn.Parameter(torch.tensor(0.0))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.activ(self.conv1(x))
        out = self.activ(self.conv2(out))
        out = self.conv3(out)
        out = self.res_multiplier * out
        out += self.shortcut(x)
        out = self.activ(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, ch_in=3, pool_adapt=False, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.pool_adapt = pool_adapt
        if pool_adapt:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = partial(F.avg_pool2d, kernel_size=4)

        self.conv1 = nn.Conv2d(ch_in, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, **kwargs)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **kwargs))
            self.in_planes = planes * block.expansion
        # # cheeky way to get the activation from the layer1, e.g. in no activation case.
        self.activ = layers[0].activ
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activ(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNetnew(num_blocks=None, **kwargs):
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1]
    return ResNet(BasicBlock, num_blocks, **kwargs)

def SkipInit18(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def SkipInit34(**kwargs):
    return ResNet(BasicBlock, [3,4,6,3], **kwargs)

def SkipInit50(**kwargs):
    return ResNet(Bottleneck, [3,4,6,3], **kwargs)

def SkipInit101(**kwargs):
    return ResNet(Bottleneck, [3,4,23,3], **kwargs)

def SkipInit152(**kwargs):
    return ResNet(Bottleneck, [3,8,36,3], **kwargs)


def test():
    net = SkipInit18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
