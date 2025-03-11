# -*- coding: utf-8 -*-
"""fully_connected.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1V7-oS-C5o23HJWLsVJR8BwMwcXCZeL64
"""

import torch.nn as nn
import torch
import math
import torch.autograd

class GradientRescaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input)
        ctx.gd_scale_weight = weight
        output = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = ctx.gd_scale_weight * grad_output
        return grad_input, grad_weight

gradient_rescale = GradientRescaleFunction.apply


class DenseBasic(nn.Module):
    def __init__(self, nIn, nOut):
        super(DenseBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(nIn, nOut, bias=False),
            nn.BatchNorm1d(nOut),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class DenseLayer(nn.Module):
    def __init__(self, nIn, nOut):
        super(DenseLayer, self).__init__()
        self.dense = nn.Linear(nIn, nOut, bias=False)
        self.bn = nn.BatchNorm1d(nOut)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.bn(self.dense(x)))


class MSDNFirstLayerTabular(nn.Module):
    def __init__(self, nIn, nOut):
        super(MSDNFirstLayerTabular, self).__init__()
        self.layer = DenseBasic(nIn, nOut)

    def forward(self, x):
        return self.layer(x)


class MSDNLayerTabular(nn.Module):
    def __init__(self, nIn, nOut):
        super(MSDNLayerTabular, self).__init__()
        self.layer = DenseLayer(nIn, nOut)

    def forward(self, x):
        return self.layer(x)


class ClassifierModuleTabular(nn.Module):
    def __init__(self, nIn, num_classes):
        super(ClassifierModuleTabular, self).__init__()
        self.fc = nn.Linear(nIn, num_classes)

    def forward(self, x):
        return self.fc(x), x


class MSDNetTabular(nn.Module):
    def __init__(self, nIn, nOut, nBlocks, num_classes):
        super(MSDNetTabular, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.nBlocks = nBlocks

        for i in range(nBlocks):
            self.blocks.append(MSDNLayerTabular(nIn, nOut))
            self.classifier.append(ClassifierModuleTabular(nOut, num_classes))
            nIn = nOut

    def forward(self, x, stage=None):
        res = []
        for i in range(self.nBlocks):
            x = self.blocks[i](x)
            x = gradient_rescale(x, 1.0 / (self.nBlocks - i))
            pred, _ = self.classifier[i](x)
            x = gradient_rescale(x, (self.nBlocks - i - 1))
            res.append(pred)
            if i == stage:
                break
        return res