#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:55:48 2018

@author: sylvain
"""

from torchvision import models as torchmodels
import torch.nn as nn
import torch.nn.functional as F
import torch

class test_model(nn.Module):
    def __init__(self,input_size = 512):
        super(test_model, self).__init__()                
        self.visual = torchmodels.resnet152(pretrained=True)
        children_counter = 0

        for n,c in self.visual.named_children():
            print("Children Counter: ",children_counter," Layer Name: ",n,)
            children_counter+=1

        self.layers = list(self.visual._modules.keys())
        print(self.layers)
        
        self.dummy_var = self.visual._modules.pop(self.layers[9])
        self.dummy_var = self.visual._modules.pop(self.layers[8])
        self.layer4 = self.visual._modules.pop(self.layers[7])
        self.layer3 = self.visual._modules.pop(self.layers[6])
        self.layer2 = self.visual._modules.pop(self.layers[5])
        self.layer1 = nn.Sequential(self.visual._modules)
        self.layer2 = nn.Sequential(self.layer2)
        self.layer3 = nn.Sequential(self.layer3)
        self.layer4 = nn.Sequential(self.layer4)
        
        
    def forward(self, input):
        out1=self.layer1(input)
        print(out1.shape)
        out2=self.layer2(out1)
        print(out2.shape)
        out3=self.layer3(out2)
        print(out3.shape)
        out4=self.layer4(out3)
        print(out4.shape)
        return out4