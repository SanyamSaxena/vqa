#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:55:48 2018

@author: sylvain
"""

from torchvision import models as torchmodels
import torch.nn as nn
import models.seq2vec
import torch.nn.functional as F
import torch

VISUAL_OUT_GAP = 3840
VISUAL_OUT = 2048
QUESTION_OUT = 2400
FUSION_IN = 1200
FUSION_HIDDEN = 256
DROPOUT_V = 0.5
DROPOUT_Q = 0.5
DROPOUT_F = 0.5

class VQAModel(nn.Module):
    def __init__(self, vocab_questions, vocab_answers, input_size = 512):
        super(VQAModel, self).__init__()
        
        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)
        self.seq2vec = models.seq2vec.factory(self.vocab_questions, {'arch': 'skipthoughts', 'dir_st': 'data/skip-thoughts', 'type': 'BayesianUniSkip', 'dropout': 0.25, 'fixed_emb': False})
        for param in self.seq2vec.parameters():
            param.requires_grad = False
        self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        
        self.visual = torchmodels.resnet152(pretrained=True)
        extracted_layers = list(self.visual.children())
        extracted_layers = extracted_layers[0:8] #Remove the last fc and avg pool
        self.visual = torch.nn.Sequential(*(list(extracted_layers)))
        for param in self.visual.parameters():
            param.requires_grad = False
        
        output_size = (input_size / 32)**2
        self.visual = torch.nn.Sequential(self.visual, torch.nn.Conv2d(2048,int(2048/output_size),1))
        self.linear_v = nn.Linear(VISUAL_OUT, FUSION_IN)
        
        self.linear_classif1 = nn.Linear(FUSION_IN, FUSION_HIDDEN)
        self.linear_classif2 = nn.Linear(FUSION_HIDDEN, self.num_classes)
        
    def forward(self, input_v, input_q):
        x_v = self.visual(input_v).view(-1, VISUAL_OUT)
        x_v = self.dropoutV(x_v)
        x_v = self.linear_v(x_v)
        x_v = nn.Tanh()(x_v)
        
        x_q = self.seq2vec(input_q)
        x_q = self.dropoutV(x_q)
        x_q = self.linear_q(x_q)
        x_q = nn.Tanh()(x_q)
        
        x = torch.mul(x_v, x_q)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        
        return x
        
class VQAGAPModel(nn.Module):
    def __init__(self, vocab_questions, vocab_answers, input_size = 512):
        super(VQAGAPModel, self).__init__()
        
        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)
        self.seq2vec = models.seq2vec.factory(self.vocab_questions, {'arch': 'skipthoughts', 'dir_st': 'data/skip-thoughts', 'type': 'BayesianUniSkip', 'dropout': 0.25, 'fixed_emb': False})
        for param in self.seq2vec.parameters():
            param.requires_grad = False
        self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        
        self.visual = torchmodels.resnet152(pretrained=True)
        self.layers = list(self.visual._modules.keys())
        
        self.dummy_var = self.visual._modules.pop(self.layers[9])
        self.dummy_var = self.visual._modules.pop(self.layers[8])
        self.layer4 = self.visual._modules.pop(self.layers[7])
        self.layer3 = self.visual._modules.pop(self.layers[6])
        self.layer2 = self.visual._modules.pop(self.layers[5])
        self.layer1 = nn.Sequential(self.visual._modules)
        self.layer2 = nn.Sequential(self.layer2)
        self.layer3 = nn.Sequential(self.layer3)
        self.layer4 = nn.Sequential(self.layer4)

        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False
        
        self.g1 = 256
        self.g2 = 512
        self.g3 = 1024
        self.g4 = 2048

        self.gap1 = nn.Sequential(
            nn.Linear(QUESTION_OUT,self.g1),
            nn.Sigmoid()
        )
        self.gap2 = nn.Sequential(
            nn.Linear(QUESTION_OUT,self.g2),
            nn.Sigmoid()
        )
        self.gap3 = nn.Sequential(
            nn.Linear(QUESTION_OUT,self.g3),
            nn.Sigmoid()
        )
        self.gap4 = nn.Sequential(
            nn.Linear(QUESTION_OUT,self.g4),
            nn.Sigmoid()
        )

        # output_size = (input_size / 32)**2
        # self.visual = torch.nn.Sequential(self.visual, torch.nn.Conv2d(2048,int(2048/output_size),1))
        self.linear_v = nn.Linear(VISUAL_OUT_GAP, FUSION_IN)
        
        self.linear_classif1 = nn.Linear(FUSION_IN, FUSION_HIDDEN)
        self.linear_classif2 = nn.Linear(FUSION_HIDDEN, self.num_classes)
        
    def forward(self, input_v, input_q):
        x_q = self.seq2vec(input_q)
        x_q = self.dropoutV(x_q)
        x_q = self.linear_q(x_q)
        x_q = nn.Tanh()(x_q)

        # x_v = self.visual(input_v).view(-1, VISUAL_OUT)
        x_v_1=self.layer1(input_v)
        x_v_2=self.layer2(x_v_1)
        x_v_3=self.layer3(x_v_2)
        x_v_4=self.layer4(x_v_3)

        x_v_1_avg = torch.mean(x_v_1, axis=(2,3))
        x_v_2_avg = torch.mean(x_v_2, axis=(2,3))
        x_v_3_avg = torch.mean(x_v_3, axis=(2,3))
        x_v_4_avg = torch.mean(x_v_4, axis=(2,3))

        x_q_m1_out=self.gap1(x_q)
        x_q_m2_out=self.gap2(x_q)
        x_q_m3_out=self.gap3(x_q)
        x_q_m4_out=self.gap4(x_q)

        x_v = torch.cat((x_v_1_avg*x_q_m1_out,x_v_2_avg*x_q_m2_out,x_v_3_avg*x_q_m3_out,x_v_4_avg*x_q_m4_out),1)
        x_v = self.dropoutV(x_v)
        x_v = self.linear_v(x_v)
        x_v = nn.Tanh()(x_v)
            
        x = torch.mul(x_v, x_q)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        
        return x
