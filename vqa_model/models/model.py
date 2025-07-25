"""
Created on Fri Nov 30 09:55:48 2018

@author: 
"""
from torch.autograd import Variable
from torchvision import models as torchmodels
import torch.nn as nn
# import models.seq2vec
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch_geometric.nn import TransformerConv
from transformers import VisualBertModel, BertTokenizer, VisualBertConfig
from sentence_transformers import SentenceTransformer
from transformers import (
    # Preprocessing / Common
    AutoTokenizer, AutoFeatureExtractor,
    # Text & Image Models (Now, image transformers like ViTModel, DeiTModel, BEiT can also be loaded using AutoModel)
    AutoModel,            
    # Training / Evaluation
    TrainingArguments, Trainer,
    # Misc
    logging
)


class VQAModel(nn.Module):
    def __init__(self, vocab_questions, vocab_answers, args, input_size = 512):
        super(VQAModel, self).__init__()
        
        self.args = args
        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)
        self.seq2vec = models.seq2vec.factory(self.vocab_questions, {'arch': 'skipthoughts', 'dir_st': '../data/skip-thoughts', 'type': 'BayesianUniSkip', 'dropout': 0.25, 'fixed_emb': False})
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
    def __init__(self, vocab_questions, vocab_answers, args, input_size = 512):
        super(VQAGAPModel, self).__init__()
        
        self.args = args
        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)
        self.seq2vec = models.seq2vec.factory(self.vocab_questions, {'arch': 'skipthoughts', 'dir_st': '../data/skip-thoughts', 'type': 'BayesianUniSkip', 'dropout': 0.25, 'fixed_emb': False})
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
        x_q_f = self.dropoutQ(x_q)
        x_q_f = self.linear_q(x_q_f)
        x_q_f = nn.Tanh()(x_q_f)

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
            
        x = torch.mul(x_v, x_q_f)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        
        return x


class VQAGAPModel_finetune(nn.Module):
    def __init__(self, vocab_questions, vocab_answers, args, input_size = 512):
        super(VQAGAPModel_finetune, self).__init__()
        
        self.args = args
        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)
        self.seq2vec = models.seq2vec.factory(self.vocab_questions, {'arch': 'skipthoughts', 'dir_st': '../data/skip-thoughts', 'type': 'BayesianUniSkip', 'dropout': 0.25, 'fixed_emb': False})
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
            param.requires_grad = True
        for param in self.layer2.parameters():
            param.requires_grad = True
        for param in self.layer3.parameters():
            param.requires_grad = True
        for param in self.layer4.parameters():
            param.requires_grad = True
        
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
        x_q_f = self.dropoutQ(x_q)
        x_q_f = self.linear_q(x_q_f)
        x_q_f = nn.Tanh()(x_q_f)

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
            
        x = torch.mul(x_v, x_q_f)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        
        return x


class VQAGAP_bert_Model(nn.Module):
    def __init__(self, vocab_questions, vocab_answers, args, input_size = 512):
        super(VQAGAP_bert_Model, self).__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)
        # self.seq2vec = models.seq2vec.factory(self.vocab_questions, {'arch': 'skipthoughts', 'dir_st': '../data/skip-thoughts', 'type': 'BayesianUniSkip', 'dropout': 0.25, 'fixed_emb': False})
        # for param in self.seq2vec.parameters():
        #     param.requires_grad = False
        self.linear_q = nn.Linear(1200, FUSION_IN)
        
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

        # self.gap1 = nn.Sequential(
        #     nn.Linear(QUESTION_OUT,self.g1),
        #     nn.Sigmoid()
        # )
        # self.gap2 = nn.Sequential(
        #     nn.Linear(QUESTION_OUT,self.g2),
        #     nn.Sigmoid()
        # )
        # self.gap3 = nn.Sequential(
        #     nn.Linear(QUESTION_OUT,self.g3),
        #     nn.Sigmoid()
        # )
        # self.gap4 = nn.Sequential(
        #     nn.Linear(QUESTION_OUT,self.g4),
        #     nn.Sigmoid()
        # )

        self.gap1 = nn.Sequential(
            nn.Linear(1200,self.g1),
            nn.Sigmoid()
        )
        self.gap2 = nn.Sequential(
            nn.Linear(1200,self.g2),
            nn.Sigmoid()
        )
        self.gap3 = nn.Sequential(
            nn.Linear(1200,self.g3),
            nn.Sigmoid()
        )
        self.gap4 = nn.Sequential(
            nn.Linear(1200,self.g4),
            nn.Sigmoid()
        )

        # output_size = (input_size / 32)**2
        # self.visual = torch.nn.Sequential(self.visual, torch.nn.Conv2d(2048,int(2048/output_size),1))
        self.linear_v = nn.Linear(VISUAL_OUT_GAP, FUSION_IN)
        
        self.linear_classif1 = nn.Linear(FUSION_IN, FUSION_HIDDEN)
        self.linear_classif2 = nn.Linear(FUSION_HIDDEN, self.num_classes)
        
    def forward(self, input_v, input_q):
        # x_q = self.seq2vec(input_q)
        x_q = self.tokenizer(input_q, return_tensors='pt', padding='max_length', max_length=1200)
        for key in x_q.keys():
          x_q[key].long().squeeze(0)
       
        input_ids = Variable(x_q['input_ids']).to(torch.device(f'cuda:{self.args.gpu}')).to(dtype=torch.float32)
        # x_q_f = self.dropoutQ(input_ids)
        x_q_f = self.linear_q(input_ids)
        x_q_f = nn.Tanh()(x_q_f)

        # x_v = self.visual(input_v).view(-1, VISUAL_OUT)
        x_v_1=self.layer1(input_v)
        x_v_2=self.layer2(x_v_1)
        x_v_3=self.layer3(x_v_2)
        x_v_4=self.layer4(x_v_3)

        x_v_1_avg = torch.mean(x_v_1, axis=(2,3))
        x_v_2_avg = torch.mean(x_v_2, axis=(2,3))
        x_v_3_avg = torch.mean(x_v_3, axis=(2,3))
        x_v_4_avg = torch.mean(x_v_4, axis=(2,3))

        x_q_m1_out=self.gap1(input_ids)
        x_q_m3_out=self.gap3(input_ids)
        x_q_m4_out=self.gap4(input_ids)
        x_q_m2_out=self.gap2(input_ids)

        x_v = torch.cat((x_v_1_avg*x_q_m1_out,x_v_2_avg*x_q_m2_out,x_v_3_avg*x_q_m3_out,x_v_4_avg*x_q_m4_out),1)
        x_v = self.dropoutV(x_v)
        x_v = self.linear_v(x_v)
        x_v = nn.Tanh()(x_v)
            
        x = torch.mul(x_v, x_q_f)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        
        return x

class VQAGAP_bert_Model_finetune(nn.Module):
    def __init__(self, vocab_questions, vocab_answers, args, input_size = 512):
        super(VQAGAP_bert_Model_finetune, self).__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)
        # self.seq2vec = models.seq2vec.factory(self.vocab_questions, {'arch': 'skipthoughts', 'dir_st': '../data/skip-thoughts', 'type': 'BayesianUniSkip', 'dropout': 0.25, 'fixed_emb': False})
        # for param in self.seq2vec.parameters():
        #     param.requires_grad = False
        self.linear_q = nn.Linear(1200, FUSION_IN)
        
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
            param.requires_grad = True
        for param in self.layer2.parameters():
            param.requires_grad = True
        for param in self.layer3.parameters():
            param.requires_grad = True
        for param in self.layer4.parameters():
            param.requires_grad = True
        
        self.g1 = 256
        self.g2 = 512
        self.g3 = 1024
        self.g4 = 2048

        # self.gap1 = nn.Sequential(
        #     nn.Linear(QUESTION_OUT,self.g1),
        #     nn.Sigmoid()
        # )
        # self.gap2 = nn.Sequential(
        #     nn.Linear(QUESTION_OUT,self.g2),
        #     nn.Sigmoid()
        # )
        # self.gap3 = nn.Sequential(
        #     nn.Linear(QUESTION_OUT,self.g3),
        #     nn.Sigmoid()
        # )
        # self.gap4 = nn.Sequential(
        #     nn.Linear(QUESTION_OUT,self.g4),
        #     nn.Sigmoid()
        # )

        self.gap1 = nn.Sequential(
            nn.Linear(1200,self.g1),
            nn.Sigmoid()
        )
        self.gap2 = nn.Sequential(
            nn.Linear(1200,self.g2),
            nn.Sigmoid()
        )
        self.gap3 = nn.Sequential(
            nn.Linear(1200,self.g3),
            nn.Sigmoid()
        )
        self.gap4 = nn.Sequential(
            nn.Linear(1200,self.g4),
            nn.Sigmoid()
        )

        # output_size = (input_size / 32)**2
        # self.visual = torch.nn.Sequential(self.visual, torch.nn.Conv2d(2048,int(2048/output_size),1))
        self.linear_v = nn.Linear(VISUAL_OUT_GAP, FUSION_IN)
        
        self.linear_classif1 = nn.Linear(FUSION_IN, FUSION_HIDDEN)
        self.linear_classif2 = nn.Linear(FUSION_HIDDEN, self.num_classes)
        
    def forward(self, input_v, input_q):
        # x_q = self.seq2vec(input_q)
        x_q = self.tokenizer(input_q, return_tensors='pt', padding='max_length', max_length=1200)
        for key in x_q.keys():
          x_q[key].long().squeeze(0)
       
        input_ids = Variable(x_q['input_ids']).to(torch.device(f'cuda:{self.args.gpu}')).to(dtype=torch.float32)
        # x_q_f = self.dropoutQ(input_ids)
        x_q_f = self.linear_q(input_ids)
        x_q_f = nn.Tanh()(x_q_f)

        # x_v = self.visual(input_v).view(-1, VISUAL_OUT)
        x_v_1=self.layer1(input_v)
        x_v_2=self.layer2(x_v_1)
        x_v_3=self.layer3(x_v_2)
        x_v_4=self.layer4(x_v_3)

        x_v_1_avg = torch.mean(x_v_1, axis=(2,3))
        x_v_2_avg = torch.mean(x_v_2, axis=(2,3))
        x_v_3_avg = torch.mean(x_v_3, axis=(2,3))
        x_v_4_avg = torch.mean(x_v_4, axis=(2,3))

        x_q_m1_out=self.gap1(input_ids)
        x_q_m3_out=self.gap3(input_ids)
        x_q_m4_out=self.gap4(input_ids)
        x_q_m2_out=self.gap2(input_ids)

        x_v = torch.cat((x_v_1_avg*x_q_m1_out,x_v_2_avg*x_q_m2_out,x_v_3_avg*x_q_m3_out,x_v_4_avg*x_q_m4_out),1)
        x_v = self.dropoutV(x_v)
        x_v = self.linear_v(x_v)
        x_v = nn.Tanh()(x_v)
            
        x = torch.mul(x_v, x_q_f)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        
        return x


class VQAGAP_qbert_Model(nn.Module):
    def __init__(self, vocab_questions, vocab_answers, args, input_size = 512):
        super(VQAGAP_qbert_Model, self).__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')
        # choose the number of pretrained layers
        config = VisualBertConfig(num_hidden_layers=12, visual_embedding_dim=3840)
        self.visual_bert = VisualBertModel(config)
        for key in self.visual_bert.state_dict().keys():
            self.visual_bert.state_dict()[key] = model.state_dict()[key]

        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)

        # self.seq2vec = models.seq2vec.factory(self.vocab_questions, {'arch': 'skipthoughts', 'dir_st': 'data/skip-thoughts', 'type': 'BayesianUniSkip', 'dropout': 0.25, 'fixed_emb': False})
        # for param in self.seq2vec.parameters():
        #     param.requires_grad = False        

        # self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        
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
            nn.Linear(40,self.g1),
            nn.Sigmoid()
        )
        self.gap2 = nn.Sequential(
            nn.Linear(40,self.g2),
            nn.Sigmoid()
        )
        self.gap3 = nn.Sequential(
            nn.Linear(40,self.g3),
            nn.Sigmoid()
        )
        self.gap4 = nn.Sequential(
            nn.Linear(40,self.g4),
            nn.Sigmoid()
        )

        # output_size = (input_size / 32)**2
        # self.visual = torch.nn.Sequential(self.visual, torch.nn.Conv2d(2048,int(2048/output_size),1))
        self.linear_v = nn.Linear(VISUAL_OUT_GAP, FUSION_IN)
        
        #Prediction Head - MLP
        # self.input_classify_linear = torch.nn.Linear(768, 1200)
        # self.hidden_classify_linear = torch.nn.Linear(1200, 256)
        # self.classify_linear = torch.nn.Linear(256, number_outputs)

        self.linear_classif1 = nn.Linear(768, 1200)
        self.linear_classif2 = nn.Linear(1200, FUSION_HIDDEN)
        # print("num classes answer", self.num_classes)
        self.linear_classif3 = nn.Linear(FUSION_HIDDEN, self.num_classes)
        
    def forward(self, input_v, input_q):
        # create placeholders for the VisualBERT, these necessary for special tasks (then there are not only 1)
        # print(type(input_q))
        x_q = self.tokenizer(input_q, return_tensors='pt', padding='max_length', max_length=40)

        # reject a possible leading 0 dimension
        for key in x_q.keys():
          x_q[key].long().squeeze(0)
        # print(x_q['input_ids'].shape)
        # print(input_q.shape)
        x_v_1=self.layer1(input_v)
        x_v_2=self.layer2(x_v_1)
        x_v_3=self.layer3(x_v_2)
        x_v_4=self.layer4(x_v_3)

        x_v_1_avg = torch.mean(x_v_1, axis=(2,3))
        x_v_2_avg = torch.mean(x_v_2, axis=(2,3))
        x_v_3_avg = torch.mean(x_v_3, axis=(2,3))
        x_v_4_avg = torch.mean(x_v_4, axis=(2,3))

        input_ids = Variable(x_q['input_ids']).to(torch.device(f'cuda:{self.args.gpu}')).to(dtype=torch.float32)
        input_ids_1 = Variable(x_q['input_ids']).to(torch.device(f'cuda:{self.args.gpu}'))
        token_type_ids = Variable(x_q['token_type_ids']).to(torch.device(f'cuda:{self.args.gpu}'))
        attention_mask = Variable(x_q['attention_mask']).to(torch.device(f'cuda:{self.args.gpu}'))

        x_q_m1_out=self.gap1(input_ids)
        x_q_m3_out=self.gap3(input_ids)
        x_q_m4_out=self.gap4(input_ids)
        x_q_m2_out=self.gap2(input_ids)

        x_v = torch.cat((x_v_1_avg*x_q_m1_out,x_v_2_avg*x_q_m2_out,x_v_3_avg*x_q_m3_out,x_v_4_avg*x_q_m4_out),1)
        B,N = x_v.shape
        # print("B,N",B,N)
        x_v = x_v.view(B,1,N)
        visual_attention_mask = Variable(torch.ones(x_v.shape[:-1], dtype=torch.long)).to(torch.device(f'cuda:{self.args.gpu}'))
        visual_token_type_ids = Variable(torch.ones(x_v.shape[:-1], dtype=torch.long)).to(torch.device(f'cuda:{self.args.gpu}'))
        # print("x_q shape",x_q['input_ids'].shape)   
        
        

        out = self.visual_bert(input_ids=input_ids_1, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                visual_embeds=x_v, visual_attention_mask=visual_attention_mask,
                                visual_token_type_ids=visual_token_type_ids)
        
        # don't know why, but used in the original visualBERT for vqa
        index_to_gather = attention_mask.sum(1) - 2
        index_to_gather = (
                index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, out.last_hidden_state.size(-1))
        )
        x = torch.gather(out.last_hidden_state, 1, index_to_gather).squeeze(1)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        x = nn.Tanh()(x)
        x = self.linear_classif3(x)
        
        return x

class VQAGAP_qbert_Model_finetune(nn.Module):
    def __init__(self, vocab_questions, vocab_answers, args, input_size = 512):
        super(VQAGAP_qbert_Model_finetune, self).__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')
        # choose the number of pretrained layers
        config = VisualBertConfig(num_hidden_layers=12, visual_embedding_dim=3840)
        self.visual_bert = VisualBertModel(config)
        for key in self.visual_bert.state_dict().keys():
            self.visual_bert.state_dict()[key] = model.state_dict()[key]

        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)

        # self.seq2vec = models.seq2vec.factory(self.vocab_questions, {'arch': 'skipthoughts', 'dir_st': 'data/skip-thoughts', 'type': 'BayesianUniSkip', 'dropout': 0.25, 'fixed_emb': False})
        # for param in self.seq2vec.parameters():
        #     param.requires_grad = False        

        # self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        
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
            param.requires_grad = True
        for param in self.layer2.parameters():
            param.requires_grad = True
        for param in self.layer3.parameters():
            param.requires_grad = True
        for param in self.layer4.parameters():
            param.requires_grad = True
        
        self.g1 = 256
        self.g2 = 512
        self.g3 = 1024
        self.g4 = 2048

        self.gap1 = nn.Sequential(
            nn.Linear(40,self.g1),
            nn.Sigmoid()
        )
        self.gap2 = nn.Sequential(
            nn.Linear(40,self.g2),
            nn.Sigmoid()
        )
        self.gap3 = nn.Sequential(
            nn.Linear(40,self.g3),
            nn.Sigmoid()
        )
        self.gap4 = nn.Sequential(
            nn.Linear(40,self.g4),
            nn.Sigmoid()
        )

        # output_size = (input_size / 32)**2
        # self.visual = torch.nn.Sequential(self.visual, torch.nn.Conv2d(2048,int(2048/output_size),1))
        self.linear_v = nn.Linear(VISUAL_OUT_GAP, FUSION_IN)
        
        #Prediction Head - MLP
        # self.input_classify_linear = torch.nn.Linear(768, 1200)
        # self.hidden_classify_linear = torch.nn.Linear(1200, 256)
        # self.classify_linear = torch.nn.Linear(256, number_outputs)

        self.linear_classif1 = nn.Linear(768, 1200)
        self.linear_classif2 = nn.Linear(1200, FUSION_HIDDEN)
        # print("num classes answer", self.num_classes)
        self.linear_classif3 = nn.Linear(FUSION_HIDDEN, self.num_classes)
        
    def forward(self, input_v, input_q):
        # create placeholders for the VisualBERT, these necessary for special tasks (then there are not only 1)
        # print(type(input_q))
        x_q = self.tokenizer(input_q, return_tensors='pt', padding='max_length', max_length=40)

        # reject a possible leading 0 dimension
        for key in x_q.keys():
          x_q[key].long().squeeze(0)
        # print(x_q['input_ids'].shape)
        # print(input_q.shape)
        x_v_1=self.layer1(input_v)
        x_v_2=self.layer2(x_v_1)
        x_v_3=self.layer3(x_v_2)
        x_v_4=self.layer4(x_v_3)

        x_v_1_avg = torch.mean(x_v_1, axis=(2,3))
        x_v_2_avg = torch.mean(x_v_2, axis=(2,3))
        x_v_3_avg = torch.mean(x_v_3, axis=(2,3))
        x_v_4_avg = torch.mean(x_v_4, axis=(2,3))

        input_ids = Variable(x_q['input_ids']).to(torch.device(f'cuda:{self.args.gpu}')).to(dtype=torch.float32)
        input_ids_1 = Variable(x_q['input_ids']).to(torch.device(f'cuda:{self.args.gpu}'))
        token_type_ids = Variable(x_q['token_type_ids']).to(torch.device(f'cuda:{self.args.gpu}'))
        attention_mask = Variable(x_q['attention_mask']).to(torch.device(f'cuda:{self.args.gpu}'))

        x_q_m1_out=self.gap1(input_ids)
        x_q_m3_out=self.gap3(input_ids)
        x_q_m4_out=self.gap4(input_ids)
        x_q_m2_out=self.gap2(input_ids)

        x_v = torch.cat((x_v_1_avg*x_q_m1_out,x_v_2_avg*x_q_m2_out,x_v_3_avg*x_q_m3_out,x_v_4_avg*x_q_m4_out),1)
        B,N = x_v.shape
        # print("B,N",B,N)
        x_v = x_v.view(B,1,N)
        visual_attention_mask = Variable(torch.ones(x_v.shape[:-1], dtype=torch.long)).to(torch.device(f'cuda:{self.args.gpu}'))
        visual_token_type_ids = Variable(torch.ones(x_v.shape[:-1], dtype=torch.long)).to(torch.device(f'cuda:{self.args.gpu}'))
        # print("x_q shape",x_q['input_ids'].shape)   
        
        

        out = self.visual_bert(input_ids=input_ids_1, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                visual_embeds=x_v, visual_attention_mask=visual_attention_mask,
                                visual_token_type_ids=visual_token_type_ids)
        
        # don't know why, but used in the original visualBERT for vqa
        index_to_gather = attention_mask.sum(1) - 2
        index_to_gather = (
                index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, out.last_hidden_state.size(-1))
        )
        x = torch.gather(out.last_hidden_state, 1, index_to_gather).squeeze(1)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        x = nn.Tanh()(x)
        x = self.linear_classif3(x)
        
        return x


class VQAGAP_qbert_dca_Model(nn.Module):
    def __init__(self, vocab_questions, vocab_answers, args, input_size = 512):
        super(VQAGAP_qbert_dca_Model, self).__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')
        # choose the number of pretrained layers
        config = VisualBertConfig(num_hidden_layers=12, visual_embedding_dim=3840)
        self.visual_bert = VisualBertModel(config)
        for key in self.visual_bert.state_dict().keys():
            self.visual_bert.state_dict()[key] = model.state_dict()[key]

        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)

        # self.seq2vec = models.seq2vec.factory(self.vocab_questions, {'arch': 'skipthoughts', 'dir_st': 'data/skip-thoughts', 'type': 'BayesianUniSkip', 'dropout': 0.25, 'fixed_emb': False})
        # for param in self.seq2vec.parameters():
        #     param.requires_grad = False        

        # self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        
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
            nn.Linear(128,self.g1),
            nn.Sigmoid()
        )
        self.gap2 = nn.Sequential(
            nn.Linear(128,self.g2),
            nn.Sigmoid()
        )
        self.gap3 = nn.Sequential(
            nn.Linear(128,self.g3),
            nn.Sigmoid()
        )
        self.gap4 = nn.Sequential(
            nn.Linear(128,self.g4),
            nn.Sigmoid()
        )

        self.ca_on_text = nn.Sequential(
            nn.Linear(3840,128),
            nn.Sigmoid()
        )

        # output_size = (input_size / 32)**2
        # self.visual = torch.nn.Sequential(self.visual, torch.nn.Conv2d(2048,int(2048/output_size),1))
        self.linear_v = nn.Linear(VISUAL_OUT_GAP, FUSION_IN)
        
        #Prediction Head - MLP
        # self.input_classify_linear = torch.nn.Linear(768, 1200)
        # self.hidden_classify_linear = torch.nn.Linear(1200, 256)
        # self.classify_linear = torch.nn.Linear(256, number_outputs)

        self.linear_classif1 = nn.Linear(768, 1200)
        self.linear_classif2 = nn.Linear(1200, FUSION_HIDDEN)
        # print("num classes answer", self.num_classes)
        self.linear_classif3 = nn.Linear(FUSION_HIDDEN, self.num_classes)
        
    def forward(self, input_v, input_q):
        # create placeholders for the VisualBERT, these necessary for special tasks (then there are not only 1)
        x_q = self.tokenizer(input_q, return_tensors='pt', padding='max_length', max_length=128)

        # reject a possible leading 0 dimension
        for key in x_q.keys():
          x_q[key].long().squeeze(0)
        # print(x_q['input_ids'].shape)
        # print(input_q.shape)
        x_v_1=self.layer1(input_v)
        x_v_2=self.layer2(x_v_1)
        x_v_3=self.layer3(x_v_2)
        x_v_4=self.layer4(x_v_3)

        x_v_1_avg = torch.mean(x_v_1, axis=(2,3))
        x_v_2_avg = torch.mean(x_v_2, axis=(2,3))
        x_v_3_avg = torch.mean(x_v_3, axis=(2,3))
        x_v_4_avg = torch.mean(x_v_4, axis=(2,3))

        input_ids = Variable(x_q['input_ids']).to(torch.device(f'cuda:{self.args.gpu}')).to(dtype=torch.float32)
        input_ids_1 = Variable(x_q['input_ids']).to(torch.device(f'cuda:{self.args.gpu}'))
        token_type_ids = Variable(x_q['token_type_ids']).to(torch.device(f'cuda:{self.args.gpu}'))
        attention_mask = Variable(x_q['attention_mask']).to(torch.device(f'cuda:{self.args.gpu}'))

        x_v_unmasked = torch.cat((x_v_1_avg,x_v_2_avg,x_v_3_avg,x_v_4_avg),1)
        x_v_mask = self.ca_on_text(x_v_unmasked)
        masked_input_ids = (x_v_mask*input_ids).to(input_ids_1.dtype)

        x_q_m1_out=self.gap1(input_ids)
        x_q_m3_out=self.gap3(input_ids)
        x_q_m4_out=self.gap4(input_ids)
        x_q_m2_out=self.gap2(input_ids)

        x_v = torch.cat((x_v_1_avg*x_q_m1_out,x_v_2_avg*x_q_m2_out,x_v_3_avg*x_q_m3_out,x_v_4_avg*x_q_m4_out),1)

        B,N = x_v.shape
        # print("B,N",B,N)
        x_v = x_v.view(B,1,N)
        visual_attention_mask = Variable(torch.ones(x_v.shape[:-1], dtype=torch.long)).to(torch.device(f'cuda:{self.args.gpu}'))
        visual_token_type_ids = Variable(torch.ones(x_v.shape[:-1], dtype=torch.long)).to(torch.device(f'cuda:{self.args.gpu}'))
        # print("x_q shape",x_q['input_ids'].shape)   

        
        out = self.visual_bert(input_ids=masked_input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                visual_embeds=x_v, visual_token_type_ids=visual_token_type_ids,
                                visual_attention_mask=visual_attention_mask)

        # don't know why, but used in the original visualBERT for vqa
        index_to_gather = attention_mask.sum(1) - 2
        index_to_gather = (
                index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, out.last_hidden_state.size(-1))
        )
        x = torch.gather(out.last_hidden_state, 1, index_to_gather).squeeze(1)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        x = nn.Tanh()(x)
        x = self.linear_classif3(x)
        
        return x

class VQAGAP_qbert_dca_Model_finetune(nn.Module):
    def __init__(self, vocab_questions, vocab_answers, args, input_size = 512):
        super(VQAGAP_qbert_dca_Model_finetune, self).__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')
        # choose the number of pretrained layers
        config = VisualBertConfig(num_hidden_layers=12, visual_embedding_dim=3840)
        self.visual_bert = VisualBertModel(config)
        for key in self.visual_bert.state_dict().keys():
            self.visual_bert.state_dict()[key] = model.state_dict()[key]

        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)

        # self.seq2vec = models.seq2vec.factory(self.vocab_questions, {'arch': 'skipthoughts', 'dir_st': 'data/skip-thoughts', 'type': 'BayesianUniSkip', 'dropout': 0.25, 'fixed_emb': False})
        # for param in self.seq2vec.parameters():
        #     param.requires_grad = False        

        # self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        
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
            param.requires_grad = True
        for param in self.layer2.parameters():
            param.requires_grad = True
        for param in self.layer3.parameters():
            param.requires_grad = True
        for param in self.layer4.parameters():
            param.requires_grad = True
        
        self.g1 = 256
        self.g2 = 512
        self.g3 = 1024
        self.g4 = 2048

        self.gap1 = nn.Sequential(
            nn.Linear(128,self.g1),
            nn.Sigmoid()
        )
        self.gap2 = nn.Sequential(
            nn.Linear(128,self.g2),
            nn.Sigmoid()
        )
        self.gap3 = nn.Sequential(
            nn.Linear(128,self.g3),
            nn.Sigmoid()
        )
        self.gap4 = nn.Sequential(
            nn.Linear(128,self.g4),
            nn.Sigmoid()
        )

        self.ca_on_text = nn.Sequential(
            nn.Linear(3840,128),
            nn.Sigmoid()
        )

        # output_size = (input_size / 32)**2
        # self.visual = torch.nn.Sequential(self.visual, torch.nn.Conv2d(2048,int(2048/output_size),1))
        self.linear_v = nn.Linear(VISUAL_OUT_GAP, FUSION_IN)
        
        #Prediction Head - MLP
        # self.input_classify_linear = torch.nn.Linear(768, 1200)
        # self.hidden_classify_linear = torch.nn.Linear(1200, 256)
        # self.classify_linear = torch.nn.Linear(256, number_outputs)

        self.linear_classif1 = nn.Linear(768, 1200)
        self.linear_classif2 = nn.Linear(1200, FUSION_HIDDEN)
        # print("num classes answer", self.num_classes)
        self.linear_classif3 = nn.Linear(FUSION_HIDDEN, self.num_classes)
        
    def forward(self, input_v, input_q):
        # create placeholders for the VisualBERT, these necessary for special tasks (then there are not only 1)
        x_q = self.tokenizer(input_q, return_tensors='pt', padding='max_length', max_length=128)

        # reject a possible leading 0 dimension
        for key in x_q.keys():
          x_q[key].long().squeeze(0)
        # print(x_q['input_ids'].shape)
        # print(input_q.shape)
        x_v_1=self.layer1(input_v)
        x_v_2=self.layer2(x_v_1)
        x_v_3=self.layer3(x_v_2)
        x_v_4=self.layer4(x_v_3)

        x_v_1_avg = torch.mean(x_v_1, axis=(2,3))
        x_v_2_avg = torch.mean(x_v_2, axis=(2,3))
        x_v_3_avg = torch.mean(x_v_3, axis=(2,3))
        x_v_4_avg = torch.mean(x_v_4, axis=(2,3))

        input_ids = Variable(x_q['input_ids']).to(torch.device(f'cuda:{self.args.gpu}')).to(dtype=torch.float32)
        input_ids_1 = Variable(x_q['input_ids']).to(torch.device(f'cuda:{self.args.gpu}'))
        token_type_ids = Variable(x_q['token_type_ids']).to(torch.device(f'cuda:{self.args.gpu}'))
        attention_mask = Variable(x_q['attention_mask']).to(torch.device(f'cuda:{self.args.gpu}'))

        x_v_unmasked = torch.cat((x_v_1_avg,x_v_2_avg,x_v_3_avg,x_v_4_avg),1)
        x_v_mask = self.ca_on_text(x_v_unmasked)
        masked_input_ids = (x_v_mask*input_ids).to(input_ids_1.dtype)

        x_q_m1_out=self.gap1(input_ids)
        x_q_m3_out=self.gap3(input_ids)
        x_q_m4_out=self.gap4(input_ids)
        x_q_m2_out=self.gap2(input_ids)

        x_v = torch.cat((x_v_1_avg*x_q_m1_out,x_v_2_avg*x_q_m2_out,x_v_3_avg*x_q_m3_out,x_v_4_avg*x_q_m4_out),1)

        B,N = x_v.shape
        # print("B,N",B,N)
        x_v = x_v.view(B,1,N)
        visual_attention_mask = Variable(torch.ones(x_v.shape[:-1], dtype=torch.long)).to(torch.device(f'cuda:{self.args.gpu}'))
        visual_token_type_ids = Variable(torch.ones(x_v.shape[:-1], dtype=torch.long)).to(torch.device(f'cuda:{self.args.gpu}'))
        # print("x_q shape",x_q['input_ids'].shape)   

        
        out = self.visual_bert(input_ids=masked_input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                visual_embeds=x_v, visual_token_type_ids=visual_token_type_ids,
                                visual_attention_mask=visual_attention_mask)

        # don't know why, but used in the original visualBERT for vqa
        index_to_gather = attention_mask.sum(1) - 2
        index_to_gather = (
                index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, out.last_hidden_state.size(-1))
        )
        x = torch.gather(out.last_hidden_state, 1, index_to_gather).squeeze(1)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        x = nn.Tanh()(x)
        x = self.linear_classif3(x)
        
        return x


class VQAGAP_qbert_max_avg_pooled(nn.Module):
    def __init__(self, vocab_questions, vocab_answers, args, finetune_resnet=False):
        super(VQAGAP_qbert_max_avg_pooled, self).__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')
        # choose the number of pretrained layers
        config = VisualBertConfig(num_hidden_layers=12, visual_embedding_dim=4096)
        self.visual_bert = VisualBertModel(config)
        for key in self.visual_bert.state_dict().keys():
            self.visual_bert.state_dict()[key] = model.state_dict()[key]

        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)
        
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
            param.requires_grad = finetune_resnet
        for param in self.layer2.parameters():
            param.requires_grad = finetune_resnet
        for param in self.layer3.parameters():
            param.requires_grad = finetune_resnet
        for param in self.layer4.parameters():
            param.requires_grad = finetune_resnet
        
        self.g1 = 256
        self.g2 = 512
        self.g3 = 1024
        self.g4 = 2048

        self.dim_chg_gap1 = nn.Sequential(
            nn.Linear(self.g1, 2048),
            nn.Sigmoid()
        )
        self.dim_chg_gap2 = nn.Sequential(
            nn.Linear(self.g2, 2048),
            nn.Sigmoid()
        )
        self.dim_chg_gap3 = nn.Sequential(
            nn.Linear(self.g3, 2048),
            nn.Sigmoid()
        )
        self.dim_chg_gap4 = nn.Sequential(
            nn.Linear(self.g4, 2048),
            nn.Sigmoid()
        )

        self.w1 = nn.Parameter(torch.tensor(0.25))
        self.w2 = nn.Parameter(torch.tensor(0.25))
        self.w3 = nn.Parameter(torch.tensor(0.25))
        self.w4 = nn.Parameter(torch.tensor(0.25))

        self.linear_classif1 = nn.Linear(768, 1200)
        self.linear_classif2 = nn.Linear(1200, FUSION_HIDDEN)
        self.linear_classif3 = nn.Linear(FUSION_HIDDEN, self.num_classes)
        
    def forward(self, input_v, input_q):
        # create placeholders for the VisualBERT, these necessary for special tasks (then there are not only 1)
        x_q = self.tokenizer(input_q, return_tensors='pt', padding='max_length', max_length=128)

        # reject a possible leading 0 dimension
        for key in x_q.keys():
          x_q[key].long().squeeze(0)
        x_v_1=self.layer1(input_v)
        x_v_2=self.layer2(x_v_1)
        x_v_3=self.layer3(x_v_2)
        x_v_4=self.layer4(x_v_3)

        x_v_1_avg = torch.mean(x_v_1, axis=(2,3))
        x_v_2_avg = torch.mean(x_v_2, axis=(2,3))
        x_v_3_avg = torch.mean(x_v_3, axis=(2,3))
        x_v_4_avg = torch.mean(x_v_4, axis=(2,3))

        input_ids = Variable(x_q['input_ids']).to(torch.device(f'cuda:{self.args.gpu}'))
        token_type_ids = Variable(x_q['token_type_ids']).to(torch.device(f'cuda:{self.args.gpu}'))
        attention_mask = Variable(x_q['attention_mask']).to(torch.device(f'cuda:{self.args.gpu}'))

        x_v_1=self.dim_chg_gap1(x_v_1_avg)
        x_v_2=self.dim_chg_gap3(x_v_3_avg)
        x_v_3=self.dim_chg_gap4(x_v_4_avg)
        x_v_4=self.dim_chg_gap2(x_v_2_avg)
        stacked_x_v = torch.stack([self.w1*x_v_1, self.w2*x_v_2, self.w3*x_v_3, self.w4*x_v_4],2)
        x_v_avg = torch.mean(stacked_x_v,2)
        x_v_max, _ = torch.max(stacked_x_v,2)
        x_v = torch.cat((x_v_max,x_v_avg), 1)
        B,N = x_v.shape
        x_v = x_v.view(B,1,N)
        visual_attention_mask = Variable(torch.ones(x_v.shape[:-1], dtype=torch.long)).to(torch.device(f'cuda:{self.args.gpu}'))
        visual_token_type_ids = Variable(torch.ones(x_v.shape[:-1], dtype=torch.long)).to(torch.device(f'cuda:{self.args.gpu}'))
        
        out = self.visual_bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                visual_embeds=x_v, visual_token_type_ids=visual_token_type_ids,
                                visual_attention_mask=visual_attention_mask)

        # don't know why, but used in the original visualBERT for vqa
        index_to_gather = attention_mask.sum(1) - 2
        index_to_gather = (
                index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, out.last_hidden_state.size(-1))
        )
        x = torch.gather(out.last_hidden_state, 1, index_to_gather).squeeze(1)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        x = nn.Tanh()(x)
        x = self.linear_classif3(x)
        
        return x
    

class VQAGAP_bert_ca_max_avg_pooled(nn.Module):
    def __init__(self, vocab_questions, vocab_answers, args, finetune_resnet=True):
        super(VQAGAP_bert_ca_max_avg_pooled,self).__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)

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
            param.requires_grad = finetune_resnet
        for param in self.layer2.parameters():
            param.requires_grad = finetune_resnet
        for param in self.layer3.parameters():
            param.requires_grad = finetune_resnet
        for param in self.layer4.parameters():
            param.requires_grad = finetune_resnet
        
        self.g1 = 256
        self.g2 = 512
        self.g3 = 1024
        self.g4 = 2048

        self.dim_chg_gap1 = nn.Sequential(
            nn.Linear(self.g1, 2048),
            nn.Sigmoid()
        )
        self.dim_chg_gap2 = nn.Sequential(
            nn.Linear(self.g2, 2048),
            nn.Sigmoid()
        )
        self.dim_chg_gap3 = nn.Sequential(
            nn.Linear(self.g3, 2048),
            nn.Sigmoid()
        )
        self.dim_chg_gap4 = nn.Sequential(
            nn.Linear(self.g4, 2048),
            nn.Sigmoid()
        )

        self.w1 = nn.Parameter(torch.tensor(0.25))
        self.w2 = nn.Parameter(torch.tensor(0.25))
        self.w3 = nn.Parameter(torch.tensor(0.25))
        self.w4 = nn.Parameter(torch.tensor(0.25))
        
        self.question_2x = nn.Linear(128, 4096)
        self.linear_classif1 = nn.Linear(4096*2, 1200)
        self.linear_classif2 = nn.Linear(1200, FUSION_HIDDEN)
        self.linear_classif3 = nn.Linear(FUSION_HIDDEN, self.num_classes)
        
    def forward(self, input_v, input_q):
        # create placeholders for the VisualBERT, these necessary for special tasks (then there are not only 1)
        x_q = self.tokenizer(input_q, return_tensors='pt', padding='max_length', max_length=128)

        # reject a possible leading 0 dimension
        for key in x_q.keys():
          x_q[key].long().squeeze(0)

        input_ids = Variable(x_q['input_ids']).to(torch.device(f'cuda:{self.args.gpu}')).to(dtype=torch.float32)
        x_q_2x = self.question_2x(input_ids)

        x_v_1=self.layer1(input_v)
        x_v_2=self.layer2(x_v_1)
        x_v_3=self.layer3(x_v_2)
        x_v_4=self.layer4(x_v_3)

        x_v_1_avg = torch.mean(x_v_1, axis=(2,3))
        x_v_2_avg = torch.mean(x_v_2, axis=(2,3))
        x_v_3_avg = torch.mean(x_v_3, axis=(2,3))
        x_v_4_avg = torch.mean(x_v_4, axis=(2,3))

        x_v_1=self.dim_chg_gap1(x_v_1_avg)
        x_v_2=self.dim_chg_gap2(x_v_2_avg)
        x_v_3=self.dim_chg_gap3(x_v_3_avg)
        x_v_4=self.dim_chg_gap4(x_v_4_avg)

        stacked_x_v = torch.stack([self.w1*x_v_1, self.w2*x_v_2, self.w3*x_v_3, self.w4*x_v_4],2)
        x_v_avg = torch.mean(stacked_x_v,2)
        x_v_max, _ = torch.max(stacked_x_v,2)
        x_v_pooled = torch.cat((x_v_max,x_v_avg), 1)

        x_attn = x_v_pooled*x_q_2x
        x = torch.cat((x_q_2x,x_attn),1)

        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        x = nn.Tanh()(x)
        x = self.linear_classif3(x)
        
        return x


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_output):
        super().__init__()
        self.conv1 = GCNConv(num_features, 256)
        self.conv2 = GCNConv(256, num_output)

    def forward(self, x, edge_index, edge_weight):
        x = x.to(torch.float)
        x = torch.nan_to_num(x,0.0)
        
        edge_weight = edge_weight.to(torch.float)
        edge_weight = torch.unsqueeze(edge_weight,1)

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        pooled_avg = torch.mean(x,0)
        pooled_max,_ = torch.max(x,0)
        pooled = torch.cat((pooled_avg,pooled_max),0)
        # shape of x will be vertices * 128
        return pooled



class GraphTransformer(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GraphTransformer, self).__init__()
        self.conv1 = TransformerConv(num_node_features, 256)
        self.conv2 = TransformerConv(256, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = x.to(torch.float)
        x = torch.nan_to_num(x,0.0)

        edge_weight = edge_weight.to(torch.float)
        edge_weight = torch.unsqueeze(edge_weight,1)

        # x = self.conv1(x, edge_index, edge_weight=edge_weight)
        # x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        pooled_avg = torch.mean(x,0)
        pooled_max,_ = torch.max(x,0)
        pooled = torch.cat((pooled_avg,pooled_max),0)

        return pooled


class VQAGAPGCN_qbert_max_avg_pooled(nn.Module):
    def __init__(self, vocab_questions, vocab_answers, args, pretrained_resnet=False):
        super(VQAGAPGCN_qbert_max_avg_pooled, self).__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')
        self.gcn_model = GCN(659, 128)
        self.gcn = 256
        # choose the number of pretrained layers
        config = VisualBertConfig(num_hidden_layers=12, visual_embedding_dim=4096)
        self.visual_bert = VisualBertModel(config)
        for key in self.visual_bert.state_dict().keys():
            self.visual_bert.state_dict()[key] = model.state_dict()[key]

        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)
        
        self.visual = torchmodels.resnet152(pretrained=pretrained_resnet)
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
            param.requires_grad = not pretrained_resnet
        for param in self.layer2.parameters():
            param.requires_grad = not pretrained_resnet
        for param in self.layer3.parameters():
            param.requires_grad = not pretrained_resnet
        for param in self.layer4.parameters():
            param.requires_grad = not pretrained_resnet
        
        self.g1 = 256
        self.g2 = 512
        self.g3 = 1024
        self.g4 = 2048

        self.dim_chg_gap1 = nn.Sequential(
            nn.Linear(self.g1, 2048),
            nn.Sigmoid()
        )
        self.dim_chg_gap2 = nn.Sequential(
            nn.Linear(self.g2, 2048),
            nn.Sigmoid()
        )
        self.dim_chg_gap3 = nn.Sequential(
            nn.Linear(self.g3, 2048),
            nn.Sigmoid()
        )
        self.dim_chg_gap4 = nn.Sequential(
            nn.Linear(self.g4, 2048),
            nn.Sigmoid()
        )
        self.dim_chg_gcn = nn.Sequential(
            nn.Linear(self.gcn, 2048),
            nn.Sigmoid()
        )

        self.w1 = nn.Parameter(torch.tensor(0.2))
        self.w2 = nn.Parameter(torch.tensor(0.2))
        self.w3 = nn.Parameter(torch.tensor(0.2))
        self.w4 = nn.Parameter(torch.tensor(0.2))
        self.w5 = nn.Parameter(torch.tensor(0.2))

        self.linear_classif1 = nn.Linear(768, 1200)
        self.linear_classif2 = nn.Linear(1200, FUSION_HIDDEN)
        self.linear_classif3 = nn.Linear(FUSION_HIDDEN, self.num_classes)
        
    def forward(self, input_v, input_q, data_x=None,edge_index=None, edge_weight=None, num_vertices=None, num_edges=None):

        # create placeholders for the VisualBERT, these necessary for special tasks (then there are not only 1)
        # num_vertices = int(num_vertices)
        # num_edges = int(num_edges)
        x_q = self.tokenizer(input_q, return_tensors='pt', padding='max_length', max_length=128)
        # reject a possible leading 0 dimension
        for key in x_q.keys():
          x_q[key].long().squeeze(0)
        x_v_1=self.layer1(input_v)
        x_v_2=self.layer2(x_v_1)
        x_v_3=self.layer3(x_v_2)
        x_v_4=self.layer4(x_v_3)

        x_v_1_avg = torch.mean(x_v_1, axis=(2,3))
        x_v_2_avg = torch.mean(x_v_2, axis=(2,3))
        x_v_3_avg = torch.mean(x_v_3, axis=(2,3))
        x_v_4_avg = torch.mean(x_v_4, axis=(2,3))

        input_ids = Variable(x_q['input_ids']).to(torch.device(f'cuda:{self.args.gpu}'))
        token_type_ids = Variable(x_q['token_type_ids']).to(torch.device(f'cuda:{self.args.gpu}'))
        attention_mask = Variable(x_q['attention_mask']).to(torch.device(f'cuda:{self.args.gpu}'))

        x_v_1=self.dim_chg_gap1(x_v_1_avg)
        x_v_2=self.dim_chg_gap3(x_v_3_avg)
        x_v_3=self.dim_chg_gap4(x_v_4_avg)
        x_v_4=self.dim_chg_gap2(x_v_2_avg)

        for i in range(data_x.shape[0]):
            if i==0:
                x_v_gcn = self.gcn_model(data_x[i][0:num_vertices[i]],edge_index[i][:,0:num_edges[i]], edge_weight[i][0:num_edges[i]]).unsqueeze(0)
            else:
                out_gcn = self.gcn_model(data_x[i][0:num_vertices[i]],edge_index[i][:,0:num_edges[i]], edge_weight[i][0:num_edges[i]]).unsqueeze(0)
                x_v_gcn = torch.cat((x_v_gcn,out_gcn),0)

        x_v_5 = self.dim_chg_gcn(x_v_gcn)

        stacked_x_v = torch.stack([self.w1*x_v_1, self.w2*x_v_2, self.w3*x_v_3, self.w4*x_v_4, self.w5*x_v_5],2)
        x_v_avg = torch.mean(stacked_x_v,2)
        x_v_max, _ = torch.max(stacked_x_v,2)
        x_v = torch.cat((x_v_max,x_v_avg), 1)
        B,N = x_v.shape
        x_v = x_v.view(B,1,N)
        visual_attention_mask = Variable(torch.ones(x_v.shape[:-1], dtype=torch.long)).to(torch.device(f'cuda:{self.args.gpu}'))
        visual_token_type_ids = Variable(torch.ones(x_v.shape[:-1], dtype=torch.long)).to(torch.device(f'cuda:{self.args.gpu}'))
        
        out = self.visual_bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                visual_embeds=x_v, visual_token_type_ids=visual_token_type_ids,
                                visual_attention_mask=visual_attention_mask)

        # don't know why, but used in the original visualBERT for vqa
        index_to_gather = attention_mask.sum(1) - 2
        index_to_gather = (
                index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, out.last_hidden_state.size(-1))
        )
        x = torch.gather(out.last_hidden_state, 1, index_to_gather).squeeze(1)
        
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        x = nn.Tanh()(x)
        x = self.linear_classif3(x)
        
        return x
    
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class VQAGAPGraphTransformer_caption_qbert_max_avg_pooled(nn.Module):
    def __init__(self, vocab_questions, vocab_answers, args, pretrained_resnet=False):
        super(VQAGAPGraphTransformer_caption_qbert_max_avg_pooled, self).__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.sentemb_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sent_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.sentemb_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        VISUAL_OUT_GAP = 3840
        VISUAL_OUT = 2048
        QUESTION_OUT = 2400
        FUSION_IN = 1200
        FUSION_HIDDEN = 256
        DROPOUT_V = 0.5
        DROPOUT_Q = 0.5
        DROPOUT_F = 0.5

        model = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')
        self.gcn_model = GraphTransformer(659, 128)
        self.gcn = 256
        # choose the number of pretrained layers
        config = VisualBertConfig(num_hidden_layers=12, visual_embedding_dim=4096)
        self.visual_bert = VisualBertModel(config)
        for key in self.visual_bert.state_dict().keys():
            self.visual_bert.state_dict()[key] = model.state_dict()[key]

        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)
        
        self.visual = torchmodels.resnet152(pretrained=pretrained_resnet)
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
            param.requires_grad = not pretrained_resnet
        for param in self.layer2.parameters():
            param.requires_grad = not pretrained_resnet
        for param in self.layer3.parameters():
            param.requires_grad = not pretrained_resnet
        for param in self.layer4.parameters():
            param.requires_grad = not pretrained_resnet
        
        self.g1 = 256
        self.g2 = 512
        self.g3 = 1024
        self.g4 = 2048

        self.dim_chg_gap1 = nn.Sequential(
            nn.Linear(self.g1, 2048),
            nn.Sigmoid()
        )
        self.dim_chg_gap2 = nn.Sequential(
            nn.Linear(self.g2, 2048),
            nn.Sigmoid()
        )
        self.dim_chg_gap3 = nn.Sequential(
            nn.Linear(self.g3, 2048),
            nn.Sigmoid()
        )
        self.dim_chg_gap4 = nn.Sequential(
            nn.Linear(self.g4, 2048),
            nn.Sigmoid()
        )
        self.dim_chg_gcn = nn.Sequential(
            nn.Linear(self.gcn, 2048),
            nn.Sigmoid()
        )

        self.w1 = nn.Parameter(torch.tensor(0.2))
        self.w2 = nn.Parameter(torch.tensor(0.2))
        self.w3 = nn.Parameter(torch.tensor(0.2))
        self.w4 = nn.Parameter(torch.tensor(0.2))
        self.w5 = nn.Parameter(torch.tensor(0.2))

        self.linear_classif1 = nn.Linear(1152, 1200)
        self.linear_classif2 = nn.Linear(1200, FUSION_HIDDEN)
        self.linear_classif3 = nn.Linear(FUSION_HIDDEN, self.num_classes)
        
    def forward(self, input_v, input_q, data_x=None,edge_index=None, edge_weight=None, num_vertices=None, num_edges=None, captions=None):

        # create placeholders for the VisualBERT, these necessary for special tasks (then there are not only 1)
        # num_vertices = int(num_vertices)
        # num_edges = int(num_edges)
        x_q = self.tokenizer(input_q, return_tensors='pt', padding='max_length', max_length=128)
        captions = list(captions[0])
        encoded_input = self.sent_tokenizer(captions, padding=True, truncation=True, return_tensors='pt').to(f"cuda:{self.args.gpu}")
        with torch.no_grad():
            model_output = self.sentemb_model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        x_c = sentence_embeddings
        
        # x_c = self.sentemb_model.encode(captions)
        
        # TODO: yet to concatenate the embeddings
        # reject a possible leading 0 dimension
        for key in x_q.keys():
          x_q[key].long().squeeze(0)
        x_v_1=self.layer1(input_v)
        x_v_2=self.layer2(x_v_1)
        x_v_3=self.layer3(x_v_2)
        x_v_4=self.layer4(x_v_3)

        x_v_1_avg = torch.mean(x_v_1, axis=(2,3))
        x_v_2_avg = torch.mean(x_v_2, axis=(2,3))
        x_v_3_avg = torch.mean(x_v_3, axis=(2,3))
        x_v_4_avg = torch.mean(x_v_4, axis=(2,3))

        input_ids = Variable(x_q['input_ids']).to(torch.device(f'cuda:{self.args.gpu}'))
        token_type_ids = Variable(x_q['token_type_ids']).to(torch.device(f'cuda:{self.args.gpu}'))
        attention_mask = Variable(x_q['attention_mask']).to(torch.device(f'cuda:{self.args.gpu}'))

        x_v_1=self.dim_chg_gap1(x_v_1_avg)
        x_v_2=self.dim_chg_gap3(x_v_3_avg)
        x_v_3=self.dim_chg_gap4(x_v_4_avg)
        x_v_4=self.dim_chg_gap2(x_v_2_avg)

        for i in range(data_x.shape[0]):
            if i==0:
                x_v_gcn = self.gcn_model(data_x[i][0:num_vertices[i]],edge_index[i][:,0:num_edges[i]], edge_weight[i][0:num_edges[i]]).unsqueeze(0)
            else:
                out_gcn = self.gcn_model(data_x[i][0:num_vertices[i]],edge_index[i][:,0:num_edges[i]], edge_weight[i][0:num_edges[i]]).unsqueeze(0)
                x_v_gcn = torch.cat((x_v_gcn,out_gcn),0)

        x_v_5 = self.dim_chg_gcn(x_v_gcn)

        stacked_x_v = torch.stack([self.w1*x_v_1, self.w2*x_v_2, self.w3*x_v_3, self.w4*x_v_4, self.w5*x_v_5],2)
        x_v_avg = torch.mean(stacked_x_v,2)
        x_v_max, _ = torch.max(stacked_x_v,2)
        x_v = torch.cat((x_v_max,x_v_avg), 1)
        B,N = x_v.shape
        x_v = x_v.view(B,1,N)
        visual_attention_mask = Variable(torch.ones(x_v.shape[:-1], dtype=torch.long)).to(torch.device(f'cuda:{self.args.gpu}'))
        visual_token_type_ids = Variable(torch.ones(x_v.shape[:-1], dtype=torch.long)).to(torch.device(f'cuda:{self.args.gpu}'))
        
        out = self.visual_bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                visual_embeds=x_v, visual_token_type_ids=visual_token_type_ids,
                                visual_attention_mask=visual_attention_mask)

        # don't know why, but used in the original visualBERT for vqa
        index_to_gather = attention_mask.sum(1) - 2
        index_to_gather = (
                index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, out.last_hidden_state.size(-1))
        )
        x = torch.gather(out.last_hidden_state, 1, index_to_gather).squeeze(1)
        stacked_x_captions = torch.cat((x, x_c), dim=1)
        x = self.dropoutF(stacked_x_captions)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        x = nn.Tanh()(x)
        x = self.linear_classif3(x)
        
        return x