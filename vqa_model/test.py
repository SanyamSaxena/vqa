#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sylvain
"""

import matplotlib
matplotlib.use('Agg')

from models import model
import VQALoader
from tqdm import tqdm
import VocabEncoder
import torchvision.transforms as T
import torch
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

import pickle
import os
import datetime
from shutil import copyfile
import argparse

print(os.getpid())

def test(model, test_dataset, batch_size, num_epochs, learning_rate, modeltype, args, Dataset='HR'):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
    criterion = torch.nn.CrossEntropyLoss()#weight=weights)
        
    testLoss = []
    
    if Dataset == 'HR':
        accPerQuestionType = {'area': [], 'presence': [], 'count': [], 'comp': []}
    else:
        accPerQuestionType = {'rural_urban': [], 'presence': [], 'count': [], 'comp': []}

    OA = []
    AA = []

    with torch.no_grad():
        RSVQA.eval()
        runningLoss = 0
        if Dataset == 'HR':
            countQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'area': 0}
            rightAnswerByQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'area': 0}
        else:
            countQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'rural_urban': 0}
            rightAnswerByQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'rural_urban': 0}

        for i, data in enumerate(tqdm(test_loader)):
            question, answer, image, type_str, image_original = data                
            if args.model == 'VQAGAP_qbert_Model' or args.model == 'VQAGAP_bert_Model' or args.model == 'VQAGAP_qbert_dca_Model' or args.model == 'VQAGAP_qbert_Model_finetune' or args.model == 'VQAGAP_bert_Model_finetune' or args.model == 'VQAGAP_qbert_dca_Model_finetune':
                answer = Variable(answer.long()).to(torch.device(f'cuda:{args.gpu}')).resize_(len(question))
            else:
                question = Variable(question.long()).to(torch.device(f'cuda:{args.gpu}'))
                answer = Variable(answer.long()).to(torch.device(f'cuda:{args.gpu}')).resize_(question.shape[0])
            image = Variable(image.float()).to(torch.device(f'cuda:{args.gpu}'))
            if modeltype == 'MCB':
                pred, att_map = RSVQA(image,question)
            else:
                pred = RSVQA(image,question)
            loss = criterion(pred, answer)
            if args.model == 'VQAGAP_qbert_Model' or args.model == 'VQAGAP_bert_Model' or args.model == 'VQAGAP_qbert_dca_Model'  or args.model == 'VQAGAP_qbert_Model_finetune' or args.model == 'VQAGAP_bert_Model_finetune' or args.model == 'VQAGAP_qbert_dca_Model_finetune':
                runningLoss += loss.cpu().item() * len(question)
            else:
                runningLoss += loss.cpu().item() * question.shape[0]
            
            answer = answer.cpu().numpy()
            pred = np.argmax(pred.cpu().detach().numpy(), axis=1)
            for j in range(answer.shape[0]):
                countQuestionType[type_str[j]] += 1
                if answer[j] == pred[j]:
                    rightAnswerByQuestionType[type_str[j]] += 1
                    
        testLoss.append(runningLoss / len(test_dataset))
        print('test loss: %.3f' % (testLoss[0]))
                        
        numQuestions = 0
        numRightQuestions = 0
        currentAA = 0
        for type_str in countQuestionType.keys():
            if countQuestionType[type_str] > 0:
                accPerQuestionType[type_str].append(rightAnswerByQuestionType[type_str] * 1.0 / countQuestionType[type_str])
            numQuestions += countQuestionType[type_str]
            numRightQuestions += rightAnswerByQuestionType[type_str]
            currentAA += accPerQuestionType[type_str][0]
            
        OA.append(numRightQuestions *1.0 / numQuestions)
        AA.append(currentAA * 1.0 / 4)
        print('OA: %.3f' % (OA[0]))
        print('AA: %.3f' % (AA[0]))
        

if __name__ == '__main__':    
    disable_log = True
    ratio_images_to_use = 1
    modeltype = 'Simple'

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--bs',
                        type=int, default=70)
    parser.add_argument('--gpu',
                        type=int, default=6)
    parser.add_argument('--lr',
                        type=float, default=0.00001)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    batch_size = args.bs
    num_epochs = 1
    learning_rate = args.lr
    Dataset = args.dataset
    gpu = args.gpu

    if Dataset == 'LR':
        data_path = '../data/lrdataset/'
        allquestionsJSON = os.path.join(data_path, 'all_questions.json')
        allanswersJSON = os.path.join(data_path, 'all_answers.json')
        questionstestJSON = os.path.join(data_path, 'LR_split_test_questions.json')
        answerstestJSON = os.path.join(data_path, 'LR_split_test_answers.json')
        imagestestJSON = os.path.join(data_path, 'LR_split_test_images.json')
        images_path = os.path.join(data_path, 'data/')
    else:
        data_path = '../data/hrdataset/'
        images_path = os.path.join(data_path, 'Data/')
        allquestionsJSON = os.path.join(data_path, 'USGSquestions.json')
        allanswersJSON = os.path.join(data_path, 'USGSanswers.json')
        questionstestJSON = os.path.join(data_path, 'USGS_split_test_questions.json')
        answerstestJSON = os.path.join(data_path, 'USGS_split_test_answers.json')
        imagestestJSON = os.path.join(data_path, 'USGS_split_test_images.json')
    encoder_questions = VocabEncoder.VocabEncoder(allquestionsJSON, questions=True)
    if Dataset == "LR":
        encoder_answers = VocabEncoder.VocabEncoder(allanswersJSON, questions=False, range_numbers = True)
    else:
        encoder_answers = VocabEncoder.VocabEncoder(allanswersJSON, questions=False, range_numbers = False)

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.ToTensor(),            
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
      ])
    
    if Dataset == 'LR':
        patch_size = 256
    else:
        patch_size = 512  
         
    test_dataset = VQALoader.VQALoader(images_path, imagestestJSON, questionstestJSON, answerstestJSON, encoder_questions, encoder_answers, args.model, train=False, ratio_images_to_use=ratio_images_to_use, transform=transform, patch_size = patch_size)
    
    model_folder = f'../data/dump-models/{Dataset}'
    model_path = f'{model_folder}/ModelBestRSVQA_{args.model}.pth'

    if modeltype == 'MCB':
        RSVQA = MCBModel.VQAModel(encoder_questions.getVocab(), encoder_answers.getVocab()).to(torch.device(f'cuda:{args.gpu}'))
    else:
        if args.model == 'VQAModel':
            RSVQA = model.VQAModel(encoder_questions.getVocab(), encoder_answers.getVocab(), args, input_size = patch_size).to(torch.device(f'cuda:{args.gpu}'))
        elif args.model == 'VQAGAPModel':
            RSVQA = model.VQAGAPModel(encoder_questions.getVocab(), encoder_answers.getVocab(), args,  input_size = patch_size).to(torch.device(f'cuda:{args.gpu}'))
        elif args.model == 'VQAGAPModel_finetune':
            RSVQA = model.VQAGAPModel_finetune(encoder_questions.getVocab(), encoder_answers.getVocab(), args,  input_size = patch_size).to(torch.device(f'cuda:{args.gpu}'))
        elif args.model == 'VQAGAP_qbert_Model':
            RSVQA = model.VQAGAP_qbert_Model(encoder_questions.getVocab(), encoder_answers.getVocab(), args,  input_size = patch_size).to(torch.device(f'cuda:{args.gpu}'))
        elif args.model == 'VQAGAP_qbert_Model_finetune':
            RSVQA = model.VQAGAP_qbert_Model_finetune(encoder_questions.getVocab(), encoder_answers.getVocab(), args,  input_size = patch_size).to(torch.device(f'cuda:{args.gpu}'))
        elif args.model == 'VQAGAP_bert_Model':
            RSVQA = model.VQAGAP_bert_Model(encoder_questions.getVocab(), encoder_answers.getVocab(), args,  input_size = patch_size).to(torch.device(f'cuda:{args.gpu}'))
        elif args.model == 'VQAGAP_bert_Model_finetune':
            RSVQA = model.VQAGAP_bert_Model_finetune(encoder_questions.getVocab(), encoder_answers.getVocab(), args,  input_size = patch_size).to(torch.device(f'cuda:{args.gpu}'))
        elif args.model == 'VQAGAP_qbert_dca_Model':
            RSVQA = model.VQAGAP_qbert_dca_Model(encoder_questions.getVocab(), encoder_answers.getVocab(), args,  input_size = patch_size).to(torch.device(f'cuda:{args.gpu}'))
        elif args.model == 'VQAGAP_qbert_dca_Model_finetune':
            RSVQA = model.VQAGAP_qbert_dca_Model_finetune(encoder_questions.getVocab(), encoder_answers.getVocab(), args,  input_size = patch_size).to(torch.device(f'cuda:{args.gpu}'))
        
    RSVQA.load_state_dict(torch.load(model_path))
    RSVQA = test(RSVQA, test_dataset, batch_size, num_epochs, learning_rate, modeltype, args, Dataset)
    
    
