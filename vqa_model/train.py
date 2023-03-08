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

def train(model, train_dataset, validate_dataset, batch_size, num_epochs, learning_rate, modeltype, args, Dataset='HR'):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,RSVQA.parameters()), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()#weight=weights)
        
    trainLoss = []
    valLoss = []
    min_valLoss = float('inf')
    best_epoch = 0
    
    if Dataset == 'HR':
        accPerQuestionType = {'area': [], 'presence': [], 'count': [], 'comp': []}
    else:
        accPerQuestionType = {'rural_urban': [], 'presence': [], 'count': [], 'comp': []}
    OA = []
    AA = []
    for epoch in range(num_epochs):
        print("Epoch Number:", epoch)
        with torch.no_grad():
            RSVQA.eval()
            runningLoss = 0
            if Dataset == 'HR':
                countQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'area': 0}
                rightAnswerByQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'area': 0}
            else:
                countQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'rural_urban': 0}
                rightAnswerByQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'rural_urban': 0}
            count_q = 0
            for i, data in enumerate(tqdm(validate_loader)):
                # if i % 1000 == 999:
                #     print(i/len(validate_loader))
                question, answer, image, type_str, image_original = data
                if args.model == 'VQAGAP_qbert_Model' or args.model == 'VQAGAP_bert_Model':
                    # question = Variable(question).cuda()
                    # question = question.cuda()
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
                if args.model == 'VQAGAP_qbert_Model' or args.model == 'VQAGAP_bert_Model':
                    runningLoss += loss.cpu().item() * len(question)
                else:
                    runningLoss += loss.cpu().item() * question.shape[0]
                
                answer = answer.cpu().numpy()
                pred = np.argmax(pred.cpu().detach().numpy(), axis=1)
                for j in range(answer.shape[0]):
                    countQuestionType[type_str[j]] += 1
                    if answer[j] == pred[j]:
                        rightAnswerByQuestionType[type_str[j]] += 1

                # if i % 50 == 2 and i < 999:
                fig1, f1_axes = plt.subplots(ncols=1, nrows=2)
                viz_img = T.ToPILImage()(image_original[0].float().data.cpu())
                if modeltype == 'MCB':
                    viz_att =  torch.squeeze(att_map[0,...]).data.cpu().numpy()
                if args.model == 'VQAGAP_qbert_Model' or args.model == 'VQAGAP_bert_Model':
                    viz_question = question[0]
                else:
                    viz_question = encoder_questions.decode(question[0].data.cpu().numpy())
                viz_answer = encoder_answers.decode([answer[0]])
                viz_pred = encoder_answers.decode([pred[0]])


                f1_axes[0].imshow(viz_img)
                f1_axes[0].axis('off')
                f1_axes[0].set_title(viz_question)
                if modeltype == 'MCB':
                    att_h = f1_axes[1].imshow(viz_att)
                #fig1.colorbar(att_h,ax=f1_axes[1])
                f1_axes[1].axis('off')
                f1_axes[1].set_title(viz_answer)
                text = f1_axes[1].text(0.5,-0.1,viz_pred, size=12, horizontalalignment='center',
                                            verticalalignment='center', transform=f1_axes[1].transAxes)
                path = f'../data/images/{Dataset}/{args.model}'
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                plt.savefig(f'{path}/{i}.png')
                plt.close(fig1)
                        
            valLoss.append(runningLoss / len(validate_dataset))
            print('epoch #%d val loss: %.3f' % (epoch, valLoss[epoch]))
                        
            numQuestions = 0
            numRightQuestions = 0
            currentAA = 0
            for type_str in countQuestionType.keys():
                if countQuestionType[type_str] > 0:
                    accPerQuestionType[type_str].append(rightAnswerByQuestionType[type_str] * 1.0 / countQuestionType[type_str])
                numQuestions += countQuestionType[type_str]
                numRightQuestions += rightAnswerByQuestionType[type_str]
                currentAA += accPerQuestionType[type_str][epoch]
                
            OA.append(numRightQuestions *1.0 / numQuestions)
            AA.append(currentAA * 1.0 / 4)

            print('epoch #%d OA: %.3f' % (epoch, OA[epoch]))
            print('epoch #%d AA: %.3f' % (epoch, AA[epoch]))
        

        RSVQA.train()
        runningLoss = 0
        for i, data in enumerate(tqdm(train_loader)):
            # if i % 1000 == 999:
            #     print(i/len(train_loader))
            question, answer, image, _ = data
            if args.model == 'VQAGAP_qbert_Model' or args.model == 'VQAGAP_bert_Model':
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.model == 'VQAGAP_qbert_Model' or args.model == 'VQAGAP_bert_Model':
                runningLoss += loss.cpu().item() * len(question)
            else:        
                runningLoss += loss.cpu().item() * question.shape[0]
            
        trainLoss.append(runningLoss / len(train_dataset))
        print('epoch #%d loss: %.3f' % (epoch, trainLoss[epoch]))
    
        if(valLoss[epoch] < min_valLoss):
            min_valLoss = valLoss[epoch]
            best_epoch = epoch
            model_folder=f'../data/dump-models/{Dataset}'
            isExist = os.path.exists(model_folder)
            if not isExist:
                os.makedirs(model_folder)
            torch.save(RSVQA.state_dict(), f'{model_folder}/ModelBestRSVQA_{args.model}.pth')
        
        model_folder=f'../data/dump-models/{Dataset}'
        isExist = os.path.exists(model_folder)
        if not isExist:
            os.makedirs(model_folder)
        torch.save(RSVQA.state_dict(), f'{model_folder}/ModelRunningRSVQA_{args.model}.pth')

    print('Best epoch #%d loss: %.3f' % (best_epoch, trainLoss[best_epoch]))
    print('Best epoch #%d val loss: %.3f' % (best_epoch, valLoss[best_epoch]))
    print('Best epoch #%d OA: %.3f' % (best_epoch, OA[best_epoch]))
    print('Best epoch #%d AA: %.3f' % (best_epoch, AA[best_epoch]))


if __name__ == '__main__':    
    disable_log = True
    ratio_images_to_use = 1
    modeltype = 'Simple'

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--bs',
                        type=int, default=70)
    parser.add_argument('--gpu',
                        type=int, default=0)
    parser.add_argument('--epochs',
                        type=int, default=150)
    parser.add_argument('--lr',
                        type=float, default=0.00001)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    batch_size = args.bs
    num_epochs = args.epochs
    learning_rate = args.lr
    Dataset = args.dataset
    gpu = args.gpu

    if Dataset == 'LR':
        data_path = '../data/lrdataset/'
        allquestionsJSON = os.path.join(data_path, 'all_questions.json')
        allanswersJSON = os.path.join(data_path, 'all_answers.json')
        questionsJSON = os.path.join(data_path, 'LR_split_train_questions.json')
        answersJSON = os.path.join(data_path, 'LR_split_train_answers.json')
        imagesJSON = os.path.join(data_path, 'LR_split_train_images.json')
        questionsvalJSON = os.path.join(data_path, 'LR_split_val_questions.json')
        answersvalJSON = os.path.join(data_path, 'LR_split_val_answers.json')
        imagesvalJSON = os.path.join(data_path, 'LR_split_val_images.json')
        images_path = os.path.join(data_path, 'data/')
    else:
        data_path = '../data/hrdataset/'
        images_path = os.path.join(data_path, 'Data/')
        allquestionsJSON = os.path.join(data_path, 'USGSquestions.json')
        allanswersJSON = os.path.join(data_path, 'USGSanswers.json')
        questionsJSON = os.path.join(data_path, 'USGS_split_train_questions.json')
        answersJSON = os.path.join(data_path, 'USGS_split_train_answers.json')
        imagesJSON = os.path.join(data_path, 'USGS_split_train_images.json')
        questionsvalJSON = os.path.join(data_path, 'USGS_split_val_questions.json')
        answersvalJSON = os.path.join(data_path, 'USGS_split_val_answers.json')
        imagesvalJSON = os.path.join(data_path, 'USGS_split_val_images.json')
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
         
    train_dataset = VQALoader.VQALoader(images_path, imagesJSON, questionsJSON, answersJSON, encoder_questions, encoder_answers, args.model, train=True, ratio_images_to_use=ratio_images_to_use, transform=transform, patch_size = patch_size)
    validate_dataset = VQALoader.VQALoader(images_path, imagesvalJSON, questionsvalJSON, answersvalJSON, encoder_questions, encoder_answers, args.model, train=False, ratio_images_to_use=ratio_images_to_use, transform=transform, patch_size = patch_size)
    
    
    if modeltype == 'MCB':
        RSVQA = MCBModel.VQAModel(encoder_questions.getVocab(), encoder_answers.getVocab()).to(torch.device(f'cuda:{args.gpu}'))
    else:
        if args.model == 'VQAModel':
            RSVQA = model.VQAModel(encoder_questions.getVocab(), encoder_answers.getVocab(), args, input_size = patch_size).to(torch.device(f'cuda:{args.gpu}'))
        elif args.model == 'VQAGAPModel':
            RSVQA = model.VQAGAPModel(encoder_questions.getVocab(), encoder_answers.getVocab(), args,  input_size = patch_size).to(torch.device(f'cuda:{args.gpu}'))
        elif args.model == 'VQAGAP_qbert_Model':
            RSVQA = model.VQAGAP_qbert_Model(encoder_questions.getVocab(), encoder_answers.getVocab(), args,  input_size = patch_size).to(torch.device(f'cuda:{args.gpu}'))
        elif args.model == 'VQAGAP_bert_Model':
            RSVQA = model.VQAGAP_bert_Model(encoder_questions.getVocab(), encoder_answers.getVocab(), args,  input_size = patch_size).to(torch.device(f'cuda:{args.gpu}'))
        # elif args.model == 'VQA_qbert_Model':
        #     train_dataset = VQALoader.VQA_BERT_Loader(images_path, imagesJSON, questionsJSON, answersJSON, encoder_questions, encoder_answers, train=True, ratio_images_to_use=ratio_images_to_use, transform=transform, patch_size = patch_size)
        #     validate_dataset = VQALoader.VQA_BERT_Loader(images_path, imagesvalJSON, questionsvalJSON, answersvalJSON, encoder_questions, encoder_answers, train=False, ratio_images_to_use=ratio_images_to_use, transform=transform, patch_size = patch_size)
        #     RSVQA = model.VQA_qbert_Model(encoder_questions.getVocab(), encoder_answers.getVocab(), input_size = patch_size).cuda()
    RSVQA = train(RSVQA, train_dataset, validate_dataset, batch_size, num_epochs, learning_rate, modeltype, args, Dataset)
    
    
