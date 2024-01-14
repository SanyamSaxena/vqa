#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sylvain
"""

# Classe définissant le jeu de donnée VQA au format pytorch

import os.path
import json
import random
# from transformers import VisualBertModel, BertTokenizer, VisualBertConfig

from skimage import io

from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage import color
# from skimage.future import graph
from skimage import graph
import re
import pickle

RANDOM_SEED = 42


class VQALoader(Dataset):
    def __init__(self, imgFolder, images_file, questions_file, answers_file, encoder_questions, encoder_answers, model_type, train=True, ratio_images_to_use = 1, transform=None, patch_size=512):
        self.transform = transform
        self.encoder_questions = encoder_questions
        self.encoder_answers = encoder_answers
        self.train = train
        self.model_type = model_type
        
        
        vocab = self.encoder_questions.words
        if 'bert' in self.model_type.split('_') or 'qbert' in self.model_type.split('_'):
            self.relationalWords = ['top', 'bottom', 'right', 'left']
        else:
            self.relationalWords = [vocab['top'], vocab['bottom'], vocab['right'], vocab['left']]
        
        with open(questions_file) as json_data:
            self.questionsJSON = json.load(json_data)
            
        with open(answers_file) as json_data:
            self.answersJSON = json.load(json_data)
            
        with open(images_file) as json_data:
            self.imagesJSON = json.load(json_data)
        
        images = [img['id'] for img in self.imagesJSON['images'] if img['active']]
        images = images[:int(len(images)*ratio_images_to_use)]
        self.images = np.empty((len(images), patch_size, patch_size, 3))
        
        self.len = 0
        for image in images:
            self.len += len(self.imagesJSON['images'][image]['questions_ids'])
        self.images_questions_answers = [[None] * 4] * self.len
        
        if model_type[:9]=="VQAGAPGCN":
            rag_path = f'{imgFolder}/rag_info.p'
            if not os.path.isfile(rag_path):
                self.rag_info = [None for _ in range(len(images))]
                index = 0
                max_vertices = 0
                max_edges = 0
                for i, image in tqdm(enumerate(images)):
                    img = io.imread(os.path.join(imgFolder, str(image)+'.tif'))
                    self.rag_info[i]=self.get_rag_info(img, imgFolder)
                    max_vertices = max(max_vertices, self.rag_info[i][0].shape[0])         
                    max_edges = max(max_edges, self.rag_info[i][1].shape[1])

                for rag in self.rag_info:
                    my_tensor = rag[0]
                    cur_vertices = my_tensor.shape[0]
                    rows_to_pad = max_vertices - cur_vertices
                    padding = torch.zeros((rows_to_pad, my_tensor.shape[1]), dtype=my_tensor.dtype)
                    rag[0] = torch.cat((my_tensor, padding), dim=0)
                    rag.append(cur_vertices)

                for rag in self.rag_info:
                    my_tensor = rag[1]
                    cur_edges = my_tensor.shape[1]
                    cols_to_pad = max_edges - cur_edges
                    padding = torch.zeros((my_tensor.shape[0], cols_to_pad), dtype=my_tensor.dtype)
                    rag[1] = torch.cat((my_tensor, padding), dim=1)
                    rag.append(cur_edges)

                file = open(rag_path, 'wb')
                pickle.dump(self.rag_info, file)
            else:
                file = open(rag_path, 'rb')
                self.rag_info = pickle.load(file)
                
        index = 0
        for i, image in enumerate(images):
            img = io.imread(os.path.join(imgFolder, str(image)+'.tif'))
            self.images[i, :, :, :] = img

            for questionid in self.imagesJSON['images'][image]['questions_ids']:
                question = self.questionsJSON['questions'][questionid]
            
                question_str = question["question"]
                # self.question_str = question_str
                type_str = question["type"]
                answer_str = self.answersJSON['answers'][question["answers_ids"][0]]['answer']
                if 'bert' in self.model_type.split('_') or 'qbert' in self.model_type.split('_'):
                    self.images_questions_answers[index] = [question_str, self.encoder_answers.encode(answer_str), i, type_str]
                else:
                    self.images_questions_answers[index] = [self.encoder_questions.encode(question_str), self.encoder_answers.encode(answer_str), i, type_str]
                index += 1
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        question = self.images_questions_answers[idx]
        img = self.images[question[2],:,:,:]
        
        if self.train and not self.relationalWords[0] in question[0] and not self.relationalWords[1] in question[0] and not self.relationalWords[2] in question[0] and not self.relationalWords[3] in question[0]:
            if random.random() < .5:
                img = np.flip(img, axis = 0)
            if random.random() < .5:
                img = np.flip(img, axis = 1)
            if random.random() < .5:
                img = np.rot90(img, k=1)
            if random.random() < .5:
                img = np.rot90(img, k=3)

        if self.transform:
            imgT = self.transform(img.copy())
        if 'bert' in self.model_type.split('_') or 'qbert' in self.model_type.split('_'):
            if self.train:
                if self.model_type[:9]=="VQAGAPGCN":
                    return question[0], np.array(question[1], dtype='int16'), imgT, question[3], self.rag_info[question[2]]
                return question[0], np.array(question[1], dtype='int16'), imgT, question[3]
            else:
                if self.model_type[:9]=="VQAGAPGCN":
                    # print(self.rag_info[question[2]][0].shape,self.rag_info[question[2]][1].shape, self.rag_info[question[2]][2])
                    return question[0], np.array(question[1], dtype='int16'), imgT, question[3], T.ToTensor()(img / 255), self.rag_info[question[2]]
                return question[0], np.array(question[1], dtype='int16'), imgT, question[3], T.ToTensor()(img / 255)
        else:
            if self.train:
                return np.array(question[0], dtype='int16'), np.array(question[1], dtype='int16'), imgT, question[3]
            else:
                return np.array(question[0], dtype='int16'), np.array(question[1], dtype='int16'), imgT, question[3], T.ToTensor()(img / 255)   
            
        
    def get_rag_info(self, img_init, dataset):
        img = cv.medianBlur(img_init, 3)
        x = np.array([[i for i in range(img.shape[0])] for _ in range(img.shape[1])])
        y = np.array([[j for i in range(img.shape[0])] for j in range(img.shape[1])])

        x = x.reshape((img.shape[0],img.shape[1],1))
        y = y.reshape((img.shape[0],img.shape[1],1))
        appended_img = np.dstack([img, x, y])
        flat_image = appended_img.reshape((-1,5))
        flat_image = np.float32(flat_image)
        
        # meanshift
        #hr 0.06 #lr 0.01 #lr 0.02 without median blur
        dataset = dataset.strip().split('/')[2]
        if dataset=="lrdataset":
            bandwidth = estimate_bandwidth(flat_image, quantile=.01, n_samples=3000)
        if dataset=="hrdataset":
            bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
        ms = MeanShift(bandwidth = bandwidth, max_iter=800, bin_seeding=True, min_bin_freq=100)
        ms.fit(flat_image)
        labeled=ms.labels_
        segments = np.unique(labeled)
        total = np.zeros((segments.shape[0], 5), dtype=float)
        count = np.zeros(total.shape, dtype=float)
        hsv_img = color.rgb2hsv(img)
        flat_image_hsv = hsv_img.reshape((-1,3))
        flat_image_hsv = np.float32(flat_image_hsv)
        hsv_total = np.zeros((segments.shape[0], 3), dtype=float)
        for i, label in enumerate(labeled):
            total[label] = total[label] + flat_image[i]
            hsv_total[label] = hsv_total[label] + flat_image_hsv[i]
            count[label] += 1
        avg = total/count
        avg = np.uint8(avg)    

        avg_hsv = hsv_total/count[:,:3]
        labels = labeled.reshape((img.shape[0],img.shape[1]))

        #rag
        g = graph.rag_mean_color(img, labels)
        num_segments = segments.shape[0]
        x = torch.zeros((num_segments+1,9))
        for i in range(num_segments):
            x[i][:5] = torch.from_numpy(avg[i])
            x[i][5:8] = torch.from_numpy(avg_hsv[i])
            x[i][-1] = torch.from_numpy(count[i][:1])

        # Convert the image from RGB to HSV
        
        # Calculate average HSV values
        x[num_segments][:3] = torch.from_numpy(np.mean(img, axis=(0, 1)).squeeze())
        x[num_segments][5:8] = torch.from_numpy(np.mean(hsv_img, axis=(0, 1)).squeeze())
        x[num_segments][3] = torch.tensor(img.shape[0]/2)
        x[num_segments][4] = torch.tensor(img.shape[1]/2)
        x[num_segments][8] = torch.tensor(img.shape[0]*img_init.shape[1])
        x = x.float()
        edge_tuple = list(g.edges())
        edge_index = [[i for i, j in edge_tuple],[j for i, j in edge_tuple]]
        for i in range(num_segments):
            edge_index[0].append(i)
            edge_index[1].append(num_segments)
        # return {'x':x,'adj_mat':torch.tensor(edge_index,dtype=torch.long)}
        return [x,torch.tensor(edge_index,dtype=torch.long)]