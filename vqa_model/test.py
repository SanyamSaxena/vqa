import matplotlib
matplotlib.use('Agg')

from models import model,test_model
import VQALoader
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

model=test_model.test_model(input_size=512)

x = torch.zeros(1,3,224,224)

out = model(x)