# -*- coding: utf-8 -*-

import torch
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from model.dataset_class import AffectiveMonitorDataset



pickle_file = "data_1_35_toTensor.pkl"

face_dataset = utils.load_object(pickle_file)

utils.plot_multi_samples(1,70,plot="PD")
utils.plot_multi_samples(1,70,plot="FAP")

utils.plot_multi_samples(71,140,plot="PD")
utils.plot_multi_samples(71,140,plot="FAP")

utils.plot_FAP(face_dataset[90])
    
## split train and test dataset
#validation_split = 0.2
#random_seed = 42
#shuffle_dataset = True
#dataset_size = len(face_dataset)
#indices = list(range(dataset_size))
#split = int(np.floor(validation_split*dataset_size))
#
#if shuffle_dataset:
#    np.random.seed(random_seed)
#    np.random.shuffle(indices)
#train_indices, val_indices = indices[split:], indices[:split]
#train_sampler = SubsetRandomSampler(train_indices)
#test_sampler = SubsetRandomSampler(val_indices)
