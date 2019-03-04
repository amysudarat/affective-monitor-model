# -*- coding: utf-8 -*-

"""
contains main loop for training
"""

import torch
import utils
import numpy as np
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from model.dataset_class import AffectiveMonitorDataset
from model.net_valence import myLSTM_valence


def train_valence(pickle_file="data_1_4_toTensor.pkl"):
    
    # Load Dataset
#    n = 2
#    subjects = [i for i in range(1,n+1)]
#    face_dataset = AffectiveMonitorDataset("C:\\Users\\dspcrew\\affective-monitor-model\\data",subjects=subjects)
#    face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\affective-monitor-model\\data")
#    face_dataset = AffectiveMonitorDataset("E:\\Research\\affective-monitor-model\\data")
    face_dataset = utils.load_object(pickle_file)
    
    
    # split train and test dataset
    validation_split = 0.2
    random_seed = 42
    shuffle_dataset = True
    dataset_size = len(face_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split*dataset_size))
    
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(val_indices)
    
    # Make Dataset Iterable
    batch_size = 10
    n_iters = 84
    train_loader = torch.utils.data.DataLoader(face_dataset,
                                                batch_size=batch_size,
                                                sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(face_dataset,
                                                batch_size=batch_size,
                                                sampler=test_sampler)
    
    # Instatiate dimension parameters
#    100 time steps
#    Each time step: input dimension = 19
#    how many hidden layer: 1 hidden layer
#    output dimension = 4

    input_dim = 19
    hidden_dim = 100
    layer_dim = 1
    output_dim = 4
    # Number of steps to unroll
    seq_dim = 100 
    
    num_epochs = int(n_iters/ (len(train_sampler)/batch_size))
    
    # Instantiate Model class
    model = myLSTM_valence(input_dim,hidden_dim,layer_dim,output_dim)
    
    # Instantiate Loss class
    criterion = nn.CrossEntropyLoss()
    
    # Instantiate Optimizer Class
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)
    
    # training loop
    iteration = 0
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            FAPs = data['FAP']
            labels = data['Valence']
#            if i == 4:
#                print(data)
#                FAP = data['FAP']
#                PD = data['PD']
#                arousal = data['Arousal']
#                valence = data['Valence']
##                FAP, PD, arousal, valence = data.values()
#                print("at %d\n"% i)
#                print(FAP.size())
#                print(PD.size())
#                print(arousal.size())
#                print(valence.size())
            # Load input vector as tensors with gradient accumulation abilities
            FAPs = FAPs.view(-1,seq_dim,input_dim).requires_grad() 
#            
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            
            # Forward pass to get output/logits
            # output.size() --> 100,4
            outputs = model(FAPs)
            
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs,labels)
            
            # Getting gradients w.r.t. parameters
            loss.backward()
            
            # Updating parameters
            optimizer.step()
            
            iteration = iteration+1
            
            # Calculate accuracy every 1000 step
            if iteration%1000 == 0:
                correct = 0
                total = 0
                # Iterate through test dataset
                for FACunit, labels in test_loader:
                    # Load FACunit to a tensor with grad_require=True
                    FACunit = FACunit.view(-1,seq_dim,input_dim).requires_grad()
                    
                    # Forward pass only to get logits/output
                    outputs = model(FACunit)
                    
                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data,1)
                    
                    # Total number of labels (sum of batches)
                    total = total + labels.size(0)
                    
                    # Total correct predictions
                    correct = correct + (predicted == labels).sum()
                
                accuracy = 100 * (correct/total)
                
                # print Loss
                print("Iteration: {}. Loss: {}. Accuracy: {}".format(iteration,loss.item(),accuracy))


if __name__ == "__main__":
    train_valence()


