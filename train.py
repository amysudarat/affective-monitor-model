# -*- coding: utf-8 -*-

"""
contains main loop for training
"""

import torch
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from model.dataset_class import AffectiveMonitorDataset
from model.net_valence import myLSTM_valence
from model.net_arousal import myLSTM_arousal


def train_valence(pickle_file="data_1_50_toTensor.pkl",learning_rate=0.03):
    
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
    batch_size = 100
    n_iters = 1000
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
#    output dimension = 5

    input_dim = 19
    hidden_dim = 100
    layer_dim = 1
    output_dim = 5
    # Number of steps to unroll
    seq_dim = 100 
    
    num_epochs = int(n_iters/ (len(train_sampler)/batch_size))
#    num_epochs = 1
    
    # Instantiate Model class
    model = myLSTM_valence(input_dim,hidden_dim,layer_dim,output_dim)
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Instantiate Loss class
    criterion = nn.CrossEntropyLoss()
    
    # Instantiate Optimizer Class
#    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)
    
    # training loop
    iteration = 0
    iter_num = []
    loss_list = []
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            FAPs = data['FAP']
            labels = data['Valence']
            
            # Cast labels to float 
            labels = labels.long()
            # Cast input to Float (Model weight is set to Float by Default)
            FAPs = FAPs.float()
            
            # Load input vector as tensors 
            if torch.cuda.is_available():
                FAPs = FAPs.view(-1,seq_dim,input_dim).cuda()
                labels = labels.cuda()
            else:
                FAPs = FAPs.view(-1,seq_dim,input_dim)
  
            # Set existing torch with gradient accumation abilities
            FAPs.requires_grad = True                          
   
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
            if iteration%100 == 0:
                correct = 0
                total = 0
                
                # Iterate through test dataset
                for i, data in enumerate(test_loader):
                    FAPs = data['FAP']
                    labels = data['Valence']
                    
                    # Cast labels to float 
                    labels = labels.long()
                    # Cast input to Float
                    FAPs = FAPs.float()
                    
                    # Load input vector as tensors 
                    if torch.cuda.is_available():
                        FAPs = FAPs.view(-1,seq_dim,input_dim).cuda()
                        labels = labels.cuda()
                    else:
                        FAPs = FAPs.view(-1,seq_dim,input_dim)                  
                   
                    # Set existing torch 
                    FAPs.requires_grad = True
                    
                    # Forward pass only to get logits/output
                    outputs = model(FAPs)
                    
                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data,1)
                    
                    # Total number of labels (sum of batches)
                    total = total + labels.size(0)
                    
                    # total accuracy prediction
                    if torch.cuda.is_available():
                        correct = correct + (predicted.cpu() == labels.cpu()).sum()
                    else:
                        correct = correct + (predicted == labels).sum()
                
                accuracy = 100 * (correct.item()/total)
                
                iter_num.append(iteration)
                loss_list.append(loss.item())
                
                # print Loss
                print("Iteration: {}. Loss: {}. Accuracy: {}".format(iteration,loss.item(),accuracy))
                
    # Plot Graph
    plt.plot(iter_num,loss_list)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss")
    plt.show()
    
def train_arousal(pickle_file="data_1_4_toTensor.pkl",learning_rate=0.01):
    
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
    batch_size = 100
    n_iters = 3500
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
#    output dimension = 5

    input_dim = 1
    hidden_dim = 100
    layer_dim = 1
    output_dim = 5
    # Number of steps to unroll
    seq_dim = 100 
    
    num_epochs = int(n_iters/ (len(train_sampler)/batch_size))
#    num_epochs = 1
    
    # Instantiate Model class
    model = myLSTM_arousal(input_dim,hidden_dim,layer_dim,output_dim)
    
    # GPU configuration
    if torch.cuda.is_available():
        model.cuda()
    
    # Instantiate Loss class
    criterion = nn.CrossEntropyLoss()
    
    # Instantiate Optimizer Class
#    learning_rate = 0.05
    optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)
    
    # training loop
    iteration = 0
    iter_num = []
    loss_list = []
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            PDs = data['PD']
            labels = data['Arousal']
#            labels = labels*10
            
            # Cast input to Float (Model weight is set to Float by Default)
            PDs = PDs.float()
            # Cast labels to float 
            labels = labels.long()           
            
            # Load input vector as tensors 
            if torch.cuda.is_available():
                PDs = PDs.view(-1,seq_dim,input_dim).cuda()
                labels = labels.cuda()
            else:
                PDs = PDs.view(-1,seq_dim,input_dim)
  
            # Set existing torch with gradient accumulation abilities
            PDs.requires_grad = True                          
   
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            
            # Forward pass to get output/logits
            # output.size() --> 100,4
            outputs = model(PDs)
            
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs,labels)
            
            # Getting gradients w.r.t. parameters
            loss.backward()
            
            # Updating parameters
            optimizer.step()
            
            iteration = iteration+1
            
            # Calculate accuracy every 1000 step
            if iteration%100 == 0:
                correct = 0
                total = 0
                
                # Iterate through test dataset
                for i, data in enumerate(test_loader):
                    PDs = data['PD']
                    labels = data['Arousal']
                    
                    # Cast input to Float
                    PDs = PDs.float()
                    # Cast labels to float 
                    labels = labels.long()
                                       
                    # Load input vector as tensors 
                    if torch.cuda.is_available():
                        PDs = PDs.view(-1,seq_dim,input_dim).cuda()
                        labels = labels.cuda()
                    else:
                        PDs = PDs.view(-1,seq_dim,input_dim)                   
                    
                    # Set existing torch 
                    PDs.requires_grad = True
                    
                    # Forward pass only to get logits/output
                    outputs = model(PDs)
                    
                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data,1)
                    
                    # Total number of labels (sum of batches)
                    total = total + labels.size(0)
                    
                    # total accuracy prediction
                    if torch.cuda.is_available():
                        correct = correct + (predicted.cpu() == labels.cpu()).sum()
                    else:
                        correct = correct + (predicted == labels).sum()
                
                accuracy = 100 * (correct.item()/total)
                
                iter_num.append(iteration)
                loss_list.append(loss.item())
                
                # print Loss
                print("Iteration: {}. Loss: {}. Accuracy: {}".format(iteration,loss.item(),accuracy))
                
    # Plot Graph
    plt.plot(iter_num,loss_list)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
#    train_valence(pickle_file="data_1_50_toTensor.pkl",learning_rate=0.01)
    train_arousal(pickle_file="data_1_50_toTensor.pkl",learning_rate=0.07)


