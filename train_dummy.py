# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from model.dummy_dataset_class import DummyDataset
from model.dummy_dataset_class import ToTensor


class myLSTM(nn.Module):
    # define layers
    def __init__(self,input_dim,hidden_dim,layer_dim,output_dim):
        # define self for the base class (in this case it's nn)
        super(myLSTM,self).__init__()
        
        # define parameters
        self.hidden_dim = hidden_dim # hidden dimension
        self.layer_dim = layer_dim   # hidden layers
        
        # add first layer
        self.lstm = nn.LSTM(input_dim,hidden_dim,layer_dim,batch_first=True)
        
        # Softmax layer
        self.fc = nn.Linear(hidden_dim,output_dim)
    
    # define node and connection
    def forward(self,x):
        # Initialize hidden state with zeros (define nodes)
        if torch.cuda.is_available():
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, requires_grad=True).cuda()
        else:
            h0 = torch.zeros(self.layer_dim, x.size0(), self.hidden_dim, requires_grad=True)
        
        # Initialize cell state
        if torch.cuda.is_available():
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, requires_grad=True).cuda()
        else:
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, requires_grad=True)
        
        # call one time will do 100 time steps (define connection line)
        out, (hn,cn) = self.lstm(x,(h0,c0))
        
        # Index hidden state of last time step
        out = self.fc(out[:,-1,:])
        
        return out
    
def split_data_indices(data_length,split_ratio):
    
    # set random seed
    random_seed = 42
   
    # create indices of the dataframe
    dataset_idx = [i for i in range(data_length)]
    # calculate the index to split
    split_idx = int(np.floor(split_ratio*data_length))
    # shuffle the dataset, the command will randomly shuffle the array in-place
    np.random.seed(random_seed)
    np.random.shuffle(dataset_idx)
    # create train and test
    train_indices, test_indices = dataset_idx[split_idx:], dataset_idx[:split_idx]
    
    return train_indices, test_indices
    

def train_dummy(learning_rate=0.01,split_ratio=0.2):
    
     # Load Dataset
    dummy_dataset = DummyDataset(transform=ToTensor())
    
    # get split train, test indices
    train_indices, test_indices = split_data_indices(len(dummy_dataset),split_ratio=split_ratio)
    
    # Create subset of dataframe according to train_set and test_set indices
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create data loader
    # len(dataset)/batch_size means one epoch    
    batch_size = 100
    n_iter = 340*2
    num_epochs = int(np.floor(n_iter / (len(dummy_dataset)/batch_size)))
    
    train_loader = torch.utils.data.DataLoader(dummy_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dummy_dataset,
                                              batch_size=batch_size,
                                              sampler=test_sampler)
    
    # define model parameters
    input_dim = dummy_dataset[0]['data'].shape[0]
    hidden_dim = 100
    layer_dim = 1
    output_dim = 4
    seq_dim = 1
    
    # initialize model
    model = myLSTM(input_dim,hidden_dim,layer_dim,output_dim)
    if torch.cuda.is_available():
        model = model.cuda()
        
    # Instantiate Loss class
    criterion = nn.CrossEntropyLoss()
    
    # Instantiate Optimizer class
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    
    # training loop
    iteration = 0
    iter_num = []
    loss_list = []
   
    for epoch in range(num_epochs):
        for i, sample in enumerate(train_loader):
            graph = sample['data']
            label = sample['label']
            
            # cast the type of input tensor
            graph = graph.float()
            label = label.long()
            
            # Load input vector as tensors
            if torch.cuda.is_available():
                graph = graph.view(-1,seq_dim,input_dim).cuda()
                label = label.cuda()
            else:
                graph = graph.view(-1,seq_dim,input_dim)
            
            # set existing torch with gradient accumulation abilities
            graph.requires_grad = True
            
            # clear gradients w.r.t. parameters
            optimizer.zero_grad()
            
            # forward pass
            output = model(graph)
            
            # calculate loss
            loss = criterion(output,label)
            
            # getting gradient w.r.t. parameters
            loss.backward()
            
            # updating parameters
            optimizer.step()
            
            iteration = iteration+1
            
            # calculate accuracy 
            if iteration%100 == 0:
                correct = 0
                total = 0
                
                # run test set
                for i, sample in enumerate(test_loader):
                    
                    # obtain data
                    graph = sample['data']
                    label = sample['label']
                    
                    # cast the type of input tensor
                    graph = graph.float()
                    label = label.long()
                    
                    # prepare sample
                    if torch.cuda.is_available():
                        graph = graph.view(-1,seq_dim,input_dim).cuda()
                        label = label.cuda()
                    else:
                        graph = graph.view(-1,seq_dim,input_dim)
                    
                    graph.requires_grad = True
                    
                    # pass test data to model
                    output = model(graph)
                    
                    # get predictions from the maximum value
                    _, predicted = torch.max(output.data,1)
                    
                    # total number
                    total = total + label.size(0)
                    
                    # total accuracy prediction
                    if torch.cuda.is_available():
                        correct = correct + (predicted.cpu() == label.cpu()).sum()
                    else:
                        correct = correct + (predicted == label).sum()
                    
                accuracy = 100* (correct.item()/total)
                
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
    train_dummy(learning_rate=0.05)


