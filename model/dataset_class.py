# -*- coding: utf-8 -*-

import os
import ast
import pandas as pd
from torch.utils.data import Dataset


class AffectiveMonitorDataset(Dataset):
    """ Affective Monitor Dataset """
    
    def __init__(self,filepath,transform=None):
        """
        Args:
            filepath (string): Path to data directory
            transform (callable,optional): optional transform to be applied on a sample.
        """
        # map pic index with arousal level and 
        self.label_lookup = self.load_label(filepath)
        # load samples from csv file
        self.samples = self.load_dataframe(filepath)
        # option for data augmentation
        self.transform = transform
    

    def load_dataframe(self,path):
        """
        Read CSV file and convert it to appropriate type
        """
        # create file path 
        filepaths = [os.path.join(path, "TestSubject"+str(i)+"\\Data.txt") for i in range(2,4)]
        
        # initialize Total dataframe
        total = pd.DataFrame()
        # Loop through each Testsubject folder
        for filepath in filepaths:
            face_df = pd.read_csv(filepath,header=2,delimiter=",",quotechar=";",index_col="PicIndex",skipinitialspace=True)
            # convert string to tuple
            for i in range(0,1347):
                face_df.iloc[:,i] = pd.Series([ast.literal_eval(x) for x in face_df.iloc[:,i]]) 
            # create face sample loop through each picture index
            for i in range(1,4):
                # group sequence of face point
                face_per_picture = face_df.loc[i]
                face_points_per_picture = face_per_picture.iloc[:,0:1347]
                face_points_in_sequence = []
                for j in range(face_points_per_picture.shape[0]):
                    face_points_in_sequence.append(list(face_points_per_picture.iloc[j]))
                # prepare pupil diameter
                pupils = list(face_per_picture.loc[:,"PupilDiameter"])
                # prepare illuminance
                illuminance = list(face_per_picture.loc[:,"Illuminance"])               
                # create one sample
                sample = {'facepoints': face_points_in_sequence,
                          'PD': pupils,
                          'illuminance': illuminance,
                          'arousal': self.label_lookup.loc[i,'Arousal_mean(IAPS)'],
                          'valence': self.label_lookup.loc[i,'Valence_mean(IAPS)'] }
                # append prepared sample to total dataframe
                total = total.append(sample, ignore_index=True)
        return total
    
    def load_label(self,path):
        filepath_label = os.path.join(path, "TestSubject2\\SAMrating.txt") 
        SAM_df = pd.read_csv(filepath_label,header=1,index_col="PictureIndex")
        return SAM_df

    def preprocess_pupil(self):
        pass
   
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        sample = self.samples.iloc[idx]
        if self.transform:
            sample= self.transform(sample)
        return sample


