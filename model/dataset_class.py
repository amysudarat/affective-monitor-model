# -*- coding: utf-8 -*-

import os
import ast
import pandas as pd
from torch.utils.data import Dataset


class AffectiveMonitorDataset(Dataset):
    """ Affective Monitor Dataset:
         Args:
            filepath (string): Path to data directory
            mode (string):  'FAC' is default mode to load vector of FAC unit 
                            'RAW' load raw data which are 1347 points of facial points cloud
            transform (callable,optional): optional transform to be applied on a sample
    """
    
    def __init__(self,filepath,mode='FAC',transform=None,fix_distance=False):
        """
        Args:
            filepath (string): Path to data directory
            mode (string):  'FAC' is default mode to load vector of FAC unit 
                            'RAW' load raw data which are 1347 points of facial points cloud
            transform (callable,optional): optional transform to be applied on a sample
        """
        # map pic index with arousal level and valence level
        self.label_lookup = self.load_label(filepath)
        self.fix_distance = fix_distance
        # load global FAPU
        self.global_fapu = self.load_FAPU(filepath)
        # load samples from csv file
        if mode == 'FAC':
            self.samples = self.load_dataframe_FAC(filepath)
        elif mode == 'RAW':
            self.samples = self.load_dataframe_raw(filepath)
        # option for data augmentation
        self.transform = transform
        
    
    def load_dataframe_FAC(self,path):
        """
        Read CSV file and convert it FAC unit
        """
        # create file path 
        filepaths = [os.path.join(path, "TestSubject"+str(i)+"\\FAP.txt") for i in range(1,3)]
        
        # initialize Total dataframe
        total = pd.DataFrame()
        # Loop through each Testsubject folder
        for i in range(len(filepaths)):
            
            face_df = pd.read_csv(filepaths[i],header=1,delimiter=",",
                                  quotechar=";",
#                                  index_col="PicIndex",
                                  skipinitialspace=True)   
            # set index column
            face_df = face_df.set_index("PicIndex")
            # convert string to tuple on pupil diameter column 
            try:
                face_df["PupilDiameter"] = pd.Series([ast.literal_eval(x) for x in face_df["PupilDiameter"]]) 
            except:   
                print("Pupil is off")
            # adjust FAPU if fix_distance is True, otherwise just go ahead and divide by the global FAPU
            if self.fix_distance:  
                self.FAPUlize(face_df,self.global_fapu[i],adjust=True)
            else:
                # convert FAP in FAPU using global fapu
                self.FAPUlize(face_df,fapu=self.global_fapu[i],adjust=False)
            # create face sample loop through each picture index
            for i in range(1,max(face_df.index.values)+1):
                # group sequence of face point
                face_per_picture = face_df.loc[i]
                face_FAP_per_picture = face_per_picture.iloc[:,0:19]
                face_FAP_in_sequence = []
                for j in range(face_FAP_per_picture.shape[0]):
                    face_FAP_in_sequence.append(list(face_FAP_per_picture.iloc[j]))
                # prepare pupil diameter
                pupils = list(face_per_picture.loc[:,"PupilDiameter"])
                # prepare illuminance
                illuminance = list(face_per_picture.loc[:,"Illuminance"])               
                # create one sample
                sample = {'faceFAP': face_FAP_in_sequence,
                          'PD': pupils,
                          'illuminance': illuminance,
                          'arousal': self.label_lookup.loc[i,'Arousal_mean(IAPS)'],
                          'valence': self.label_lookup.loc[i,'Valence_mean(IAPS)'] }
                # append prepared sample to total dataframe
                total = total.append(sample, ignore_index=True)
        return total

    def load_dataframe_raw(self,path):
        """
        Read CSV file and convert it to all points obtained
        """
        # create file path 
        filepaths = [os.path.join(path, "TestSubject"+str(i)+"\\Data.txt") for i in range(1,3)]
        
        # initialize Total dataframe
        total = pd.DataFrame()
        # Loop through each Testsubject folder
        for filepath in filepaths:
            face_df = pd.read_csv(filepath,header=2,
                                  delimiter=",",
                                  quotechar=";",
                                  index_col="PicIndex",
                                  skipinitialspace=True)
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
        filepath_label = os.path.join(path, "TestSubject1\\SAMrating.txt") 
        SAM_df = pd.read_csv(filepath_label,header=1,index_col="PictureIndex")
        return SAM_df
    
    def load_FAPU(self,path):
        # create file path 
        filepaths_fapu = [os.path.join(path, "TestSubject"+str(i)+"\\TestSubjectInfo.txt") for i in range(1,3)]
        
        # loop through each test subject
        subject_number = 0
        # initialize Total dataframe
        total = pd.DataFrame()
        for filepath in filepaths_fapu:
            subject_number += 1
            FAPU_df = pd.read_csv(filepath,header=6)
            total = total.append(FAPU_df,ignore_index=True)
        return total
    
    def FAPUlize(self,face_df,fapu,adjust=False):
        if adjust:            
            # convert data to fap unit for each frame
            for i in range(face_df.shape[0]):
                # adjust fapu based on distance
                fapu_per_frame = face_df.iloc[i,21:26]
                 # convert data to fap unit using global fapu
                 
                face_df.iloc[i]['31'] = face_df['31']/fapu_per_frame['ENS']
                face_df['32'] = face_df['32']/fapu_per_frame['ENS']
                face_df['35'] = face_df['35']/fapu_per_frame['ENS']
                face_df['36'] = face_df['36']/fapu_per_frame['ENS']
                face_df['37'] = face_df['37']/fapu_per_frame['ES']
                face_df['38'] = face_df['38']/fapu_per_frame['ES']
                face_df['19'] = face_df['19']/fapu_per_frame['IRSD']
                face_df['20'] = face_df['20']/fapu_per_frame['IRSD']
                face_df['41'] = face_df['41']/fapu_per_frame['ENS']
                face_df['42'] = face_df['42']/fapu_per_frame['ENS']
                face_df['61'] = face_df['61']/fapu_per_frame['ENS']
                face_df['62'] = face_df['62']/fapu_per_frame['ENS']
                face_df['59'] = face_df['59']/fapu_per_frame['MNS']
                face_df['60'] = face_df['60']/fapu_per_frame['MNS']
                face_df['53'] = face_df['53']/fapu_per_frame['MW']
                face_df['54'] = face_df['54']/fapu_per_frame['MW']
                face_df['5'] = face_df['5']/fapu_per_frame['MNS']
                face_df['4'] = face_df['4']/fapu_per_frame['MNS']
                face_df['3'] = face_df['3']/fapu_per_frame['MNS']
                
                
            
        else:           
            # convert data to fap unit using global fapu
            face_df['31'] = face_df['31']/fapu['ENS']
            face_df['32'] = face_df['32']/fapu['ENS']
            face_df['35'] = face_df['35']/fapu['ENS']
            face_df['36'] = face_df['36']/fapu['ENS']
            face_df['37'] = face_df['37']/fapu['ES']
            face_df['38'] = face_df['38']/fapu['ES']
            face_df['19'] = face_df['19']/fapu['IRSD']
            face_df['20'] = face_df['20']/fapu['IRSD']
            face_df['41'] = face_df['41']/fapu['ENS']
            face_df['42'] = face_df['42']/fapu['ENS']
            face_df['61'] = face_df['61']/fapu['ENS']
            face_df['62'] = face_df['62']/fapu['ENS']
            face_df['59'] = face_df['59']/fapu['MNS']
            face_df['60'] = face_df['60']/fapu['MNS']
            face_df['53'] = face_df['53']/fapu['MW']
            face_df['54'] = face_df['54']/fapu['MW']
            face_df['5'] = face_df['5']/fapu['MNS']
            face_df['4'] = face_df['4']/fapu['MNS']
            face_df['3'] = face_df['3']/fapu['MNS']
        

    def preprocess_pupil(self):
        pass
   
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        sample = self.samples.iloc[idx]
        if self.transform:
            sample= self.transform(sample)
        return sample


