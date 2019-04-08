# -*- coding: utf-8 -*-

import os
import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class AffectiveMonitorDataset(Dataset):
    """ Affective Monitor Dataset:
         Args:
            filepath (string): Path to data directory
            
            mode (string):  'FAC' is default mode to load vector of FAC unit 
                            'RAW' load raw data which are 1347 points of facial points cloud
                            
            transform (callable,optional): optional transform to be applied on a sample
            
            fix_distance (bool): will FAPU get adjusted by distance or not. Default is 'False' 
            
            subjects (list): list of test subject number specified to be loaded
    """
    
    def __init__(self,filepath,mode='FAC',transform=None,fix_distance=False,subjects=None,fix_PD=True):
        """
        Args:
            filepath (string): Path to data directory
            mode (string):  'FAC' is default mode to load vector of FAC unit 
                            'RAW' load raw data which are 1347 points of facial points cloud
            transform (callable,optional): optional transform to be applied on a sample            
        """
        # determine how many test subjects data will be loaded
        if subjects:
            self.subjects = subjects
        else:
            self.subjects = [i for i in range(1,2)]
#        # removePLR or not
#        self.removePLR = removePLR
        # fix PD
        self.fix_PD = fix_PD
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
        filepaths = [os.path.join(path, "TestSubject"+str(i)+"\\FAP.txt") for i in self.subjects]
        
        # initialize Total dataframe
        total = pd.DataFrame()
        self.face_df = pd.DataFrame()
        # Loop through each Testsubject folder
        for i in range(len(filepaths)):
            
            face_df = pd.read_csv(filepaths[i],header=1,delimiter=",",
                                  quotechar=";",
#                                  index_col="PicIndex",
                                  skipinitialspace=True) 
            
            # set index column
            face_df = face_df.set_index("PicIndex")
            self.face_df = self.face_df.append(face_df)
            # fill pupil diameter of the first row by the second row
            face_df.iloc[0,face_df.columns.get_loc("PupilDiameter")] = face_df.iloc[1,face_df.columns.get_loc("PupilDiameter")]
            
            # convert string to tuple on pupil diameter column 
            # right now the way we handle sample is to replace the invalid value to (0,0)
            a_prev = (0,0)
            for i in range(face_df.shape[0]):
                try:
                    # convert string to tuple
                    a = ast.literal_eval(face_df.iloc[i,face_df.columns.get_loc("PupilDiameter")])
                    if self.fix_PD:
                        # handling missing value 
                        if a[0] < 2.5:
                            a[0] = a_prev[0]
                        if a[1] < 2.5:
                            a[1] = a_prev[1]
                    face_df.iat[i,face_df.columns.get_loc("PupilDiameter")] = a  
    #                    a_prev = a                        
                except:
                    a = a_prev
                    face_df.iat[i,face_df.columns.get_loc("PupilDiameter")] = a   
            
            if self.fix_PD:
                # find average (discard missing value)
                pd_sum = [0,0]
                count_left = 0
                count_right = 0
                for i in range(face_df.shape[0]):
                    a = face_df.iloc[i]['PupilDiameter']
                    if a[0] != 0:
                        pd_sum[0] = pd_sum[0]+a[0]
                        count_left += 1
                    if a[1] != 0:
                        pd_sum[1] = pd_sum[1]+a[1]
                        count_right += 1
                pd_avg = (pd_sum[0]/count_left,pd_sum[1]/count_right)
                
                # Pad (0,0) with average value
                for i in range(face_df.shape[0]):
                    a = face_df.iloc[i]['PupilDiameter']
                    b = list(a)
                    if b[0] == 0:
                        b[0] = pd_avg[0]
                    if b[1] == 0:
                        b[1] = pd_avg[1]
                    face_df.iat[i,face_df.columns.get_loc('PupilDiameter')] = b
                
                # Remove PLR
                
                illum = face_df['Illuminance'].values
                depth = face_df['Depth']
                pd_left, pd_right = self.tuple_to_list(face_df['PupilDiameter'])
                filtered_pupil_left = self.remove_PLR(pd_left,illum,10,15)
                filtered_pupil_right = self.remove_PLR(pd_right,illum,10,15)
                pupil_left_to_merge = filtered_pupil_left
                pupil_left_to_merge[:101] = pd_left[:101]
                pupil_right_to_merge = filtered_pupil_right
                pupil_right_to_merge[:101] = pd_right[:101]
                
            else:
                illum = face_df['Illuminance'].values
                pd_left, pd_right = self.tuple_to_list(face_df['PupilDiameter'])                
                pupil_left_to_merge = pd_left
                pupil_right_to_merge = pd_right
                
#            pupil_avg_to_merge = [x+y for x,y in zip(pupil_left_to_merge,pupil_right_to_merge)]
            # merge two eye sides signals together
            pupil_comb_to_merge = []
            for x,y in zip(pupil_left_to_merge,pupil_right_to_merge):
                if x > y:
                    pupil_comb_to_merge.append(x)
                else:
                    pupil_comb_to_merge.append(y)               
                           
            # adjust FAPU if fix_distance is True, otherwise just go ahead and divide by the global FAPU
            if self.fix_distance:  
                self.FAPUlize(face_df,self.global_fapu.iloc[0],adjust=True)
            else:
                # convert FAP in FAPU using global fapu
                self.FAPUlize(face_df,fapu=self.global_fapu.iloc[0],adjust=False)
            # create face sample loop through each picture index
#            self.face_df = face_df
            for i in range(1,max(face_df.index.values)+1):
                # number of rows per sample
                start = (i*100)-100# 0,100,200,...
                end = (i*100)  # 100,200,300,...
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
                          'PD_left_filtered': pupil_left_to_merge[start:end],
                          'PD_right_filtered': pupil_right_to_merge[start:end],
                          'PD_avg_filtered': pupil_comb_to_merge[start:end],
                          'illuminance': illuminance,
                          'arousal': self.label_lookup.loc[i,'Arousal_target'],
                          'valence': self.label_lookup.loc[i,'Valence_target'] }
                # append prepared sample to total dataframe
                total = total.append(sample, ignore_index=True)
        return total

    def load_dataframe_raw(self,path):
        """
        Read CSV file and convert it to all points obtained
        """
        # create file path 
        filepaths = [os.path.join(path, "TestSubject"+str(i)+"\\Data.txt") for i in self.subjects]
        
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
                          'arousal': self.label_lookup.loc[i,'Arousal_target'],
                          'valence': self.label_lookup.loc[i,'Valence_target'] }
                # append prepared sample to total dataframe
                total = total.append(sample, ignore_index=True)
        return total
    
    def load_label(self,path):
        filepath_label = os.path.join(path, "TestSubject1\\SAMrating.txt") 
        SAM_df = pd.read_csv(filepath_label,header=1,index_col="PictureIndex")
        
        # define function to convert the raw SAM to our 5 labels
        def convert_to_label(SAM):
            scale = 1
            target_scale = scale*((SAM-5)/4)
            
            if -1.0 <= target_scale < -0.6:
                target_scale = 1
            elif -0.6 <= target_scale < -0.2:
                target_scale = 2
            elif -0.2 <= target_scale < 0.2:
                target_scale = 3
            elif 0.2 <= target_scale < 0.6:
                target_scale = 4
            elif 0.6 <= target_scale <= 1:
                target_scale = 5
            return target_scale
        
        # Apply function convert_to_label to Arousal and Valence Columns
        SAM_df['Arousal_target'] = SAM_df['Arousal_mean(IAPS)'].apply(convert_to_label)
        SAM_df['Valence_target'] = SAM_df['Valence_mean(IAPS)'].apply(convert_to_label)        
        
        return SAM_df
    
    def load_FAPU(self,path):
        # create file path 
        filepaths_fapu = [os.path.join(path, "TestSubject"+str(i)+"\\TestSubjectInfo.txt") for i in self.subjects]
        
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
                depth = face_df.iloc[i,26]
                 # convert data to fap unit using global fapu
                face_df.iloc[i,face_df.columns.get_loc("31")] = face_df.iloc[i,face_df.columns.get_loc("31")]/fapu_per_frame['ENS']
                face_df.iloc[i,face_df.columns.get_loc("32")] = face_df.iloc[i,face_df.columns.get_loc("32")]/fapu_per_frame['ENS']
                face_df.iloc[i,face_df.columns.get_loc("35")] = face_df.iloc[i,face_df.columns.get_loc("35")]/fapu_per_frame['ENS']
                face_df.iloc[i,face_df.columns.get_loc("36")] = face_df.iloc[i,face_df.columns.get_loc("36")]/fapu_per_frame['ENS']
                face_df.iloc[i,face_df.columns.get_loc("37")] = face_df.iloc[i,face_df.columns.get_loc("37")]/fapu_per_frame['ES']
                face_df.iloc[i,face_df.columns.get_loc("38")] = face_df.iloc[i,face_df.columns.get_loc("38")]/fapu_per_frame['ES']
                face_df.iloc[i,face_df.columns.get_loc("19")] = face_df.iloc[i,face_df.columns.get_loc("19")]/fapu_per_frame['IRSD']
                face_df.iloc[i,face_df.columns.get_loc("20")] = face_df.iloc[i,face_df.columns.get_loc("20")]/fapu_per_frame['IRSD']
                face_df.iloc[i,face_df.columns.get_loc("41")] = face_df.iloc[i,face_df.columns.get_loc("41")]/fapu_per_frame['ENS']
                face_df.iloc[i,face_df.columns.get_loc("42")] = face_df.iloc[i,face_df.columns.get_loc("42")]/fapu_per_frame['ENS']
                face_df.iloc[i,face_df.columns.get_loc("61")] = face_df.iloc[i,face_df.columns.get_loc("61")]/fapu_per_frame['ENS']
                face_df.iloc[i,face_df.columns.get_loc("62")] = face_df.iloc[i,face_df.columns.get_loc("62")]/fapu_per_frame['ENS']
                face_df.iloc[i,face_df.columns.get_loc("59")] = face_df.iloc[i,face_df.columns.get_loc("59")]/fapu_per_frame['MNS']
                face_df.iloc[i,face_df.columns.get_loc("60")] = face_df.iloc[i,face_df.columns.get_loc("60")]/fapu_per_frame['MNS']
                face_df.iloc[i,face_df.columns.get_loc("53")] = face_df.iloc[i,face_df.columns.get_loc("53")]/fapu_per_frame['MW']
                face_df.iloc[i,face_df.columns.get_loc("54")] = face_df.iloc[i,face_df.columns.get_loc("54")]/fapu_per_frame['MW']
                face_df.iloc[i,face_df.columns.get_loc("5")]  = face_df.iloc[i,face_df.columns.get_loc("5")]/fapu_per_frame['MNS']
                face_df.iloc[i,face_df.columns.get_loc("4")]  = face_df.iloc[i,face_df.columns.get_loc("4")]/fapu_per_frame['MNS']
                face_df.iloc[i,face_df.columns.get_loc("3")]  = face_df.iloc[i,face_df.columns.get_loc("3")]/fapu_per_frame['MNS']          
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
        
        
    def my_lms(self,d,r,L,mu):
        e = np.zeros(d.shape)
        y = np.zeros(r.shape)
        w = np.zeros(L)
        
        for k in range(L,len(r)):
            x = r[k-L:k]
            y[k] = np.dot(x,w)
            e[k] = d[k]-y[k]
            w_next = w + (2*mu*e[k])*x
            w = w_next   
        return y, e, w
    
    def remove_PLR(self,pd,illum,n,mu):
        d = np.array(pd)
        d_norm = d / np.linalg.norm(d)
        illum_norm = illum / np.linalg.norm(illum)
        illum_norm = 1.2*illum_norm
        illum_norm = illum_norm - np.mean(illum_norm) + np.mean(d_norm)
        y, e, w = self.my_lms(d_norm,illum_norm,n,mu)
        return e
    
    def tuple_to_list(self,pd_tuple):
        # Unpack tuple to two lists
        L = []
        R=[]       
        for item in pd_tuple:
            L.append(item[0])
            R.append(item[1])
        return L,R
    
    def preprocess_pupil(self):
        pass
   
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        sample = self.samples.iloc[idx]
        if self.transform:
            sample= self.transform(sample)
        return sample

class ToTensor(object):
    """
        convert to tensor object
    """
    def __call__(self,sample):
        transformed_sample = {'FAP': torch.from_numpy(np.array(sample['faceFAP'])),
                              'PD': torch.from_numpy(np.array(sample['PD_avg_filtered'])),
                              'Arousal': torch.from_numpy(np.array(sample['arousal'])),
                              'Valence': torch.from_numpy(np.array(sample['valence'])) }
        return transformed_sample
    
class ToTensor_and_Skorch(object):
    
    def __init__(self,data,label):
        self.data = data
        self.label = label
    
    def __call__(self,sample):
        transformed_sample = {'FAP': torch.from_numpy(np.array(sample['faceFAP'])),
                              'PD': torch.from_numpy(np.array(sample['PD_avg_filtered'])),
                              'Arousal': torch.from_numpy(np.array(sample['arousal'])),
                              'Valence': torch.from_numpy(np.array(sample['valence'])) }
        return transformed_sample[self.data], transformed_sample[self.label]
        
    
    
    
    
    
    
    
