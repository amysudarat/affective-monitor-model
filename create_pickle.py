# -*- coding: utf-8 -*-

from model.dataset_class import AffectiveMonitorDataset
import utils



#############--------- Select test subjects to include ----------##########
n = 50
subjects = [i for i in range(1,n+1)]

##############--------- Generate face_dataset getting raw PD ----------##########
#face_dataset = AffectiveMonitorDataset("C:\\Users\\dspcrew\\affective-monitor-model\\data",
#                                           subjects=subjects,
#                                           transform=ToTensor())
    
#face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\ExperimentData",
#                                       subjects=subjects,
#                                       fix_PD=False)
#    
face_dataset = AffectiveMonitorDataset("E:\\Research\\ExperimentData",
                                           subjects=subjects,
                                           fix_PD=False,
                                           convert_label=False)

##############--------- Generate pickle file based on generateing face_dataset ----------##########
utils.save_object(face_dataset, "data_1_50_fixPD_Label_False.pkl")
