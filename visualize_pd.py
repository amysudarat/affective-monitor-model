# -*- coding: utf-8 -*-

import utils

# load object if pickle file already exists
face_dataset = utils.load_object("data_1_50_fixPD_False.pkl")


#############------------- plot ---------------#################

# function plot will load pickle automatically
subjects= [i for i in range(1,51)]
utils.plot_pd_subjects(subjects=subjects,pickle="data_1_50_fixPD_False.pkl")

utils.plot_pd_sample(face_dataset[8])









