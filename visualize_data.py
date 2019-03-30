# -*- coding: utf-8 -*-
import utils

pickle_file = "data_1_50_toTensor.pkl"

face_dataset = utils.load_object(pickle_file)
subject_ids = [i for i in range(1,51)]
utils.plot_subjects(subject_ids,plot='PD')
#utils.plot_subjects(subject_ids,plot='FAP')

#############---------- plot by testsubject ID -------------##########
#utils.plot_subjects([5,8,15],plot='PD')
#utils.plot_sample(face_dataset[19])
#utils.plot_FAP(face_dataset[3])

#############---------- plot multisample -------------##########
#utils.plot_multi_samples(1,70,plot="PD")
#utils.plot_multi_samples(1,70,plot="FAP")

#############---------- plot one sample -------------##########
#utils.plot_sample(face_dataset[19])
#utils.plot_FAP(face_dataset[90])
#utils.plot_PD(face_dataset[90])
