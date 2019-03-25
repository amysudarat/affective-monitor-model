# -*- coding: utf-8 -*-
import utils


pickle_file = "data_1_35_toTensor.pkl"

face_dataset = utils.load_object(pickle_file)

utils.plot_sample(face_dataset[19])

#utils.plot_FAP(face_dataset[3])

#subject_ids = [i for i in range(1,21)]
#
#utils.plot_subjects(subject_ids,plot='PD')

#utils.plot_multi_samples(1,70,plot="PD")
#utils.plot_multi_samples(71,140,plot="PD")
#utils.plot_multi_samples(141,210,plot="PD")


#utils.plot_multi_samples(1,70,plot="FAP")
#
#utils.plot_multi_samples(71,140,plot="PD")
#utils.plot_multi_samples(71,140,plot="FAP")
#
#utils.plot_FAP(face_dataset[90])
    
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
