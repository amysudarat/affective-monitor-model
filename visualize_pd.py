# -*- coding: utf-8 -*-

from model.dataset_class import AffectiveMonitorDataset
import matplotlib.pyplot as plt
import utils



##############--------- Select test subjects to include ----------##########
n = 2
subjects = [i for i in range(1,n+1)]

##############--------- Generate face_dataset getting raw PD ----------##########
#face_dataset = AffectiveMonitorDataset("C:\\Users\\dspcrew\\affective-monitor-model\\data",
#                                           subjects=subjects,
#                                           transform=ToTensor())
    
face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\ExperimentData",
                                       subjects=subjects,
                                       fix_PD=False)
#    
#    face_dataset = AffectiveMonitorDataset("E:\\Research\\ExperimentData",
#                                           subjects=subjects,
#                                           transform=ToTensor())

# save to pickle
utils.save_object(face_dataset, "data_1_50_fixPD_False.pkl")

# load object if pickle file already exists
utils.load_object("data_1_50_fixPD_False.pkl")

##############--------- visualize PD signals ----------##########
def plot_pd_sample(sample,ax=None):
    
    if ax is None:        
        ax = plt.axes()
    
    PD = sample["PD"]
    arousal = sample["Arousal"].numpy()
    color = check_Q_color(arousal)
    ax.text(0, 0, str(arousal), bbox=dict(facecolor=color, alpha=0.5))
    ax.plot(PD.numpy(),color='black')
    
    # Turn off tick labels
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    
    if ax is None:       
        plt.show()
    return
    
def generate_array_samples(start_idx, stop_idx, pickle_file=pickle):
    face_dataset = load_object(pickle_file)
    
    return array_samples

def plot_pd_multi_samples(samples,start_idx,stop_idx):
    
    # get array of samples
    samples = generate_array_samples(start_idx,stop_idx,pickle_file)
    
    # ploting
#    plt.figure()
    fig, axes = plt.subplots(ncols=10,nrows=round(((stop_idx-start_idx)+1)/10),figsize=(14, 12))
    
    for i,ax in enumerate(axes.flatten()):
        plot_pd_sample(samples[i])
    # plot title of the figure
    fig.suptitle("Testsubject: "+str(subject_idx), fontsize=16)
    plt.show()
    return

def plot_subjects(subjects=[1,2],plot='PD'):
    """ subjects is the array of testsubject id"""
    for subject_idx in subjects:
        # [1,71,141,...]
        start_idx = ((subject_idx*70)-70)+1
        # [70,140,210,...]
        stop_idx = subject_idx*70
        plot_multi_samples(start_idx,stop_idx,plot=plot,subject_idx=subject_idx)
    return

############------------- plot ---------------#################

array_samples = []
for i in range(start_idx-1,stop_idx):
    array_samples.append(face_dataset[i])










