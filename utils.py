# -*- coding: utf-8 -*-

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import numpy as np
import model.dataset_class
from model.dataset_class import AffectiveMonitorDataset
from model.dataset_class import ToTensor
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)
        
def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)
    
def plot_face(face,annotate=False):
        """
        Args:
        face: list of face points cloud
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
#        face = list(face.iloc[0:1347])
        
#        ax.scatter(*zip(*face),c='r')   
        X = []
        Y=[]
        Z=[]
        for item in face:
            X.append(item[0])
            Y.append(item[1])
            Z.append(item[2])
        
        ax.scatter(X,Y,Z,c='r')
        # annotate each point
        if annotate:
            xyzn = zip(X,Y,Z)
            for j, xyz_ in enumerate(xyzn): 
                annotate3D(ax, s=str(j-1), xyz=xyz_, fontsize=10, xytext=(-3,3),
                textcoords='offset points', ha='right',va='bottom')   
        
        # label axis
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
       
        plt.show()
        
def show_face():
    face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\ExperimentData",mode='RAW',subjects=[1])
#    face_dataset = AffectiveMonitorDataset("E:\\Research\\ExperimentData",mode='RAW',subjects=[1])
    data = face_dataset[0]
    face = data["facepoints"]
    plot_face(face[0])
    
def plot_signal(data,title):
    plt.figure()
    plt.plot(data,'k',label=title)
    plt.legend()
    plt.show()
    
def check_pupil(data):
    pupil_left = data['PD_left_filtered']
    pupil_right = data['PD_right_filtered']
    pupil_avg = data['PD_avg_filtered']
    plt.figure()
    plt.plot(pupil_left,'y--',label='left')
    plt.plot(pupil_right,'b--',label='right')
    plt.plot(pupil_avg,'k',label='average')
    plt.legend()
    plt.show()
    

def save_object(obj, filename):
    with open("pkl/"+filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    return
        
def load_object(filename):
    with open("pkl/"+filename, 'rb') as input:
        data = pickle.load(input)
        return data

def create_pickle_tensor(pickle_name,n,path="C:\\Users\\DSPLab\\Research\\ExperimentData"):
    subjects = [i for i in range(1,n+1)]    
    face_dataset = AffectiveMonitorDataset(path,
                                           subjects=subjects,
                                           transform=ToTensor())        
    # save face_dataset to pikle file
    save_object(face_dataset, pickle_name)
    return

def create_pickle(pickle_name,n,path="C:\\Users\\DSPLab\\Research\\ExperimentData"):
    subjects = [i for i in range(1,n+1)]    
    face_dataset = AffectiveMonitorDataset(path,
                                           subjects=subjects)        
    # save face_dataset to pikle file
    save_object(face_dataset, pickle_name)
    return

def dataset_info(df):
    for i in df.columns:
        x = np.array(df.loc[0][i])
        print("%s : %s , %s"%(i,str(x.shape),type(x)))
    return
        
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 0: 
        return True
    return False

###############--------- visualize dataset sample ----------##########

def generate_array_samples(start_idx, stop_idx, pickle_file="data_1_50_toTensor.pkl"):
    face_dataset = load_object(pickle_file)
    array_samples = []
    for i in range(start_idx-1,stop_idx):
        array_samples.append(face_dataset[i])
    return array_samples

def check_Q_color(label):
    label = int(label)
    if label == 1:
        color = 'yellow'
    elif label == 2:
        color = 'green'
    elif label == 3:
        color = 'red'
    elif label == 4:
        color = 'blue'
    elif label == 5:
        color = 'black'
    return color

def plot_FAP(sample,ax=None):
    """plot FAP of one sample"""
    if ax is None:        
        plt.figure()
        ax = plt.axes()
    
    FAP = sample["FAP"].numpy()
    valence = sample["Valence"].numpy()
    
    color = check_Q_color(valence)
    
    for fap in FAP:
        ax.plot(fap,marker='o')
#    ax.set_title("color",color='black')
    ax.text(0, 0, str(valence), bbox=dict(facecolor=color, alpha=0.5))
    # Turn off tick labels
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    if ax is None:       
        plt.show()
    return

def plot_PD(sample,ax=None):
    
    if ax is None:        
#        plt.figure()
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

def plot_sample(sample):
    
    # extract data from sample
    pupil = sample['PD'].numpy()
    FAPs = sample['FAP'].numpy()
    
    arousal = sample["Arousal"].numpy()
    valence = sample["Valence"].numpy()
    color_arousal = check_Q_color(arousal)
    color_valence = check_Q_color(valence)
    
    # plot 1 row 2 column (pupil on left and FAP on right)
    plt.figure()
    plt.subplot(121)
    plt.plot(pupil,color='black')
    plt.text(0, 0, str(arousal), bbox=dict(facecolor=color_arousal, alpha=0.5))

    plt.title("PD")
    plt.subplot(122)
    for fap in FAPs:
        plt.plot(fap,marker='o')
    plt.text(0,0, str(valence), bbox=dict(facecolor=color_valence, alpha=0.5))
    plt.title("FAP")
    plt.show()
    return 
    
def plot_multi_samples(start_idx,stop_idx,plot='PD',subject_idx=None):
    
    # get array of samples
    samples = generate_array_samples(start_idx,stop_idx)
    
    # ploting
#    plt.figure()
    fig, axes = plt.subplots(ncols=10,nrows=round(((stop_idx-start_idx)+1)/10),figsize=(14, 12))
    
    for i,ax in enumerate(axes.flatten()):
        if plot == 'FAP':
            plot_FAP(samples[i],ax) 
        elif plot == 'PD':
            plot_PD(samples[i],ax)
        else:
            print("plot parameter is not valid")
            return
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

def plot_FAP_linear(sample):
    FAPs = sample['FAP'].numpy()
    
    plt.figure()
    for fap in FAPs:
        plt.plot(fap)
    return


##############--------- visualize PD signals ----------##########
def plot_pd_sample(sample,ax=None):
    
    if ax is None:        
        ax = plt.axes()
        ax.set_title("left= red, right= blue, merge= black, depth= green")
        ax.grid(True)
    
#    pd_left = sample["PD_left_filtered"]
#    zero_line = [0 for i in range(len(pd_left))]
#    pd_right = sample["PD_right_filtered"]
    pd_merge = sample["PD_avg_filtered"]
#    depth = sample["depth"]
    arousal = sample["arousal"]    
    ax.text(0, pd_merge[0], str(arousal), bbox=dict(facecolor='red', alpha=0.5))
    ax.plot(pd_merge,'k',linewidth=3)
    ax.grid(True)
#    ax.plot(pd_left,'--r')
#    ax.plot(pd_right,'--b')
#    ax.plot(zero_line,'y')
    
#    ax.plot(depth,'g')
    
#    if ax is None:   
        
#        plt.show()
        
#    else:
        # Turn off tick labels
#        ax.xaxis.set_visible(False)
#        ax.yaxis.set_visible(False)
    return

    
def generate_array_samples_pd(start_idx, stop_idx, pickle_file):
    face_dataset = load_object(pickle_file)
    array_samples = []
    for i in range(start_idx-1,stop_idx):
        array_samples.append(face_dataset[i])
    return array_samples

def plot_pd_multi_samples(start_idx,stop_idx,subject_idx=None,pickle="data_1_50_fixPD_False.pkl"):
    
    # get array of samples
    samples = generate_array_samples(start_idx,stop_idx,pickle_file=pickle)
    
    # ploting
#    plt.figure()
    fig, axes = plt.subplots(ncols=10,nrows=round(((stop_idx-start_idx)+1)/10),figsize=(14, 12))
    
    for i,ax in enumerate(axes.flatten()):
        plot_pd_sample(samples[i],ax)
    # plot title of the figure
    fig.suptitle("Testsubject: "+str(subject_idx), fontsize=16)
    plt.show()
    return

def plot_pd_subjects(subjects=[1,2],pickle="data_1_50_fixPD_False.pkl"):
    """ subjects is the array of testsubject id"""
    for subject_idx in subjects:
        # [1,71,141,...]
        start_idx = ((subject_idx*70)-70)+1
        # [70,140,210,...]
        stop_idx = subject_idx*70
        plot_pd_multi_samples(start_idx,stop_idx,subject_idx=subject_idx,pickle=pickle)
    return

def print_pdf(figs,filename):
    """save figures to pdf file"""
    pdf = matplotlib.backends.backend_pdf.PdfPages("pdf/"+filename+".pdf")
    i = 0
    for fig in figs: ## will open an empty extra figure :(
        pdf.savefig( fig )
        i+=1
        print("printing: "+str(i))
    pdf.close()
    return

#def update_plot(i,data,scat)

#if __name__ == "__main__":
##    # FAP is loaded by default
##    # how many subjects to load
#    n = 35
#    subjects = [i for i in range(1,n+1)]
##    face_dataset = AffectiveMonitorDataset("C:\\Users\\dspcrew\\affective-monitor-model\\data",subjects=subjects)
##    face_dataset = AffectiveMonitorDataset("C:\\Users\\dspcrew\\affective-monitor-model\\data",subjects=subjects,transform=ToTensor())
#    
#    face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\ExperimentData",
#                                           subjects=subjects,
#                                           transform=ToTensor())
#    
##    face_dataset = AffectiveMonitorDataset("E:\\Research\\ExperimentData",
##                                           subjects=subjects,
##                                           transform=ToTensor())
###    data = face_dataset[:]
#    # save face_dataset to pikle file
#    save_object(face_dataset, "data_1_35_toTensor.pkl")
##    del face_dataset
#    
##    face_dataset = load_object("data_1_4_toTensor.pkl")
    
    
    
    
    
    
    
    