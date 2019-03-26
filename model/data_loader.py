# -*- coding: utf-8 -*-

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from dataset_class import AffectiveMonitorDataset
import matplotlib.pyplot as plt
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
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with open(filename, 'rb') as input:
        data = pickle.load(input)
        return data

#def update_plot(i,data,scat)

if __name__ == "__main__":
    # FAP is loaded by default
    # how many subjects to load
#    n = 4
#    subjects = [i for i in range(1,n+1)]
#    face_dataset = AffectiveMonitorDataset("C:\\Users\\dspcrew\\affective-monitor-model\\data",subjects=subjects)
##    face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\ExperimentData",subjects=[1])
#    face_dataset = AffectiveMonitorDataset("E:\\Research\\ExperimentData",subjects=[1])
##    data = face_dataset[:]
#    # save face_dataset to pikle file
#    save_object(face_dataset, "data_testsub1_4.pkl")
#    del face_dataset
    
#    face_dataset = load_object("./../data_testsub1_4.pkl")
#    check_pupil(face_dataset[50])
    
    
    
    
    
#face_dataset = AffectiveMonitorDataset("C:\\Users\\dspcrew\\affective-monitor-model\\data",
#                                           subjects=subjects,
#                                           transform=ToTensor())
#    
#    face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\ExperimentData",
#                                           subjects=subjects,
#                                           transform=ToTensor())
#    
#    face_dataset = AffectiveMonitorDataset("E:\\Research\\ExperimentData",
#                                           subjects=subjects,
#                                           transform=ToTensor())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



