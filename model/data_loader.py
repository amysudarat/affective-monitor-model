# -*- coding: utf-8 -*-

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
    
def plot_face(face):
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
        xyzn = zip(X,Y,Z)
        for j, xyz_ in enumerate(xyzn): 
            annotate3D(ax, s=str(j-1), xyz=xyz_, fontsize=10, xytext=(-3,3),
            textcoords='offset points', ha='right',va='bottom')   
        
        # label axis
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
       
        plt.show()
        
def show_face(face):
    face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\affective-monitor-model\\data",'RAW')
#    face_dataset = AffectiveMonitorDataset("E:\\Research\\affective-monitor-model\\data")
    data = face_dataset[0]
    face = data["facepoints"]
    plot_face(face[0])

#def update_plot(i,data,scat)

if __name__ == "__main__":
    # FAP is loaded by default
    face_dataset = AffectiveMonitorDataset("C:\\Users\\DSPLab\\Research\\affective-monitor-model\\data")
#    face_dataset = AffectiveMonitorDataset("E:\\Research\\affective-monitor-model\\data")
    data = face_dataset[0]
    faceFAC = data["faceFAP"]


