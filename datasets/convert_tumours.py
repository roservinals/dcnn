import numpy as np
from os import listdir
from os.path import isfile, join
import nrrd
import matplotlib.pyplot as plt
import random
from itertools import product, combinations
from math import fabs
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
import pandas as pd
import tensorflow as tf

MIN_DISTANCE_X=16;
MIN_DISTANCE_Y=16;
MIN_DISTANCE_Z=3;
MAX_ITERATIONS=1000;
NODULE_PERCENTAGE=50;
MAX_SAMPLES=20;
CHANNELS=MIN_DISTANCE_Z*2;


def box_overlap(point1,point2):
    '''Cond1.  If A's left face is to the right of the B's right face,
               -  then A is Totally to right Of B
                  CubeA.X2 < CubeB.X1
    Cond2.  If A's right face is to the left of the B's left face,
               -  then A is Totally to left Of B
                  CubeB.X2 < CubeA.X1
    Cond3.  If A's top face is below B's bottom face,
               -  then A is Totally below B
                  CubeA.Z2 < CubeB.Z1
    Cond4.  If A's bottom face is above B's top face,
               -  then A is Totally above B
                  CubeB.Z2 < CubeA.Z1
    Cond5.  If A's front face is behind B's back face,
               -  then A is Totally behind B
                  CubeB.Y2 < CubeA.Y1
    Cond6.  If A's left face is to the left of B's right face,
               -  then A is Totally to the right of B
                  CubeB.X2 < CubeA.X1'''
    x_cube=(int)(MIN_DISTANCE_X/4);
    y_cube=(int)(MIN_DISTANCE_Y/4);
    z_cube=(int)(MIN_DISTANCE_Z/4);
    cubeA_x1=point1[0]-x_cube;
    cubeA_x2=point1[0]+x_cube;
    cubeA_y1=point1[1]-y_cube;
    cubeA_y2=point1[1]+y_cube;
    cubeA_z1=point1[2]-z_cube;
    cubeA_z2=point1[2]+z_cube;
    cubeB_x1=point2[0]-x_cube;
    cubeB_x2=point2[0]+x_cube;
    cubeB_y1=point2[1]-y_cube;
    cubeB_y2=point2[1]+y_cube;
    cubeB_z1=point2[2]-z_cube;
    cubeB_z2=point2[2]+z_cube;

    cond1=cubeA_x2<cubeB_x1
    cond2=cubeB_x2<cubeA_x1
    cond3=cubeA_z2<cubeB_z1
    cond4=cubeB_z2<cubeA_z1
    cond5=cubeB_y2<cubeA_y1
    cond6=cubeB_x2<cubeA_x1
    return (cond1 or cond2 or cond3 or cond4 or cond5 or cond6) # if no overlap, returns true

def sampling(image,readdata):
    points=list()
    num_slices=readdata.shape[2];
    num_non_zeros=np.count_nonzero(readdata);
    image_size=[readdata.shape[0],readdata.shape[1],readdata.shape[2]]

    x_cube=(int)(MIN_DISTANCE_X/2);
    y_cube=(int)(MIN_DISTANCE_Y/2);
    z_cube=(int)(MIN_DISTANCE_Z/2);
    box_size=MIN_DISTANCE_X*MIN_DISTANCE_Y*MIN_DISTANCE_Z
    non_zero_indices=np.nonzero(readdata)
    num_samples=0;
    samples=list();
    # First point

    # Two conditions:
    # 1. Contains 50% of nodule
    # 2. Distance between points

    # Obtain pixels around point
    condition=True
    count=0
    while condition:
        count+=1;
        n=random.randint(0,num_non_zeros-1)
        point=[non_zero_indices[0][n],non_zero_indices[1][n],non_zero_indices[2][n]]
        subimage_nodule=0
        for i in range(-x_cube,x_cube):
            index_x=point[0]+i;
            for j in range(-y_cube,y_cube):
                index_y=point[1]+j;
                for k in range(-z_cube,z_cube+1):
                    index_z=point[2]+k;
                    if((index_x>=0 and index_x<readdata.shape[0])and(index_y>=0 and index_y<readdata.shape[1])and(index_z>=0 and index_z<readdata.shape[2])):
                        if(readdata[index_x][index_y][index_z]==1):
                            subimage_nodule+=1;
        if(subimage_nodule*100/box_size>=NODULE_PERCENTAGE):
            condition=False
            points.append(point)
            num_samples+=1
            tmp_image=np.zeros((MIN_DISTANCE_X,MIN_DISTANCE_Y,MIN_DISTANCE_Z*2))
            for i in range(-x_cube,x_cube):
                index_x=point[0]+i;
                for j in range(-y_cube,y_cube):
                    index_y=point[1]+j;
                    for k in range(-z_cube,z_cube):
                        index_z=point[2]+k;
                        # MIRRORING
                        if(index_x<0):
                            index_x=-index_x
                        if(index_x>=readdata.shape[0]):
                            index_x=readdata.shape[0]-1-(index_x-readdata.shape[0])
                        if(index_y<0):
                            index_y=-index_y
                        if(index_y>=readdata.shape[1]):
                            index_y=readdata.shape[1]-1-(index_y-readdata.shape[1])
                        if(index_z<0):
                            index_z=-index_z
                        if(index_z>=readdata.shape[2]):
                            index_z=readdata.shape[2]-1-(index_z-readdata.shape[2])
                        tmp_image[i+x_cube,j+y_cube,MIN_DISTANCE_Z+k+z_cube]=readdata[index_x,index_y,index_z]
                        tmp_image[i+x_cube,j+y_cube,k+z_cube]=image[index_x,index_y,index_z]
            samples.append(tmp_image)
        #this point is  not elegible anymore:
        a=non_zero_indices[0]
        b=non_zero_indices[1]
        c=non_zero_indices[2]
        a=np.delete(a,n)
        b=np.delete(b,n)
        c=np.delete(c,n)
        non_zero_indices=[a,b,c]
        num_non_zeros-=1

        if(count>MAX_ITERATIONS):
            condition=False;
            break
        if(num_non_zeros==0):
            condition=False;
            break

    condition=True
    while condition and num_non_zeros>0:
        count+=1;
        n=random.randint(0,num_non_zeros-1)
        point=[non_zero_indices[0][n],non_zero_indices[1][n],non_zero_indices[2][n]]
        # First condition
        c1=True
        for r in range(0,len(points)):
            if(box_overlap(point,points[r])==0):
                c1=False;
                #this point is  not elegible anymore:
                break;
        if(c1):
            subimage_nodule=0
            for i in range(0,MIN_DISTANCE_X):
                index_x=point[0]-i-x_cube;
                for j in range(0,MIN_DISTANCE_Y):
                    index_y=point[1]-j-y_cube;
                    for k in range(0,MIN_DISTANCE_Z):
                        index_z=point[2]-k-z_cube;
                        if((index_x>=0 and index_x<readdata.shape[0])and(index_y>=0 and index_y<readdata.shape[1])and(index_z>=0 and index_z<readdata.shape[2])):
                            if(readdata[index_x][index_y][index_z]==1):
                                subimage_nodule+=1;
            # second condition: % nodule
            if(subimage_nodule*100/box_size>=NODULE_PERCENTAGE):
                points.append(point)
                num_samples+=1
                tmp_image=np.zeros((MIN_DISTANCE_X,MIN_DISTANCE_Y,MIN_DISTANCE_Z*2))
                for i in range(-x_cube,x_cube):
                    index_x=point[0]+i;
                    for j in range(-y_cube,y_cube):
                        index_y=point[1]+j;
                        for k in range(-z_cube,z_cube):
                            index_z=point[2]+k;
                            # MIRRORING
                            if(index_x<0):
                                index_x=-index_x
                            if(index_x>=readdata.shape[0]):
                                index_x=readdata.shape[0]-1-(index_x-readdata.shape[0])
                            if(index_y<0):
                                index_y=-index_y
                            if(index_y>=readdata.shape[1]):
                                index_y=readdata.shape[1]-1-(index_y-readdata.shape[1])
                            if(index_z<0):
                                index_z=-index_z
                            if(index_z>=readdata.shape[2]):
                                index_z=readdata.shape[2]-1-(index_z-readdata.shape[2])
                            tmp_image[i+x_cube,j+y_cube,MIN_DISTANCE_Z+k+z_cube]=readdata[index_x,index_y,index_z]
                            tmp_image[i+x_cube,j+y_cube,k+z_cube]=image[index_x,index_y,index_z]
                samples.append(tmp_image)
        #this point is  not elegible anymore:
        a=non_zero_indices[0]
        b=non_zero_indices[1]
        c=non_zero_indices[2]
        a=np.delete(a,n)
        b=np.delete(b,n)
        c=np.delete(c,n)
        non_zero_indices=[a,b,c]
        num_non_zeros-=1
        if(count>MAX_ITERATIONS):
            condition=False;
            break
        if(num_non_zeros==0):
            condition=False;
            break
        if(num_samples>=MAX_SAMPLES):
            condition=False;
            break
    return samples;


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

samplesList_benign=list()
samplesList_malign=list()
namesList_benign=list()
namesList_malign=list()
trainLabels = pd.read_excel('labels_modified.xls', sheet_name='y_train')
trainLabels.as_matrix()
size=np.shape(trainLabels)
label=np.zeros((size[0],1))
for i in range(0,size[0]):
    name=trainLabels.iloc[i,0]
    label[i]=trainLabels.iloc[i,1]
    folder='DB/'+name
    for f in listdir(folder):
        if (isfile(join(folder, f))):
            if f.endswith('1.nrrd'):
                filename=f
            elif f.endswith('label.nrrd'):
                filelabel=f
    data, options = nrrd.read(folder+'/'+filename) # read the data back from file
    dataLabel, optionsLabel = nrrd.read(folder+'/'+filelabel)
    x=sampling(data,dataLabel)
    sz=np.shape(x)
    for j in range(0,sz[0]):
        if(label[i]==0):
            samplesList_benign.append(x[j][:][:][:])
            namesList_benign.append(name)
        else:
            samplesList_malign.append(x[j][:][:][:])
            namesList_malign.append(name)
np.save('samplesList_malign',samplesList_malign)
np.save('namesList_malign',namesList_malign)
np.save('samplesList_benign',samplesList_benign)
np.save('namesList_benign',namesList_benign)

########################## TRAIN ##########################
percentage=1; #### Same file for train and validation
smt=np.shape(samplesList_benign)
num_benign=smt[0]
smt=np.shape(samplesList_malign)
num_malign=smt[0] # same number of malign and benign
num_train=int(percentage*num_benign);
#num_val=int((1-percentage)*num_benign);

# open the TFRecords file
train_filename = 'train.tfrecords'  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)
print np.shape(samplesList_benign)
print np.shape(samplesList_malign)
for i in range(0,num_train):
    name=namesList_benign[i]
    x=samplesList_benign[i]
    sz=np.shape(x)
    # Create a feature
    features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(x.tostring()),
        'image/class/label': _int64_feature(0),
        'image/filename': _bytes_feature(str(name)),
        'image/height': _int64_feature(MIN_DISTANCE_X),
        'image/width': _int64_feature(MIN_DISTANCE_Y),
        'image/channels': _int64_feature(CHANNELS),
    })
    # # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=features))
    # # Serialize to string and write on the file
    writer.write(example.SerializeToString())
for i in range(0,num_train):
    name=namesList_malign[i]
    x=samplesList_malign[i]
    sz=np.shape(x)
    # Create a feature
    feature={
        'image/encoded': _bytes_feature(x.tostring()),
        'image/class/label': _int64_feature(1),
        'image/filename': _bytes_feature(str(name)),
        'image/height': _int64_feature(MIN_DISTANCE_X),
        'image/width': _int64_feature(MIN_DISTANCE_Y),
        'image/channels': _int64_feature(CHANNELS),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # # Create a feature
    # feature = {'train/label': _int64_feature(1),
    #             'train/id':  _bytes_feature(str(name)),
    #             'train/height':_int64_feature(MIN_DISTANCE_X),
    #             'train/width':_int64_feature(MIN_DISTANCE_Y),
    #             'train/channels':_int64_feature(CHANNELS),
    #            'train/image': _bytes_feature(x.tostring())}
    # # Create an example protocol buffer
    # example = tf.train.Example(features=tf.train.Features(feature=feature))
    # # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()

# ########################## VALIDATION ##########################
# # open the TFRecords file
# val_filename = 'val.tfrecords'  # address to save the TFRecords file
# writer = tf.python_io.TFRecordWriter(val_filename)
# for i in range(num_train,num_benign):
#     name=namesList_benign[i]
#     x=samplesList_benign[i]
#     sz=np.shape(x)
#     # Create a feature
#     feature = {'val/label': _int64_feature(0),
#                 'val/id': _bytes_feature(str(name)),
#                 'val/height':_int64_feature(MIN_DISTANCE_X),
#                 'val/width':_int64_feature(MIN_DISTANCE_Y),
#                 'val/channels':_int64_feature(CHANNELS),
#                'val/image': _bytes_feature(x.tostring())}
#     # Create an example protocol buffer
#     example = tf.train.Example(features=tf.train.Features(feature=feature))
#     # Serialize to string and write on the file
#     writer.write(example.SerializeToString())
# for i in range(num_train,num_benign):
#     name=namesList_malign[i]
#     x=samplesList_malign[i]
#     # Create a feature
#     feature = {'val/label': _int64_feature(1),
#                 'val/id': _bytes_feature(str(name)),
#                 'val/height':_int64_feature(MIN_DISTANCE_X),
#                 'val/width':_int64_feature(MIN_DISTANCE_Y),
#                 'val/channels':_int64_feature(CHANNELS),
#                'val/image': _bytes_feature(x.tostring())}
#     # Create an example protocol buffer
#     example = tf.train.Example(features=tf.train.Features(feature=feature))
#     # Serialize to string and write on the file
#     writer.write(example.SerializeToString())
# writer.close()


######################## TEST ############################
samplesListTest=list()
numSamplesListTest=list()
testLabels = pd.read_excel('labels_modified.xls', sheet_name='y_test')
testLabels.as_matrix()
size=np.shape(testLabels)
label=np.zeros((size[0],1))
for i in range(0,size[0]):#
    name=testLabels.iloc[i,0]
    label[i]=testLabels.iloc[i,1]
    folder='DB/'+name
    for f in listdir(folder):
        if (isfile(join(folder, f))):
            if f.endswith('1.nrrd'):
                filename=f
            elif f.endswith('label.nrrd'):
                filelabel=f
    data, options = nrrd.read(folder+'/'+filename) # read the data back from file
    dataLabel, optionsLabel = nrrd.read(folder+'/'+filelabel)
    x=sampling(data,dataLabel)
    samplesListTest.append(x)
    sz=np.shape(x)
    numSamplesListTest.append(sz[0])
print np.shape(samplesListTest)
np.save('samplesListTest',samplesListTest)
# open the TFRecords file
test_filename = 'test.tfrecords'  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)
for i in range(0,size[0]):
    name=testLabels.iloc[i,0]
    x=samplesListTest[i]
    sz=np.shape(x)
    for j in range(0,sz[0]):
        y=x[j]
        # Create a feature
        features={
            'image/encoded': _bytes_feature(y.tostring()),
            'image/class/label': _int64_feature(label[i]),
            'image/filename': _bytes_feature(str(name)),
            'image/height': _int64_feature(MIN_DISTANCE_X),
            'image/width': _int64_feature(MIN_DISTANCE_Y),
            'image/channels': _int64_feature(CHANNELS),
        }
        # feature = {'test/id': _bytes_feature(str(name)),
        #             'test/height':_int64_feature(MIN_DISTANCE_X),
        #             'test/width':_int64_feature(MIN_DISTANCE_Y),
        #             'test/channels':_int64_feature(CHANNELS),
        #             'test/image': _bytes_feature(y.tostring())}
        # # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=features))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
writer.close()
