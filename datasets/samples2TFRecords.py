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



def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

samplesList_benign=list()
samplesList_malign=list()
namesList_benign=list()
namesList_malign=list()

samplesList_malign=np.load('samplesList_malign.npy')
samplesList_benign=np.load('samplesList_benign.npy')
namesList_malign=np.load('namesList_malign.npy')
namesList_benign=np.load('namesList_benign.npy')

percentage=1; #### Same file for train and validation
smt=np.shape(samplesList_benign)
num_benign=smt[0]
smt=np.shape(samplesList_malign)
num_malign=smt[0] # same number of malign and benign
num_train=int(percentage*num_benign);
#num_val=int((1-percentage)*num_benign);

# open the TFRecords file
train_filename = 'train.tfrecord'  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)


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
    writer.write(example.SerializeToString())
writer.close()

samplesListTest=list()
numSamplesListTest=list()
testLabels = pd.read_excel('labels_modified.xls', sheet_name='y_test')
testLabels.as_matrix()
size=np.shape(testLabels)
label=np.zeros((size[0],1))
for i in range(0,size[0]):#
    name=testLabels.iloc[i,0]
    label[i]=testLabels.iloc[i,1]
samplesListTest=list()
samplesListTest=np.load('samplesListTest.npy')
# open the TFRecords file
test_filename = 'test.tfrecord'  # address to save the TFRecords file
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
        example = tf.train.Example(features=tf.train.Features(feature=features))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
writer.close()
print np.shape(samplesList_benign)
print np.shape(samplesList_malign)
print np.shape(samplesListTest)
