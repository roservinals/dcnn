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

MIN_DISTANCE_X=16
MIN_DISTANCE_Y=16
MIN_DISTANCE_Z=3
MAX_ITERATIONS=1000
NODULE_PERCENTAGE=50
MAX_SAMPLES=20
CHANNELS=MIN_DISTANCE_Z*2



def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

samplesListTrain=list()
namesListTrain=list()
labelsListTrain=list()

samplesListTrain=np.load('samplesListTrain')
namesListTrain=np.load('namesListTrain')
labelsListTrain=np.load('labelsListTrain')
num=np.shape(labelsListTrain)
num_train=num[0]

# open the TFRecords file
train_filename = 'train.tfrecord'  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(0,num_train):
    name=namesListTrain[i]
    x=samplesListTrain[i]
    label=labelsListTrain[i]
    # Create a feature
    features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(x.tostring()),
        'image/class/label': _int64_feature(label),
        'image/filename': _bytes_feature(str(name)),
        'image/height': _int64_feature(MIN_DISTANCE_X),
        'image/width': _int64_feature(MIN_DISTANCE_Y),
        'image/channels': _int64_feature(CHANNELS),
    })
    # # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=features))
    # # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()

############# TEST ############
samplesListTest=list()
namesListTest=list()
labelsListTest=list()
samplesListTest=np.load('samplesListTest')
namesListTest=np.load('namesListTTest')
labelsListTestnp.load('labelsListTest')
num=np.shape(labelsListTest)
num_test=num[0]
# open the TFRecords file
test_filename = 'test.tfrecord'  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)
for i in range(0,num_test):
    y=samplesListTest[i]
    name=namesListTest[i]
    # Create a feature
    features={
        'image/encoded': _bytes_feature(y.tostring()),
        'image/class/label': _int64_feature(labelsListTest[i]),
        'image/filename': _bytes_feature(str(name)),
        'image/height': _int64_feature(MIN_DISTANCE_X),
        'image/width': _int64_feature(MIN_DISTANCE_Y),
        'image/channels': _int64_feature(CHANNELS),
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()