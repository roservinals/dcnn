
"""Provides dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s*.tfrecord'

_NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A image with 16x16x6',
    'label': 'A single integer between 0 and 1',
}

_SPLITS_TO_SIZES = {
    'train': 264,
    'validation': 264,
}

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  file_pattern = split_name+'.tfrecord'#os.path.join(dataset_dir, file_pattern % split_name)
  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
      reader = tf.TFRecordReader
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/height':tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/width':tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/channels':tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
  }
  items_to_handlers = {
      'image': slim.tfexample_decoder.Image('image/encoded',shape=[16, 16, 6]),#shape=[16, 16, 1], channels=6)
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = slim.datase.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
