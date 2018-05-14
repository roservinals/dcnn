import tensorflow as tf
import tumours

slim = tf.contrib.slim
DATA_DIR=''
# Selects the 'validation' dataset.
dataset = tumours.get_split('train', DATA_DIR)

# Creates a TF-Slim DataProvider which reads the dataset in the background
# during both training and testing.
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])
