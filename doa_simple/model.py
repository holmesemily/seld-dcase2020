# from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate, Flatten
# from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
# from keras.layers.recurrent import GRU
# from keras.layers.normalization import BatchNormalization
# from keras.models import Model
# from keras.layers.wrappers import TimeDistributed
# from keras.optimizers import Adam
# from keras.models import load_model
# import keras
import tensorflow as tf
from IPython import embed
import numpy as np
import os 
import random
import cls_feature_class
import cls_data_generator
import parameter
import time
import evaluation_metrics, SELD_evaluation_metrics
params = parameter.get_params(1)
from collections import deque

# tf.keras.backend.set_image_data_format('channels_first')

# constants
f_pool_size = [4, 4, 2]    
nb_cnn2d_filt = 64
dropout_rate = 0
t_pool_size = [int(0.1/0.02), 1, 1]
fnn_size=[128]

# spec_start = Input(shape=(3000,512))

model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(3000,128,4)),
        tf.keras.layers.Conv2D(filters=nb_cnn2d_filt, kernel_size=(5, 5), padding="same"),                   
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(5, 5),
        tf.keras.layers.Dropout(0.1),      
        # bloc
        tf.keras.layers.Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding="same"),                   
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(1, 1),
        tf.keras.layers.Dropout(0.1),   
        # bloc 
        tf.keras.layers.Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding="same"),                   
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(1, 1),
        tf.keras.layers.Dropout(0.1),   
        # fc
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Reshape((600, 1600)),
        tf.keras.layers.Dense(64),              
        tf.keras.layers.Dense(3)              
        ])
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

# train and test datasets
split_test = 60
feat_dir = '../dataset/myfeatures/foa_dev_norm'
label_dir = '../dataset/myfeatures/foa_dev_label'
data_dir = '../dataset/myfeatures/sets'
test_splits = [1]
val_splits = [2]
train_splits = [[3, 4, 5, 6]]
batchsize = 70
filenames_list = os.listdir(feat_dir)
nb_total_batches = 4
feature_seq_len = 600
feature_batch_seq_len = batchsize*feature_seq_len
nb_mel_bins = 128
nb_ch = 4


def split_in_seqs(data, seq_len):
        if len(data.shape) == 1:
            if data.shape[0] % seq_len:
                data = data[:-(data.shape[0] % seq_len), :]
            data = data.reshape((data.shape[0] // seq_len, seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % seq_len:
                data = data[:-(data.shape[0] % seq_len), :]
            data = data.reshape((data.shape[0] // seq_len, seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % seq_len:
                data = data[:-(data.shape[0] % seq_len), :, :]
            data = data.reshape((data.shape[0] // seq_len, seq_len, data.shape[1], data.shape[2]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

def generate():
        while 1:
                random.shuffle(filenames_list)

                # Ideally this should have been outside the while loop. But while generating the test data we want the data
                # to be the same exactly for all epoch's hence we keep it here.
                circ_buf_feat = deque()
                circ_buf_label = deque()

                file_cnt = 0
                for i in range(nb_total_batches):
                        # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                        # circular buffer. If not keep refilling it.
                        while len(circ_buf_feat) < feature_batch_seq_len:
                                temp_feat = np.load(os.path.join(feat_dir, filenames_list[file_cnt]))
                                for row_cnt, row in enumerate(temp_feat):
                                        circ_buf_feat.append(row)
                                file_cnt = file_cnt + 1
                        
                        # Read one batch size from the circular buffer
                        feat = np.zeros((feature_batch_seq_len, nb_mel_bins * nb_ch))
                        for j in range(feature_batch_seq_len):
                                feat[j, :] = circ_buf_feat.popleft()
                        feat = np.reshape(feat, (feature_batch_seq_len, nb_mel_bins, nb_ch))

                        # Split to sequences
                        feat = split_in_seqs(feat, feature_seq_len)
                        feat = np.transpose(feat, (0, 3, 1, 2))

                        yield feat, label


for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
        print('---------------------------------------------------------------------------------------------------')

