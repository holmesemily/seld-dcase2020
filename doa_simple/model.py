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

# train_data = np.zeros((240, 3000, 512))
# train_label = np.zeros((240, 600, 3))
# test_data = np.zeros((60, 3000, 512))
# test_label = np.zeros((60, 600, 3))

# test_list = random.sample(os.listdir(feat_dir), k=split_test)
# train_list = [x for x in os.listdir(feat_dir) if x not in test_list]

# for index, filename in enumerate(train_list):
#     print(index)
#     cur_file = os.path.join(feat_dir, filename)
#     filename = filename + ".csv"
#     cur_label = os.path.join(label_dir, filename)
#     a = np.genfromtxt(cur_file, delimiter=',', dtype=float)
#     b = np.genfromtxt(cur_label, delimiter=',', dtype=float)
#     train_data[index, :, :] = a
#     train_label[index, :, :] = b

# train_data = np.reshape(train_data, (240, 3000*512))
# train_label = np.reshape(train_label, (240, 600*3))
# np.savetxt(os.path.join(data_dir, "train_data.csv"), train_data, delimiter = ",")  
# np.savetxt(os.path.join(data_dir, "train_label.csv"), train_label, delimiter = ",")  

# print("saved")

# for index, filename in enumerate(test_list):
#     print(index)
#     cur_file = os.path.join(feat_dir, filename)
#     filename = filename + ".csv"
#     cur_label = os.path.join(label_dir, filename)
#     a = np.genfromtxt(cur_file, delimiter=',', dtype=float)
#     b = np.genfromtxt(cur_label, delimiter=',', dtype=float)
#     test_data[index, :, :] = a
#     test_label[index, :, :] = b

# test_data = np.reshape(test_data, (60, 3000*512))
# test_label = np.reshape(test_label, (60, 600*3))
# np.savetxt(os.path.join(data_dir, "test_data.csv"), test_data, delimiter = ",")  
# np.savetxt(os.path.join(data_dir, "test_label.csv"), test_label, delimiter = ",")  

# print("saved")

epochs = 10
batchsize = 240

print("Loading data...")

# find a way to load progressively
train_data = np.genfromtxt(os.path.join(data_dir, "train_data.csv"), delimiter=',', dtype=float, max_rows=batchsize)
train_label = np.genfromtxt(os.path.join(data_dir, "train_label.csv"), delimiter=',', dtype=float, max_rows=batchsize)
# test_data = np.genfromtxt(os.path.join(data_dir, "test_data.csv"), delimiter=',', dtype=float)
# test_label= np.genfromtxt(os.path.join(data_dir, "test_label.csv"), delimiter=',', dtype=float)

# train_data = np.reshape(train_data, (batchsize, 3000, 512))
train_data = np.reshape(train_data, (batchsize, 3000, 128, 4))
train_label = np.reshape(train_label, (batchsize, 600, 3))
# test_data = np.reshape(test_data, (60, 3000, 512))
# test_label = np.reshape(test_label, (60, 600, 3))



print("Start training...")
# tf.compat.v1.disable_eager_execution()
model.fit(train_data, train_label, epochs=epochs)
# def make_iterator(dataset):
#     iterator = iter(dataset)
#     next_val = iterator.get_next()

#     with tf.compat.v1.keras.backend.get_session().as_default() as sess:
#         while True:
#             *inputs, labels = sess.run(next_val)
#             yield inputs, labels

# def configure_for_performance(ds):
#   ds = ds.cache()
#   ds = ds.shuffle(buffer_size=1000)
#   ds = ds.batch(batchsize)
#   ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
#   return ds

# batch_train = tf.data.experimental.CsvDataset([os.path.join(data_dir, "train_data.csv"), os.path.join(data_dir, "train_label.csv")], 
#                                               [tf.float32, ])
# batch_valid = tf.data.experimental.CsvDataset([os.path.join(data_dir, "test_data.csv"), os.path.join(data_dir, "test_label.csv")], tf.float32)

# print(batch_train)
# print(batch_train.shape)

# batch_train=configure_for_performance(batch_train)
# batch_valid=configure_for_performance(batch_valid)

# itr_train = make_iterator(batch_train)
# itr_valid = make_iterator(batch_valid)

# model.fit(
#     x=batch_train, validation_data=batch_valid, validation_steps=batchsize,
#     epochs=epochs, steps_per_epoch=2, verbose=2)

