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

# tf.keras.backend.set_image_data_format('channels_first')

# constants
f_pool_size = [4, 4, 2]    
nb_cnn2d_filt = 72
dropout_rate = 0
t_pool_size = [int(0.1/0.02), 1, 1]
fnn_size=[128]

# spec_start = Input(shape=(3000,512))

# compute if prediction is within 20deg of grond truth for all batch
# output average (1 = 100% of predictions were within 20deg)
def is_within_20deg(y_true, y_pred):
    delta_theta = tf.abs(tf.subtract(y_true[1], y_pred[1]))
    return tf.reduce_mean(tf.cast(tf.less_equal(delta_theta, 20.0), tf.float32))

model1 = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(300,64,4)),
        tf.keras.layers.Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding="same"),                   
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.1),      
        # bloc
        tf.keras.layers.Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding="same"),                   
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.1),   
        # bloc 
        tf.keras.layers.Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding="same"),                   
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(1, 1),
        tf.keras.layers.Dropout(0.1),   
        # fc
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Reshape((600, 144)),
        tf.keras.layers.Dense(64),              
        tf.keras.layers.Dense(2)              
        ])

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(300,64,4)),
    tf.keras.layers.Reshape((600, 128)),
    tf.keras.layers.Dense(32),              
    tf.keras.layers.Dense(2)  
    ])
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=['msle'], metrics=['accuracy', is_within_20deg])

quant_aware_model = tensorflow_model_optimization.quantization.keras.quantize_model(model)

# train and test datasets
split_test = 60
feat_dir = '../dataset/myfeatures/foa_dev'
feat_dir_norm = '../dataset/myfeatures/foa_dev_norm'
label_dir = '../dataset/myfeatures/foa_dev_label'
data_dir = '../dataset/myfeatures/sets'
dump_dir = '../dataset/myfeatures/dump'
filenames_list = os.listdir(feat_dir)
epoches = 25


print("Generate dataset...")
x_train = np.zeros((240,300,256))
y_train = np.zeros((240,600,2))

x_val = np.zeros((60,300,256))
y_val = np.zeros((60,600,2))
for filecount, filename in enumerate(os.listdir(feat_dir)):
    cur_file = os.path.join(feat_dir, filename)
    cur_label = os.path.join(label_dir, filename)
    if filecount < 240:
        x_train[filecount,:,:] = np.genfromtxt(cur_file, delimiter=',')
        y_train[filecount,:,:] = np.genfromtxt(cur_label, delimiter=',')
    else:
        x_val[filecount-240,:,:] = np.genfromtxt(cur_file, delimiter=',')
        y_val[filecount-240,:,:] = np.genfromtxt(cur_label, delimiter=',')

x_train = np.reshape(x_train, (240,300,64,4))
x_val = np.reshape(x_val, (60,300,64,4))

data_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
data_train = data_train.shuffle(buffer_size=1024).batch(40, drop_remainder=True).repeat()

data_val = tf.data.Dataset.from_tensor_slices((x_val,y_val))
data_val = data_val.shuffle(buffer_size=1024).batch(30, drop_remainder=True).repeat()

# nb_batches = tf.data.experimental.cardinality(data_train).numpy()
# for batch in range(nb_batches):
#     model.fit(data_train.take(batch), epochs = epoches, validation_data=data_val,
#               steps_per_epoch=nb_batches)
model.fit(data_train, epochs = epoches, validation_data=data_val, verbose = 2,
          steps_per_epoch=5 ,validation_steps=2)

print("predictions")

sample = os.path.join(feat_dir, "fold1_room1_mix001_ov1.csv")
x_pred = np.genfromtxt(sample, delimiter=',', skip_header=0, dtype=float)
x_pred = np.reshape(x_pred, (1, 300,64,4))
pred = model.predict(x_pred, verbose=2, steps=1)[0]
dump_file = os.path.join(dump_dir, "fold1_room1_mix001_ov1_pred.csv")
np.savetxt(dump_file, pred, delimiter = ",")  