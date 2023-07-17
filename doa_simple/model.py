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
import time

# constants
GENERATE_DATASET = 1
GENERATE_MODEL = 1
QUANTIZE_IO = 0
MODEL_FIT = 1
PREDICT = 1
CONVERT_LITE = 0

NAME = "model_cnn_nq_norm_td"
MODEL_FOLDER = 'model/1707/' + NAME
PREDICT_FOLDER = NAME + "_predict_1_b256_tanh_nobm.csv"
TFLITE_FILE_PATH = MODEL_FOLDER + '/ptq/' + 'lite.tflite'

# compute if prediction is within 20deg of grond truth for all batch
# output average (1 = 100% of predictions were within 20deg)
# @tf.keras.saving.register_keras_serializable()
def is_within_20deg(y_true, y_pred):
    delta_theta = tf.abs(tf.subtract(y_true[1], y_pred[1]))
    return tf.reduce_mean(tf.cast(tf.less_equal(delta_theta, 20.0), tf.float32))
nb_cnn2d_filt = 64

model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(300,64,4)),
        tf.keras.layers.Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding="same"),                   
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('tanh'),
        tf.keras.layers.MaxPooling2D(pool_size=(5, 4)),
        tf.keras.layers.Dropout(0.1),      
        # bloc
        tf.keras.layers.Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding="same"),                   
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('tanh'),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 4)),
        tf.keras.layers.Dropout(0.1),   
        # bloc 
        tf.keras.layers.Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding="same"),                   
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('tanh'),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),
        tf.keras.layers.Dropout(0.1),   
        
        #rnn
        tf.keras.layers.Reshape((60, 128)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, activation='tanh', dropout=0.1, recurrent_dropout=0.1,
                return_sequences=True),merge_mode='mul'),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, activation='tanh', dropout=0.1, recurrent_dropout=0.1,
                return_sequences=True),merge_mode='mul'),
        
        # fc
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128)),
        tf.keras.layers.Dropout(0.1),   
        tf.keras.layers.Dense(2),
        tf.keras.layers.Activation('tanh')
        ])

model2 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(300,64,4)),
    tf.keras.layers.Reshape((60, 1280)),
    tf.keras.layers.Dense(2),  
    ])

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=['msle'], metrics=['accuracy'])

# train and test datasets
feat_dir = '../dataset/myfeatures/foa_dev'
feat_dir_norm = '../dataset/myfeatures/foa_dev_norm2'
label_dir = '../dataset/myfeatures/foa_dev_label_norm2'
data_dir = '../dataset/myfeatures/sets'
dump_dir = '../dataset/myfeatures/dump'
model_dir = 'model'
filenames_list = os.listdir(feat_dir)
epoches = 25

if GENERATE_DATASET:
    print("Generate dataset...")
    start = time.time()
    x_train = np.zeros((240,3000,256))
    y_train = np.zeros((240,600,2))

    x_val = np.zeros((60,3000,256))
    y_val = np.zeros((60,600,2))
    for filecount, filename in enumerate(os.listdir(feat_dir_norm)):
        cur_file = os.path.join(feat_dir_norm, filename)
        cur_label = os.path.join(label_dir, filename)
        if filecount < 240:
            x_train[filecount,:,:] = np.genfromtxt(cur_file, delimiter=',')
            y_train[filecount,:,:] = np.genfromtxt(cur_label, delimiter=',')
        else:
            x_val[filecount-240,:,:] = np.genfromtxt(cur_file, delimiter=',')
            y_val[filecount-240,:,:] = np.genfromtxt(cur_label, delimiter=',')

    x_train = np.reshape(x_train, (240,3000,64,4))
    x_val = np.reshape(x_val, (60,3000,64,4))
    # fix in 2 chunks
    x_train = np.reshape(x_train, (2400,300,64,4))
    x_val = np.reshape(x_val, (600,300,64,4))

    y_train = np.reshape(y_train, (2400,60,2))
    y_val = np.reshape(y_val, (600,60,2))

    data_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    data_train = data_train.shuffle(buffer_size=256).batch(256, drop_remainder=True).repeat()

    data_val = tf.data.Dataset.from_tensor_slices((x_val,y_val))
    data_val = data_val.shuffle(buffer_size=256).batch(256, drop_remainder=True).repeat()

    print("Time to generate dataset:", time.time() - start, 's')
    
if QUANTIZE_IO:    
    # quantize input/output
    print("Quantize...")
    min_i = -73.14660305467869
    max_i = 28.056959861353036
    x_train, dump, dump = tf.quantization.quantize(x_train, min_i, max_i, 'qint8', "SCALED")
    x_val, dump, dump = tf.quantization.quantize(x_val, min_i, max_i, 'qint8', "SCALED")

    # print(x_train)
    # print(x_val)

    min_o_1 = -180.0
    max_o_1 = 180.0
    min_o_2 = -40.0
    max_o_2 = 40.0

    y_train, dump, dump = tf.quantization.quantize(y_train, min_o_1, max_o_1, 'qint8', "SCALED")
    y_val, dump, dump = tf.quantization.quantize(y_val, min_o_1, max_o_1, 'qint8', "SCALED")

    print(y_train)
    print(y_val)

if MODEL_FIT:
    model.fit(data_train, epochs = epoches, validation_data=data_val, verbose = 2,
            steps_per_epoch=2 ,validation_steps=2)
    model.save(MODEL_FOLDER +'.h5')

if PREDICT:
    print("Predicting...")
    # model = tf.keras.models.load_model(MODEL_FOLDER + '.h5')
    sample = os.path.join(feat_dir_norm, "fold1_room1_mix001_ov1.csv")
    x_pred = np.genfromtxt(sample, delimiter=',', skip_header=0, dtype=float)
    x_pred = np.reshape(x_pred, (1, 3000,64,4))
    x_pred = np.reshape(x_pred, (10,300,64,4))
    pred = model.predict(x_pred, verbose=2, steps=1)
    pred = np.reshape(pred, (600,2))
    dump_file = os.path.join(dump_dir, PREDICT_FOLDER)
    np.savetxt(dump_file, pred, delimiter = ",")  

if CONVERT_LITE:
    def representative_dataset():
        for data in tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1).take(100):
            yield [np.array(data[0], dtype=np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_FOLDER + '.keras')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_quant_model = converter.convert()

    with open(TFLITE_FILE_PATH, 'wb') as f:
      f.write(tflite_quant_model)

