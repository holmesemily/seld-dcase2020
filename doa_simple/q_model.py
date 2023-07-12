import numpy as np
import tensorflow as tf
import os

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model/ptq/test_a.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
print("input:", input_shape)

feat_dir = '../dataset/myfeatures/foa_dev'
sample = os.path.join(feat_dir, "fold1_room1_mix001_ov1.csv")
x_pred = np.genfromtxt(sample, delimiter=',', skip_header=0, dtype=np.int8)
x_pred = np.reshape(x_pred, (10,300,64,4))
# x_pred = x_pred[:1,:,:,:]

final_output = np.zeros((10, 60, 2))

for index in range(10):
    cur_x_pred = x_pred[index:index+1,:,:,:]
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.int8)
    interpreter.set_tensor(input_details[0]['index'], cur_x_pred)
    interpreter.allocate_tensors()
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    final_output[index, :,:] = interpreter.get_tensor(output_details[0]['index'])

# print(output_data)

final_output = np.reshape(final_output, (600,2))
dump_dir = '../dataset/myfeatures/dump/'
dump_file = dump_dir + 'fold1_room1_mix001_ov1_pred_12_qmodelonly.csv'
np.savetxt(dump_file, final_output, delimiter = ",")  


# tensor_details = interpreter.get_tensor_details()

# for dict in tensor_details:
#     i = dict['index']
#     tensor_name = dict['name']
#     scales = dict['quantization_parameters']['scales']
#     zero_points = dict['quantization_parameters']['zero_points']
#     tensor = interpreter.tensor(i)()

#     print(i, type, tensor_name, scales.shape, zero_points.shape, tensor.shape)
