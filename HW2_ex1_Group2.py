import argparse
import numpy as np
from subprocess import call
import tensorflow as tf
import time
from scipy import signal
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=False,
        help='model full path')
args = parser.parse_args()

#call('sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
#            shell=True)

interpreter = tf.lite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

tensor_specs = (tf.TensorSpec([None,6,2], dtype=tf.float32),tf.TensorSpec([None,6,2]))
test_ds = tf.data.experimental.load('./th_test',tensor_specs)

test_ds = test_ds.unbatch().batch(1)

sum_mae = [0, 0]
for data, label in test_ds:
    interpreter.set_tensor(input_details[0]['index'], data)#list(train_ds)[0][0])
    interpreter.invoke()
    my_output = interpreter.get_tensor(output_details[0]['index'])
    mae_vector = np.abs(my_output-label)
    sum_mae += np.average(mae_vector, axis = 1)
    
mae_temp = sum_mae[0][0]/len(list(test_ds))
mae_hum = sum_mae[0][1]/len(list(test_ds))

tflsize = os.path.getsize(args.model)
print('size of TFlite model = ', tflsize)

print("mae for temperature = ", mae_temp)
print("mae for humidity = ", mae_hum)

