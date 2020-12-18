import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow.lite as tflite
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
import zlib

#from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model name', default = 'mlp')
parser.add_argument('--labels', type=int, required=True, help='model output', default = 2)
parser.add_argument('-outdir', type=str, help='model save output', default = "./hw2ex1/")
args = parser.parse_args()

alpha = 0.05
final_sparsity = 0.20
initial_sparsity = 0.10

output_folder = args.outdir
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
else:
    #os.system(f"mkdir -r {output_folder}")
    raise NameError('output folder already exists, choose another folder')

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

label_width = 6
input_width = 6
LABEL_OPTIONS = args.labels
MODEL_OPTIONS = args.model

class WindowGenerator:
    def __init__(self, input_width, label_width, label_options, mean, std):
        self.input_width = input_width
        self.label_width = label_width
        self.label_options = label_options
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        #input_indeces = np.arange(self.input_width)
        inputs = features[:, :-self.label_width, :]
        #num_labels=2
        if self.label_options < 2:
            labels = features[:, -self.label_width:, self.label_options]
            labels = tf.expand_dims(labels, -1)
            num_labels = 1
        else:
            labels = features[:, -self.label_width:, :]
            num_labels = 2

        inputs.set_shape([None, self.input_width, num_labels])
        labels.set_shape([None, self.label_width, num_labels]) # CHANGED

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=input_width+label_width,
                sequence_stride=1,
                batch_size=32)
        ds = ds.map(self.preprocess)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds
        
if LABEL_OPTIONS == 2:
    num_labels = 2*label_width
else:
    num_labels = 1*label_width

if MODEL_OPTIONS == 'mlp':
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128*alpha, activation='relu'),
        tf.keras.layers.Dense(units=128*alpha, activation='relu'),
        tf.keras.layers.Dense(units=num_labels),
        tf.keras.layers.Reshape((label_width,LABEL_OPTIONS), input_shape=(num_labels,))
    ])
        
elif MODEL_OPTIONS == 'cnn-1d':
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64*alpha, kernel_size=(3,), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64*alpha, activation='relu'),
        tf.keras.layers.Dense(units=num_labels), 
        tf.keras.layers.Reshape((label_width,LABEL_OPTIONS), input_shape=(num_labels,))
    ])
        
elif MODEL_OPTIONS == 'lstm':
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64*alpha),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=num_labels), 
        tf.keras.layers.Reshape((label_width,LABEL_OPTIONS), input_shape=(num_labels,))
    ])

else:
    raise Exception("model not supported")

class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name='mean_absolute_error', **kwargs):
        super().__init__(name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros', shape=[2])
        self.count = self.add_weight('count', initializer='zeros')
       
    
    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))
        return
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)
        #error = tf.reduce_mean(error, axis=1)
        error = tf.reduce_mean(error, axis=[0,1])
        self.total.assign_add(error)
        self.count.assign_add(1)
        return
        
    def result(self):
        result = tf.math.divide_no_nan((self.total[0]+self.total[1])/2, self.count)
        
        return result

generator = WindowGenerator(input_width, label_width, LABEL_OPTIONS, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)

"""cp_callback = tf.keras.callbacks.ModelCheckpoint(
        f'./base/cnn_chkp_best',
        monitor='val_mean_absolute_error',
        verbose=0, 
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )
callbacks = [cp_callback]"""

model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError(), metrics=[MultiOutputMAE()])
history = model.fit(train_ds, epochs=20, validation_data=val_ds)#, callbacks=callbacks)
error = model.evaluate(test_ds) #validation qui

print("\n\n")
print(error)
#model.summary()

run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2],
    tf.float32))
tf.keras.models.save_model(model, output_folder, signatures=concrete_func)

#tf.data.experimental.save(train_ds, './th_train')
#tf.data.experimental.save(val_ds, './th_val')
#tf.data.experimental.save(test_ds, './th_test')

pruning_params = {'pruning_schedule':tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=initial_sparsity, 
                                                                          final_sparsity=final_sparsity,
                                                                          begin_step=len(train_ds)*5,
                                                                          end_step=len(train_ds)*15)}
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude


model = prune_low_magnitude(model, **pruning_params)
p_cp_callback = tf.keras.callbacks.ModelCheckpoint(
        f'./pruned/{args.model}_best',
        monitor='val_mean_absolute_error',
        verbose=0, 
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )
callbacks = [tfmot.sparsity.keras.UpdatePruningStep(), p_cp_callback]
input_shape = [32,32,32]
model.build(input_shape)
model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError(), metrics=[MultiOutputMAE()])
model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks)
strip_model = tfmot.sparsity.keras.strip_pruning(model)
strip_model.save(f'./stripped/dscnn_chkp_best_mfccs')

converter = tf.lite.TFLiteConverter.from_saved_model('./stripped/dscnn_chkp_best_mfccs')
"""WEIGHTS_ONLY QUANTIZATION"""
converter.optimizations= [tf.lite.Optimize.DEFAULT]
quant_model= converter.convert()

with open('my_model.tflite', 'wb') as f:
    f.write(quant_model)
    
with open('Group2_th_a.tflite.zlib', 'wb') as fp:
    tflite_compressed = zlib.compress(quant_model)
    fp.write(tflite_compressed)

tfl_size=os.path.getsize("my_model.tflite")
tflc_size=os.path.getsize("Group2_th_a.tflite.zlib")


interpreter = tf.lite.Interpreter(model_path='my_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

tensor_specs = (tf.TensorSpec([None,6,2], dtype=tf.float32),tf.TensorSpec([None,6,2]))
#test_ds = tf.data.experimental.load('./th_test',tensor_specs)

test_ds = test_ds.unbatch().batch(1)

sum_mae = 0
for data, label in test_ds:
    interpreter.set_tensor(input_details[0]['index'], data)#list(train_ds)[0][0])
    interpreter.invoke()
    my_output = interpreter.get_tensor(output_details[0]['index'])
    mae_vector = np.abs(my_output-label)
    sum_mae += np.average(mae_vector, axis = 1)
    
mae_temp = sum_mae[0][0]/len(list(test_ds))
mae_hum = sum_mae[0][1]/len(list(test_ds))

print(f'size of optimized model: {tfl_size/1024} kB \ncompressed: {tflc_size/1024} kB')
print("mae for temperature = ", mae_temp)
print("mae for humidity = ", mae_hum)
print(f'alpha = {alpha}, initial sparsity: {initial_sparsity}, final sparsity: {final_sparsity}')

history.history 