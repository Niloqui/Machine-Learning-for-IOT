import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model name', default = 'mlp')
parser.add_argument('--labels', type=int, required=True, help='model output', default = 2)
parser.add_argument('-outdir', type=str, help='model save output', default = "./hw2ex1/")
args = parser.parse_args()

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
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=num_labels),
        tf.keras.layers.Reshape((label_width,LABEL_OPTIONS), input_shape=(num_labels,))
    ])
        
elif MODEL_OPTIONS == 'cnn-1d':
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=(3,), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=num_labels), 
        tf.keras.layers.Reshape((label_width,LABEL_OPTIONS), input_shape=(num_labels,))
    ])
        
elif MODEL_OPTIONS == 'lstm':
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64),
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
        result = tf.math.divide_no_nan(self.total, self.count)
        
        return result

if LABEL_OPTIONS == 2:
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[MultiOutputMAE()])#['mean_absolute_error'])
    
else:
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mean_absolute_error'])

generator = WindowGenerator(input_width, label_width, LABEL_OPTIONS, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)

history = model.fit(train_ds, epochs=20, validation_data=val_ds)
error = model.evaluate(test_ds)

print("\n\n")
print(error)
#model.summary()

run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2],
    tf.float32))
model.save("saved_models", signatures=concrete_func)

tf.data.experimental.save(train_ds, './th_train')
tf.data.experimental.save(val_ds, './th_val')
tf.data.experimental.save(test_ds, './th_test')


tensor_specs = (tf.TensorSpec([None,6,2], dtype=tf.float32),tf.TensorSpec([None,2]))
train_ds = tf.data.experimental.load('./th_train',tensor_specs)
val_ds = tf.data.experimental.load('./th_val', tensor_specs)
test_ds = tf.data.experimental.load('./th_test',tensor_specs)


converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('my_model.tflite', 'wb') as f:
    f.write(tflite_model)
    
interpreter = tflite.Interpreter(model_path="my_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Number of inputs:", len(input_details))
print("Number of outputs:", len(output_details))
print("Input name:", input_details[0]['name'])
print("Input shape:", input_details[0]['shape'])

inputs = []
outputs = []

#range could be inf it is needed to feed the model with new datas
for i in range(10):

    my_input = np.array(np.random.uniform(-1, 1, input_details[0]['shape']), dtype=np.float32)
    print("Input:", my_input)
    inputs.append(my_input[0, 0])
    
    interpreter.set_tensor(input_details[0]['index'], my_input)
    
    interpreter.invoke()
    
    my_output = interpreter.get_tensor(output_details[0]['index'])
    print("Output:", my_output)  
    outputs.append(my_output[0, 0])

plt.plot(x_train, y_train, 'r-')
plt.plot(inputs, outputs, 'b*')
