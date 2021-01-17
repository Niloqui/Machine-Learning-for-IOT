import argparse
import os
import shutil
import time as t

import numpy as np
import pandas as pd
import zlib
from scipy import signal

import tensorflow_model_optimization as tfmot
import tensorflow as tf
tflite = tf.lite
keras = tf.keras


### Reading arguments
parser = argparse.ArgumentParser()
parser.add_argument('--version', default=1, type=int, help='Model version')
parser.add_argument('--seed', default=42, type=int, help='Set initial seed')
args = parser.parse_args()

version = int(args.version)
model_name = str(version)
seed = args.seed

# Setting seed for random number generation
tf.random.set_seed(seed)
np.random.seed(seed)

### Loading the dataset
data_dir = os.path.join('.','data', 'mini_speech_commands')
if not os.path.exists(data_dir):
    zip_path = keras.utils.get_file(
        origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')

labels_file = open("./labels.txt", "r")
LABELS = labels_file.read()
LABELS = np.array(LABELS.split(" "))
labels_file.close()
print(LABELS)

def my_resample(audio, downsample):
    audio = signal.resample_poly(audio, 1, downsample)
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)
    return audio

class SignalGenerator:
    def __init__(self, labels, sampling_rate=16000, resampling_rate=None, frame_length=1920 , frame_step=960,
                num_mel_bins=40, lower_freq=20, upper_freq=48000, num_coefficients=10, mfccs=False):
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.mel_inputs = [num_mel_bins, None, sampling_rate, lower_freq, upper_freq]
        self.mfccs_coeff = num_coefficients
        self.labels = labels
        self.sampling_rate = sampling_rate
        
        if resampling_rate is None:
            self.resampling_rate = sample_rate
        else:
            self.resampling_rate = resampling_rate
        
        self.downsampling = int(self.sampling_rate / self.resampling_rate)
    
        if mfccs:
            num_spectrogram_bins = (frame_length) // 2 + 1
            self.l2mel_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.resampling_rate,
                    lower_freq, upper_freq)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft
    
    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        
        if self.resampling_rate != self.sampling_rate:
            audio = tf.numpy_function(my_resample, [audio, self.downsampling], tf.float32)
        
        audio = tf.squeeze(audio, axis=1)
        return audio, label_id
    
    def pad(self, audio):
        zero_padding = tf.zeros([self.resampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio,zero_padding],0)
        audio.set_shape([self.resampling_rate])
        return audio
    
    def get_spectrogram(self, audio):
        tfstft = tf.signal.stft(audio, frame_length=self.frame_length, frame_step=self.frame_step,fft_length=self.frame_length)
        spectrogram = tf.abs(tfstft)
        
        return spectrogram
    
    def get_mfcc(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram, self.l2mel_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        mfccs_ = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :self.mfccs_coeff]
        return mfccs_
    
    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32,32])
        return spectrogram, label
    
    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs_ = self.get_mfcc(spectrogram)
        mfccs_ = tf.expand_dims(mfccs_, -1)
        return mfccs_, label
    
    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()
        
        if train:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)
        
        return ds


OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfccs': True,
        'lower_freq': 20, 'upper_freq': 4000, 'num_mel_bins': 80, 'num_coefficients': 20}
stride = [2, 1]


#sample_rate = 8000
sample_rate = 16000

train_files = tf.strings.split(tf.io.read_file('./kws_train_split.txt'),sep='\n')[:-1]
val_files = tf.strings.split(tf.io.read_file('./kws_val_split.txt'),sep='\n')[:-1]
test_files = tf.strings.split(tf.io.read_file('./kws_test_split.txt'),sep='\n')[:-1]

generator = SignalGenerator(LABELS, 16000, sample_rate, **OPTIONS)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)


### Model definition
# A modified version of the DSCNN
def input_layer(filters):
    return keras.Sequential([keras.layers.Conv2D(filters=filters, kernel_size=[3, 3], strides=stride, use_bias=False),
                            keras.layers.BatchNormalization(momentum=0.1),
                            keras.layers.Dropout(0.1),
                            keras.layers.ReLU()])

def expansion_block(filters,loop=1):
    return keras.Sequential([keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
                            keras.layers.Conv2D(filters=filters, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                            keras.layers.BatchNormalization(momentum=0.1),
                            keras.layers.Dropout(0.1),
                            keras.layers.ReLU()])

def in_block(x,filters):
    s = 1
    f1 = filters
    # first block
    x = keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False)(x)
    x = keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.ReLU()(x)
    return x  

def exp_block(x, filters):
    s = 1
    f1 = filters
    x = keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False)(x)
    x = keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.ReLU()(x)
    
    return x

def res_block(input, filters):
    s = 1
    f1 = filters
    
    # first block
    x_init = keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False)(input)
    x = keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='same', use_bias=False)(x_init)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.1)(x)
    
    #### TO SUM THE TENSOR THEY MUST HAVE THE SAME SHAPE (PADDING OR SAME DEPTHWISE)
    #diff = tf.subtract(tf.shape(input) ,tf.shape(x)) 
    #print((diff[0],diff[1],diff[2],diff[3]))
    #x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
    input = keras.layers.BatchNormalization()(x_init)
    
    # add 
    x = keras.layers.Add()([input, x])
    x = keras.layers.ReLU()(x)
    
    return x

def parallel_block(input, filters):
    s = 1
    f1 = filters
    
    x_init = keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False)(input)
    
    # first block
    x = keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='same', use_bias=False)(x_init)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.1)(x)
    # second block
    x2 = keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(s, s), padding='same', use_bias=False)(x_init)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.Dropout(0.1)(x2)
    
    # add 
    x = keras.layers.Add()([x2, x])
    x = keras.layers.ReLU()(x)
    
    return x

def inception_block(input, filters):
    s = 1
    f1 = filters
    
    
    x_init = keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False)(input)
    
    # first block
    conv1x1 = keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='same', use_bias=False)(x_init)
    
    # second block
    conv3x3 = keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(s, s), padding='same', use_bias=False)(x_init)
    
    x = keras.layers.concatenate([x_init, conv3x3, conv1x1], axis=3)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.ReLU()(x)
    
    return x

def dscnn_res():
    input = keras.Input(shape=(int(sample_rate/OPTIONS['frame_step']) - 1, OPTIONS['num_coefficients'], 1))
    x = in_block(input,32)
    x = exp_block(x,64)
    x = exp_block(x,128)
    x = exp_block(x,256)
    x = exp_block(x,512)
    x = res_block(x,512) #single res block accuracy on test 0.89 on training 0.98
    x = res_block(x,512)
    x = exp_block(x,512)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(len(LABELS), kernel_initializer='he_normal')(x) 
    return keras.Model(inputs=input, outputs=x, name='DSCNN_res')

def dscnn_sub():
    input = keras.Input(shape=(int(sample_rate/OPTIONS['frame_step']) - 1, OPTIONS['num_coefficients'], 1))
    x = in_block(input,32)
    x = exp_block(x,64)
    x = exp_block(x,128)
    x = exp_block(x,256)
    x = exp_block(x,512)
    x = parallel_block(x,512) #single res block accuracy on test 0.45 on training 0.993
    x = parallel_block(x,512)
    x = exp_block(x,512)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(len(LABELS), kernel_initializer='he_normal')(x) 
    return keras.Model(inputs=input, outputs=x, name='DSCNN_res')

def dscnn_sub_redux():
    input = keras.Input(shape=(int(sample_rate/OPTIONS['frame_step']) - 1, OPTIONS['num_coefficients'], 1))
    x = in_block(input,32)
    x = exp_block(x,64)
    x = parallel_block(x,64) #single res block accuracy on test 0.89 on training 0.98
    x = exp_block(x,128)
    x = parallel_block(x,128)
    x = exp_block(x,128)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(len(LABELS), kernel_initializer='he_normal')(x) 
    return keras.Model(inputs=input, outputs=x, name='DSCNN_res')

def dscnn_sub_super_redux():
    input = keras.Input(shape=(int(sample_rate/OPTIONS['frame_step']) - 1, OPTIONS['num_coefficients'], 1))
    x = in_block(input,32)
    x = exp_block(x,64)
    x = parallel_block(x,64) 
    x = exp_block(x,128)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(len(LABELS), kernel_initializer='he_normal')(x) 
    return keras.Model(inputs=input, outputs=x, name='DSCNN_res')

def dscnn_inc():
    input = keras.Input(shape=(int(sample_rate/OPTIONS['frame_step']) - 1, OPTIONS['num_coefficients'], 1))
    x = in_block(input,32)
    x = exp_block(x,64)
    x = inception_block(x,64)
    x = exp_block(x,64) 
    x = exp_block(x,128)
    x = inception_block(x,128) 
    x = exp_block(x,128)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(len(LABELS), kernel_initializer='he_normal')(x) 
    return keras.Model(inputs=input, outputs=x, name='DSCNN_inc')


if version == 1: #LITTLE DSCNN
    model = keras.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=stride, use_bias=False),
            keras.layers.BatchNormalization(momentum=0.1),
            keras.layers.Dropout(0.1),
            keras.layers.ReLU(),
            keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
            keras.layers.Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
            keras.layers.BatchNormalization(momentum=0.1),
            keras.layers.Dropout(0.1),
            keras.layers.ReLU(),
            keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
            keras.layers.Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
            keras.layers.BatchNormalization(momentum=0.1),
            keras.layers.Dropout(0.1),
            keras.layers.ReLU(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=len(LABELS))
    ])
elif version == 2: #BIG DSCNN
    model = keras.Sequential([
        keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=stride, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.Dropout(0.1),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.Dropout(0.1),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.Dropout(0.1),
        keras.layers.ReLU(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=len(LABELS))
    ])
elif version == 3: #SIMPLE SEQUENTIAL (MLP)
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(units=256,activation=keras.activations.relu),
        keras.layers.Dense(units=256,activation=keras.activations.relu),
        keras.layers.Dense(units=256,activation=keras.activations.relu),
        keras.layers.Dense(units=len(LABELS))
    ])
elif version == 4: #SIMPLE CNN (CNN)
    model = keras.Sequential([
        keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides = stride, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides = stride, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides = stride, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=len(LABELS))
    ])
elif version == 5: # MASKING
    model = keras.Sequential([
        keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=stride, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.Masking(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.Masking(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.Masking(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=len(LABELS))
    ])
elif version == 6: #schifo 0.75
    model = keras.applications.MobileNet(
            #input_shape=(49, 49, 1),
            input_shape=(int(sample_rate/OPTIONS['frame_step']) - 1, OPTIONS['num_coefficients'], 1),
            alpha=1.0,
            depth_multiplier=1,
            dropout=0.5,
            include_top=True,
            classes=len(LABELS),
            weights=None,
            pooling="max"
        )
elif version == 7: #schifo 0.75
    model = keras.applications.MobileNetV2(
            #input_shape=(49, 49, 1),
            input_shape=(int(sample_rate/OPTIONS['frame_step']) - 1, OPTIONS['num_coefficients'], 1),
            alpha=1.0,
            include_top=True,
            classes=len(LABELS),
            weights=None,
            pooling="max"
        )
elif version == 8: #migliore probabilmente
    model = keras.Sequential([
        input_layer(32),
        expansion_block(64),
        expansion_block(128),
        expansion_block(256),
        expansion_block(512),
        expansion_block(512),
        expansion_block(512),
        expansion_block(512),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=len(LABELS))
    ])
elif version == 9: #da rivedere
    model = dscnn_res()
elif version == 10: #da rivedere
    model = dscnn_sub()
elif version == 11:
    model = dscnn_sub_redux()
elif version == 12:
    model = dscnn_sub_super_redux()
elif version == 13:
    model = dscnn_inc()
else:
    print("model not found")
    exit(-1)


### Training for the first time
def training_model(model, model_name, train_ds, val_ds):
    if not os.path.exists('./callback_train_chkp'):
        os.mkdir('./callback_train_chkp')
    
    callback_folder_name = f'./callback_train_chkp/{model_name}_chkp_best'
    if not os.path.exists(callback_folder_name):
        os.mkdir(callback_folder_name)
    
    model.compile(
        optimizer = 'Adam',
        loss = keras.losses.SparseCategoricalCrossentropy(True),
        metrics = [keras.metrics.SparseCategoricalAccuracy()]
    )
    
    cp_callback = keras.callbacks.ModelCheckpoint(
        callback_folder_name,
        monitor = 'val_sparse_categorical_accuracy',
        verbose = 0, 
        save_best_only = True,
        save_weights_only = False,
        mode = 'auto',
        save_freq = 'epoch'
    )
    
    model.fit(train_ds, batch_size=32, epochs=20, validation_data=val_ds, callbacks=[cp_callback], verbose=2)
    model.summary()
    
    _, test_acc = model.evaluate(test_ds, verbose=2)
    
    model_path = f'{callback_folder_name}/saved_model.pb'
    msize = os.path.getsize(model_path)
    print(f'\nacc: {test_acc}, size: {msize/1024}kB\n')
        
    return callback_folder_name


### Generating tflite models
def training_and_pruning_model(model, model_name, train_ds, val_ds, num_coeff):
    if not os.path.exists('./callback_train_chkp'):
        os.mkdir('./callback_train_chkp')
    
    callback_folder_name = f'./callback_train_chkp/{model_name}_chkp_best'
    if not os.path.exists(callback_folder_name):
        os.mkdir(callback_folder_name)
    
    input_shape=[32, 49, num_coeff]
    model.build(input_shape)
    model.compile(
        optimizer = 'Adam',
        loss = keras.losses.SparseCategoricalCrossentropy(True),
        metrics = [keras.metrics.SparseCategoricalAccuracy()]
    )
    
    cp_callback = keras.callbacks.ModelCheckpoint(
        callback_folder_name,
        monitor = 'val_sparse_categorical_accuracy',
        verbose = 0, 
        save_best_only = True,
        save_weights_only = False,
        mode = 'auto',
        save_freq = 'epoch'
    )
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep(), cp_callback]
    
    model.fit(train_ds, batch_size=32, epochs=20, validation_data=val_ds, callbacks=callbacks, verbose=2)
    
    stripped_model_folder = f'./stripped/{model_name}_chkp_best'
    strip_model = tfmot.sparsity.keras.strip_pruning(model)
    strip_model.save(stripped_model_folder)
    strip_model.summary()
    
    return stripped_model_folder, callback_folder_name

def generate_tflite(model_folder, output_name, test_ds):
    test_ds = test_ds.unbatch().batch(1)
    
    tflite_dirs = 'tflite_models'
    if not os.path.exists(tflite_dirs): 
        os.mkdir(tflite_dirs)
    tflite_dirs = f'tflite_models/{output_name}'
    
    basic_file = f'{tflite_dirs}_basic.tflite'
    optimized_file = f'{tflite_dirs}_optimized.tflite'
    compressed_file = f'{tflite_dirs}_compressed.tflite.zlib'
    
    
    # Basic file
    converter = tflite.TFLiteConverter.from_saved_model(model_folder)
    tflite_model = converter.convert()
    
    with open(basic_file, 'wb') as f:
        f.write(tflite_model)
    tflb_size = os.path.getsize(basic_file)
    
    
    # Optimized file
    converter.optimizations = [tflite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    
    with open(optimized_file, 'wb') as f:
        f.write(tflite_quant_model)
    
    tflo_size = os.path.getsize(optimized_file)
    
    
    # Compressed file
    tflite_compressed = zlib.compress(tflite_quant_model)
    
    with open(compressed_file, 'wb') as fp:
        fp.write(tflite_compressed)
            
    tflc_size=os.path.getsize(compressed_file)
    
    
    # Evaluating the model
    interpreter = tflite.Interpreter(model_path = optimized_file)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    num_corr = 0
    num = 0
    start = t.time()
    for input_data, label in test_ds:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = np.argmax(interpreter.get_tensor(output_details[0]['index']))
        
        if label.numpy()[0] == output_data:
            num_corr += 1
        num += 1
    
    end = t.time() - start
    
    # Final outputs
    print(f'Size of basic model: {tflb_size/1024} kB')
    print(f'Size of optimized model: {tflo_size/1024} kB')
    print(f'Compressed: {tflc_size/1024} kB')
    print(f'Accuracy: {num_corr/num}\nTime: {end} ms\n\n')
    
    return basic_file, optimized_file, compressed_file, num_corr/num


trained_model_path = training_model(model, model_name, train_ds, val_ds)

_, model_path, compressed_file, accuracy_not_pruned = generate_tflite(trained_model_path, model_name+"not_pruned", test_ds)
shutil.copyfile(model_path, f"./{model_name}_not_pruned.tflite")

model = keras.models.load_model(trained_model_path)
pruning_params = {
        'pruning_schedule' : tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity = 0.3, 
            final_sparsity = 0.8,
            begin_step = len(train_ds) * 5,
            end_step = len(train_ds) * 15
        )
    }
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

#model_path = trained_model_path
'''
    if version == 1:   #little
        #if pruning
        model = prune_low_magnitude(model, **pruning_params)
        model_path, _ = training_and_pruning_model(model, str(version)+"_pruned", train_ds, val_ds, OPTIONS['num_coefficients'])
    elif version == 2: #BIG DSCNN
        print()
    elif version == 3: #dense
        model = prune_low_magnitude(model, **pruning_params)
        model_path, _ = training_and_pruning_model(model, str(version)+"_pruned", train_ds, val_ds, OPTIONS['num_coefficients'])
    elif version == 4: #cnn
        model = prune_low_magnitude(model, **pruning_params)
        model_path, _ = training_and_pruning_model(model, str(version)+"_pruned", train_ds, val_ds, OPTIONS['num_coefficients'])
    elif version == 5: #masking
        print()
    elif version == 6: #mobilenet
        print()
    elif version == 7: #mobilenetV2
        print()
    elif version == 8: #my expansion DSCNN
        print()
    elif version == 9: #my expansion + residual DSCNN
        print()
'''
model = prune_low_magnitude(model, **pruning_params)
model_path, _ = training_and_pruning_model(model, model_name+"_pruned", train_ds, val_ds, OPTIONS['num_coefficients'])

_, model_path, compressed_file, accuracy_pruned = generate_tflite(model_path, model_name+"_pruned", test_ds)
shutil.copyfile(model_path, f"./{model_name}_pruned.tflite")

print(LABELS)
print(f"Final recap:\nModel: {model_name}\nAccuracy not pruned: {accuracy_not_pruned}\nAccuracy pruned: {accuracy_pruned}")



print("End.")
