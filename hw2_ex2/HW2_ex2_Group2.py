import argparse
import os
import time as t

import numpy as np
import pandas as pd
import zlib

import tensorflow_model_optimization as tfmot
import tensorflow as tf
import tensorflow.lite as tflite
#from tensorflow import keras
keras = tf.keras


# Reading arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mfccs', action='store_true', help='Use MFCCs')
parser.add_argument('--force-first-train', action='store_true', help='Overwrite original models')
args = parser.parse_args()

mfccs = args.mfccs
force_first_train = args.force_first_train


# Setting seed for random number generation
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


# Loading the dataset
data_dir = os.path.join('.','data', 'mini_speech_commands')
if not os.path.exists(data_dir):
    zip_path = keras.utils.get_file(
        origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')

LABELS = np.array(tf.io.gfile.listdir(str(data_dir))) 
LABELS = LABELS[LABELS != 'README.md']

class SignalGenerator:
    def __init__(self, labels, sampling_rate=16000, frame_length=1920 , frame_step=960, num_mel_bins=40,
                 lower_freq=20, upper_freq=48000, num_coefficients=10, mfccs=False):
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.mel_inputs = [num_mel_bins, None, sampling_rate, lower_freq, upper_freq]
        self.mfccs_coeff = num_coefficients
        self.labels=labels
        self.sampling_rate=sampling_rate
        num_spectrogram_bins = (frame_length) // 2 + 1

        if mfccs:
            self.l2mel_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
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
        audio = tf.squeeze(audio, axis=1)
        return audio, label_id

    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio,zero_padding],0)
        audio.set_shape([self.sampling_rate])
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
        #spectrogram = tf.expand_dims(spectrogram, -1)
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

STFT_OPTIONS = {'frame_length': 256, 'frame_step': 128, 'mfccs': False}
MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfccs': True,
        'lower_freq': 20, 'upper_freq': 4000, 'num_mel_bins': 40, 'num_coefficients': 10}

if mfccs:
    options = MFCC_OPTIONS
    stride = [2, 1]
else:
    options = STFT_OPTIONS
    stride = [2, 2]

dataset_dir = 'data/mini_speech_commands' + ("_mfccs" if mfccs else "_stft")
if os.path.exists(dataset_dir):
    if mfccs:
        tensor_specs = (
            tf.TensorSpec([None,49,10,1],dtype=tf.float32),
            tf.TensorSpec([None,],dtype=tf.int64)
        )
    else: # stft
        tensor_specs = (
            tf.TensorSpec([None,32,32,1],dtype=tf.float32),
            tf.TensorSpec([None,],dtype=tf.int64)
        )
    
    train_ds = tf.data.experimental.load(f'{dataset_dir}/th_train', tensor_specs)
    val_ds = tf.data.experimental.load(f'{dataset_dir}/th_val', tensor_specs)
    test_ds = tf.data.experimental.load(f'{dataset_dir}/th_test', tensor_specs)
else:
    os.mkdir(dataset_dir)
    train_files = tf.strings.split(tf.io.read_file('./kws_train_split.txt'),sep='\n')[:-1]
    val_files = tf.strings.split(tf.io.read_file('./kws_val_split.txt'),sep='\n')[:-1]
    test_files = tf.strings.split(tf.io.read_file('./kws_test_split.txt'),sep='\n')[:-1]
    
    generator = SignalGenerator(LABELS, 16000, **options)
    train_ds = generator.make_dataset(train_files, True)
    val_ds = generator.make_dataset(val_files, False)
    test_ds = generator.make_dataset(test_files, False)
    
    tf.data.experimental.save(train_ds, f'{dataset_dir}/th_train')
    tf.data.experimental.save(val_ds, f'{dataset_dir}/th_val')
    tf.data.experimental.save(test_ds, f'{dataset_dir}/th_test')


# Model definition
models = {
    'MLP' : keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(units=256,activation=keras.activations.relu),
        keras.layers.Dense(units=256,activation=keras.activations.relu),
        keras.layers.Dense(units=256,activation=keras.activations.relu),
        keras.layers.Dense(units=len(LABELS))
    ]),
    'CNN' : keras.Sequential([
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
    ]),
    'DSCNN' : keras.Sequential([
        keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=stride, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=len(LABELS))
    ])
}

# Training for the first time
for i, model_name in enumerate(models):
    callback_folder_name = f'./callback_test_chkp/{model_name}{"_mfccs" if mfccs else "_stft"}_chkp_best'
    
    if force_first_train or not os.path.exists(callback_folder_name):
        model = models[model_name]
        model.compile(
            optimizer = 'Adam',
            loss = keras.losses.SparseCategoricalCrossentropy(True),
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
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
        
        model.fit(train_ds, batch_size=32, epochs=20, validation_data=val_ds,
                callbacks=[cp_callback])
        model.summary()
        
        start = t.time()
        test_loss, test_acc = model.evaluate(test_ds, verbose=2)
        end = t.time() - start
        
        msize = os.path.getsize(f'{callback_folder_name}/saved_model.pb')
        print(f'\nacc: {test_acc}, size: {msize} Inference Latency {end}ms\n')
    else:
        print(f'{callback_folder_name} already exists.')


# Saving basic tflite model
test_ds = test_ds.unbatch().batch(1)
tflite_dirs = './tflite_models'
if not os.path.exists(tflite_dirs): 
    os.mkdir(tflite_dirs)

for model_name in models:
    mod = models[model_name]
    model_folder = f'./callback_test_chkp/{model_name}{"_mfccs" if mfccs else "_stft"}_chkp_best'
    basic_file = f'{tflite_dirs}/{model_name}{"_mfccs" if mfccs else "_stft"}_basic.tflite'
    optimized_file = f'{tflite_dirs}/{model_name}{"_mfccs" if mfccs else "_stft"}_optimized.tflite'
    compressed_file = f'{tflite_dirs}/{model_name}{"_mfccs" if mfccs else "_stft"}_compressed.tflite.zlib'
    
    # Basic file
    if force_first_train or not os.path.exists(basic_file):
        with open(basic_file, 'wb') as f:
            converter = tf.lite.TFLiteConverter.from_saved_model(model_folder)
            tflite_model = converter.convert()
            
            f.write(tflite_model)
    
    tflb_size = os.path.getsize(basic_file)
    
    # Optimized file
    if force_first_train or not os.path.exists(optimized_file):
        with open(optimized_file, 'wb') as f:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_quant_model = converter.convert()
            
            f.write(tflite_quant_model)
    
    tflo_size = os.path.getsize(optimized_file)
    
    # Compressed file
    if force_first_train or not os.path.exists(compressed_file):
        with open(compressed_file, 'wb') as fp:
            tflite_compressed = zlib.compress(tflite_quant_model)
            fp.write(tflite_compressed)
    
    tflc_size=os.path.getsize(compressed_file)
    
    
    # Evaluating the model
    interpreter = tflite.Interpreter(model_path = optimized_file)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    output_details = interpreter.get_output_details()
    
    num_corr = 0
    num = 0
    start = t.time()
    for input_data, label in test_ds:
        #input_data = convert(input_data, 0, 255, np.uint8)
        #label = convert(label, 0, 255, np.uint8)
        #input_data = tf.quantization.quantize(input_data,min(input_data),max(input_data),tf.quint8)
        #label = tf.quantization.quantize(label,min(LABELS),max(LABELS),tf.quint8)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = np.argmax(interpreter.get_tensor(output_details[0]['index']))
        #y_pred = tf.constant([output_data],dtype=tf.int64)
        
        if label.numpy()[0] == output_data:
            num_corr += 1
        num += 1
    
    end = t.time() - start
    
    # Final outputs
    print(f'Size of basic model: {tflb_size/1024} kB')
    print(f'Size of optimized model: {tflo_size/1024} kB')
    print(f'Compressed: {tflc_size/1024} kB')
    print(f'Accuracy: {num_corr/num}\nTime: {end} ms\n\n')


# Stripping the model
pruning_params = {
    'pruning_schedule' : tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.30, 
        final_sparsity=0.8,
        begin_step=len(train_ds)*5,
        end_step=len(train_ds)*15
    )
}
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

for model_name in models:
    model_folder = f'./callback_test_chkp/{model_name}{"_mfccs" if mfccs else "_stft"}_chkp_best'
    stripped_model_folder = f'./stripped/{model_name}{"_mfccs" if mfccs else "_stft"}_chkp_best'
    
    if force_first_train or not os.path.exists(stripped_model_folder) or True:
        model = tf.keras.models.load_model(model_folder)
        model = prune_low_magnitude(model, **pruning_params)
        
        cp_callback = keras.callbacks.ModelCheckpoint(
            f'./pruned/dscnn_chkp_best_mfccs',
            monitor='val_loss',
            verbose=0, 
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            save_freq='epoch'
        )
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep(), cp_callback]
        
        input_shape = [32, 32, 32]
        model.build(input_shape)
        model.compile(
            optimizer = 'Adam',
            loss = keras.losses.SparseCategoricalCrossentropy(True),
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        
        model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks)
        strip_model = tfmot.sparsity.keras.strip_pruning(model)
        strip_model.save(stripped_model_folder)
        strip_model.summary()
    
    """
    # Evaluating the model
    interpreter = tflite.Interpreter(model_path = stripped_model_folder)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
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
    
    msize = os.path.getsize(f'{stripped_model_folder}/saved_model.pb')
    print(f'\nacc: {test_acc}, size: {msize} Inference Latency {end}ms\n')
    
    """
    '''
    start = t.time()
    test_loss, test_acc2 = model.evaluate(test_ds, verbose=2)
    end = t.time() - start
    msize = os.path.getsize(f'./pruned/dscnn_chkp_best/saved_model.pb')
    print()
    print(f'acc: {test_acc2}, size: {msize} Inference Latency {end}ms')
    print()
    '''






















print("End.")
