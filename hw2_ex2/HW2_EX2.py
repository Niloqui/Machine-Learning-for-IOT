import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import time as t

from tensorflow import convert_to_tensor, float32, tensordot
from tensorflow import abs as tfabs
from tensorflow.io import serialize_tensor, write_file
from tensorflow.math import log
from tensorflow.signal import linear_to_mel_weight_matrix,mfccs_from_log_mel_spectrograms, stft

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
mfccs = True

if not os.path.exists('data/mini_speech_commands'):
    zip_path = tf.keras.utils.get_file(
        origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.','data', 'mini_speech_commands')
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
n = len(filenames)

train_files = filenames[:int(n*0.8)]
val_files = filenames[int(n*0.8):int(n*0.9)]
test_files = filenames[int(n*0.9):]

LABELS = np.array(tf.io.gfile.listdir(str(data_dir))) 
LABELS = [label for label in LABELS if label != 'README.md']

class SignalGenerator:
    def __init__(self, labels, sampling_rate=16000, frame_length=1920 , frame_step=960, num_mel_bins=40,
                 lower_freq=20, upper_freq=48000, num_coefficients=10, mfccs=False):
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.mel_inputs =  [num_mel_bins, None, sampling_rate, lower_freq, upper_freq]
        self.mfccs_coeff = num_coefficients
        self.labels=labels
        self.sampling_rate=sampling_rate
        num_spectrogram_bins = (frame_length) // 2 + 1

        if mfccs:
            self.l2mel_matrix = linear_to_mel_weight_matrix(
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
        tfstft = stft(audio, frame_length=self.frame_length, frame_step=self.frame_step,fft_length=self.frame_length)
        spectrogram = tf.abs(tfstft)
        return spectrogram

    def get_mfcc(self, spectrogram):
        mel_spectrogram = tensordot(spectrogram, self.l2mel_matrix, 1)
        log_mel_spectrogram = log(mel_spectrogram + 1e-6)
        mfccs = mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :self.mfccs_coeff]
        return mfccs

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
        spectrogram = tf.expand_dims(spectrogram, -1)
        mfccs = get_mfcc(spectrogram)
        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()

        if train:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds

generator = SignalGenerator(LABELS)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)

stride = [2,2] if not mfccs else [2,1]

MLP = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(units=256,activation=keras.activations.relu),
    keras.layers.Dense(units=256,activation=keras.activations.relu),
    keras.layers.Dense(units=256,activation=keras.activations.relu),
    keras.layers.Dense(units=len(LABELS))
])

CNN = keras.Sequential([
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

DSCNN = keras.Sequential([
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
models = [MLP,CNN,DSCNN]
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = keras.metrics.SparseCategoricalAccuracy()
cp_callback = keras.callbacks.ModelCheckpoint(
    #'./callback_test_chkp/chkp_{epoch:02d}',
    './callback_test_chkp/chkp_best',
    monitor='val_loss',
    verbose=0, 
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    save_freq='epoch'
)

for i,model in enumerate(models):
    model.compile(optimizer='adam',loss=loss, metrics=[metric])
    model.fit(train_ds, batch_size=32, epochs=2, validation_data=val_ds,callbacks=[cp_callback])
    model.summary()
    start = t.time()
    test_acc, test_acc2 = model.evaluate(test_ds, verbose=2)
    end = t.time() - start
    print(f'acc: {test_acc}, size: {os.path.getsize('./callback_test_chkp/chkp_best')} Inference Latency {end}ms')


