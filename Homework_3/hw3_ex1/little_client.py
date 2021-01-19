import argparse
import numpy as np
from subprocess import call
import tensorflow as tf
from scipy import signal
import os
import time
import socket
import requests
import json
import base64
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--server_ip', type=str, default="192.168.1.54",
        help='local adress o')
parser.add_argument('--server_port', type=int, default=8080,
        help='server port')		
parser.add_argument('--rate', type=int, default=16000,
        help='sampling rate after resampling')
parser.add_argument('--length', type=int, default=640,
        help='stft window legnth in number of samples')
parser.add_argument('--stride', type=int, default=320,
        help='stft window stride in number of samples')
parser.add_argument('--bins', type=int, default=40,
        help='number of mel bins')
parser.add_argument('--coeff', type=int, default=10,
        help='number of MFCCs')
parser.add_argument('--th', type=float, default=0.40,
        help='threshold for score margin')
args = parser.parse_args()


call('sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
            shell=True)

rate = args.rate
length = args.length
stride = args.stride
num_mel_bins = args.bins
num_coefficients = args.coeff
threshold = args.th

data_dir = os.path.join('.','data', 'mini_speech_commands')
if not os.path.exists(data_dir):
    zip_path = tf.keras.utils.get_file(
        origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')

# raspberry ip 
RASPIP = '192.168.1.54'#'198.168.1.60'

model_path = './little.tflite' # load small model

num_frames = (rate - length) // stride + 1
num_spectrogram_bins = length // 2 + 1

linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, rate, 20, 4000)


interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

f = open("../kws_test_split.txt", "r")
test_set = f.readlines()
f.close()

data_dir = os.path.join('.','data', 'mini_speech_commands')
labels_file = open("../labels.txt", "r")
LABELS = labels_file.read()
labels_file.close()
labels = np.array(LABELS.split(" "))


num = 0
num_corr = 0
net_cost = 0
calls = 0
for audio_path in test_set:
    num += 1
    #if num%100 == 0:
        #print(num)

    parts = tf.strings.split(audio_path, os.path.sep)
    label = parts[-2]
    label = tf.argmax(label == labels)
    label = label.numpy()
    
    audio_binary = tf.io.read_file(audio_path.replace("\n",''))
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio_string = tf.io.encode_base64(audio_binary).numpy().decode()
    audio = tf.squeeze(audio, axis=1)
    zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio,zero_padding],0)
    #sample = audio.numpy()


    # Resampling
    #sample = signal.resample_poly(sample, 1, 16000 // rate)
    #sample = tf.convert_to_tensor(sample, dtype=tf.float32)

    # STFT
    stft = tf.signal.stft(audio, length, stride,
            fft_length=length)
    spectrogram = tf.abs(stft)

    # MFCC
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :num_coefficients]
    mfccs = tf.reshape(mfccs, [1, num_frames, num_coefficients, 1])
    input_tensor = mfccs


    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probs = tf.nn.softmax(output_data)

    now = datetime.datetime.now()
    timestamp = int(now.timestamp())

    if (np.sort(probs[0])[-1] - np.sort(probs[0])[-2]) <= threshold:
        calls+=1

        body = {
                'bn': f'http://{RASPIP}/',
                'bt': timestamp,
                'e': [{'n': 'audio', 'u': '/', 't': 0, 'vd': audio_string}]
                }

        url = f'http://{args.server_ip}:{args.server_port}/'
        r = requests.put(url, json=body)

        if r.status_code == 200:
            rbody = r.json()
            #prob = rbody['probability']
            prediction = rbody['label']
            
        else:
            print('Error')
            print(r.text)
    
        net_cost += len(audio_string)
        if labels[label] == prediction:
            num_corr += 1

    else:
        if label == np.argmax(probs[0]):
            num_corr += 1

    time.sleep(0.1)
    
#print('calls to big model {}'.format(calls))
print(f'Accuracy: {num_corr*100/num}%')
print(f'Communication Cost: {net_cost/(2**20)} MB')
