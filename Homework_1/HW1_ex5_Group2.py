import subprocess

performance = ['sudo', 'sh', '-c', 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']
powersave = ['sudo', 'sh', '-c', 'echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']

# subprocess.Popen(powersave)
# subprocess.Popen(performance)

#subprocess.check_call(performance)
subprocess.check_call(powersave)

import os
import argparse
import pyaudio
import time as t
import wave
import tensorflow as tf
import io
import numpy as np
from scipy import signal
from scipy.io import wavfile
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--num-samples", help="num samples", type=int, default=5)
parser.add_argument("--output", help="output folder", type=str, default="./HW1_ex5_Group2_output/")
args = parser.parse_args()

number_of_sample = args.num_samples 
output_folder = args.output
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
else:
    os.system(f"rm -r {output_folder}")
    #raise NameError('output folder already exists, choose another folder')

sample_format = pyaudio.paInt16 # Number of bits per sample
sample_rate = 48000 # Number of samples per second
resample_rate = 16000
downsample = int(sample_rate / resample_rate)

chunk = int(sample_rate / 5) # Record in chunks of 1024 samples
channels = 1
seconds = 1

chunk_readings = range(int(sample_rate * seconds / chunk))

l = 0.040
frame_length = int(resample_rate * l)

s = 0.020
frame_step = int(resample_rate * s)

p = pyaudio.PyAudio()  # Create an interface to PortAudio

microphone_name = "USB Microphone: Audio"
dev_index = -1 # USB microphone index

# Searching the microphone index among all devices
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if microphone_name in dev['name']:
        dev_index = i
        break
print(f"Microphone index: {dev_index}\n")

stream = p.open(format = sample_format,
                channels = channels,
                rate = sample_rate,
                frames_per_buffer = chunk,
                input_device_index = dev_index,
                input = True)
stream.stop_stream()

performances = pd.DataFrame(columns=['Record','Resample','STFT','MFCCs','Saving','Preprocessing','TOTAL'])
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(40, 321, 16000, 20, 4000)
subprocess.check_call(['sudo', 'sh', '-c', 'echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset'])
output = []
for n in range(number_of_sample):
    start = t.time()
    
    #### Record
    #print('Recording audio', str(n))
    stream.start_stream()
    
    subprocess.Popen(powersave)
    frames = []
    
    frames.append(stream.read(chunk*4))
    subprocess.Popen(performance)
    frames.append(stream.read(chunk))
    
    stream.stop_stream()
    #print('Finished recording')
    
    t_record = t.time()
    
    #### Resample
    frame = np.frombuffer(io.BytesIO(b''.join(frames)).getbuffer(), dtype=np.uint16)
    audio = signal.resample_poly(frame, 1, downsample)
    tf_audio = tf.convert_to_tensor(audio, dtype=tf.float32) / 32767 - 1
    
    t_resample = t.time()
    
    #### STFT
    stft = tf.signal.stft(tf_audio, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)
    
    t_stft = t.time()
    
    #### MFCCs
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :10]
    
    t_mfccs = t.time()
    
    #### Saving the output
    #subprocess.Popen(powersave)
    
    f_res = f'{output_folder}/mfccs{n}.bin'
    mfccs_ser = tf.io.serialize_tensor(mfccs)
    tf.io.write_file(f_res, mfccs_ser)
    
    t_savefile = t.time()
    print(t_savefile - start)
    
    #### Record
    # ['Record','Resample','Stft','Mfccs','Preprocessing','TOTAL'])
    
    performances.loc[n] = [(t_record - start) * 1000,
                        (t_resample - t_record) * 1000,
                        (t_stft - t_resample) * 1000,
                        (t_mfccs - t_stft) * 1000,
                        (t_savefile - t_mfccs) * 1000,
                        (t_savefile - t_record) * 1000,
                        (t_savefile - start) * 1000]


subprocess.Popen(powersave)

cpu_usage = subprocess.check_output(['cat', '/sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state'])
cpu_usage = str(cpu_usage).replace("\\n", "\n").replace("b'", "").replace("'", "")

# Terminate the PortAudio interface
stream.close()
p.terminate()

print(cpu_usage)
print(performances.round(2))
