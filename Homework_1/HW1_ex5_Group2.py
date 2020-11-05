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
    #os.mkdir(output_folder)
    #raise NameError('output folder already exists, choose another folder')

sample_format = pyaudio.paInt16 # Number of bits per sample
sample_rate = 48000 # Number of samples per second
resample_rate = 16000

chunk = int(sample_rate / 2) # Record in chunks of 1024 samples
channels = 1
seconds = 1

ffs = range(0, int(sample_rate * seconds / chunk))

l = 0.040
frame_length = int(sample_rate * l)

s = 0.020
frame_step = int(sample_rate * s)

p = pyaudio.PyAudio()  # Create an interface to PortAudio

microphone_name = "USB Microphone: Audio"
dev_index = -1 # USB microphone index

performances = pd.DataFrame(columns=['Record','Resample','Stft','Mfccs','Preprocessing','TOTAL'])

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

output = []
for n in range(number_of_sample):
    start = t.time()
    print('Recording audio', str(n))
    stream.start_stream()
    frames = [] # Initialize array to store frames
    
    # Store data in chunks for 1 seconds
    for i in ffs:
        data = stream.read(chunk)
        frames.append(data)
    
    # Stop and close the stream
    stream.stop_stream()
    print('Finished recording')
    
    t_record = t.time()
    #print(f"Time needed to record: {round((t_record - start)*1000, 2)} ms")
    
    ######################### MFCC
    frames = b''.join(frames)
    frames_io = io.BytesIO(frames)
    frames_io_buf = frames_io.getbuffer()
    frame = np.frombuffer(frames_io_buf, dtype=np.uint16)
    
    audio = signal.resample_poly(frame, resample_rate, sample_rate)
    tf_audio = tf.convert_to_tensor(audio, dtype=tf.float32)
    tf_audio = 2 * tf_audio / 65535 - 1
    
    t_resample = t.time()
    #print(f"Time needed to resample: {round((t_resample - t_record)*1000, 2)} ms")
    
    stft = tf.signal.stft(tf_audio, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)
    t_stft = t.time()
    
    num_spectrogram_bins = spectrogram.shape[-1]
    linear_to_mel_weight_matrix = \
        tf.signal.linear_to_mel_weight_matrix(40, num_spectrogram_bins, 16000, 20, 4000)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    
    mel_spectrogram.set_shape( spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]) )
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :10]
    
    f_res = f'{output_folder}/mfccs{n}.bin'
    mfccs_ser = tf.io.serialize_tensor(mfccs)
    tf.io.write_file(f_res, mfccs_ser)
    
    t_mfccs = t.time()
    #print(f"Time needed to mfccs: {round((t_conversion - t_resample)*1000, 2)} ms")
    
    #print(f"Time needed for prepocessing: {round((t_conversion - t_record)*1000, 2)} ms")
    #print(f"Time needed for everything: {round((t_conversion - start)*1000, 2)} ms", end="\n\n")
    performances.loc[n] = [(t_record - start)*1000,
                        (t_resample - t_record)*1000,
                        (t_stft - t_resample)*1000,
                        (t_mfccs - t_stft)*1000,
                        (t_mfccs - t_record)*1000,
                        (t_mfccs - start)*1000]


# Terminate the PortAudio interface
stream.close()
p.terminate()

print(performances.round(2))


print("End.")