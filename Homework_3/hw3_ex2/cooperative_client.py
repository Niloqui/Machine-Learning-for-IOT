import paho.mqtt.client as PahoMQTT
import argparse
import os
import shutil
import time as t
import json
import numpy as np
import pandas as pd
import tensorflow_model_optimization as tfmot
import tensorflow as tf
from scipy import signal
#import tensorflow.lite as tflite
tflite = tf.lite
keras = tf.keras

class Processor():
    def __init__(self, clientID, dataf, preprocess, broker="mqtt.eclipseprojects.io", port=1883):
        self.broker = broker
        self.port = port
        self.notifier = self
        self.clientID = clientID
        self._pub_topic = f"/{clientID}/data/" #from any publisher any audio preprocessed
        self._sub_topic = "/+/data/+" #this model version
        self._isSubscriber = True
        self._records = pd.DataFrame(columns = ['audio','model','label','score'])
        self._test_ds = os.path.join(dataf,"kws_test_split.txt")
        self._ground_truth = []
        self._LABELS = os.path.join(dataf,"labels.txt")
        self._dataf = dataf
        self._preprocess = preprocess

        
        # create an instance of paho.mqtt.client
        self._paho_mqtt = PahoMQTT.Client(clientID, False) 

        # register the callback
        self._paho_mqtt.on_connect = self.myOnConnect
        self._paho_mqtt.on_message = self.myOnMessageReceived


    def myOnConnect (self, paho_mqtt, userdata, flags, rc):
        print("Connected to %s with result code: %d" % (self.broker, rc))

    def myOnMessageReceived (self, paho_mqtt , userdata, msg):
        # A new message is received
        self.notifier.notify (msg.topic, msg.payload)
        self.store_record(msg.topic,msg.payload)


    def myPublish (self, topic, msg):
        # if needed, you can do some computation or error-check before publishing
        topic = "/".join([self._pub_topic,topic])
        print("publishing topic '%s'" % (topic))
        # publish a message with a certain topic
        self._paho_mqtt.publish(topic, msg, 2)

    def mySubscribe (self, topic=None):
        if topic:
            self._sub_topic = topic
        # if needed, you can do some computation or error-check before subscribing
        print("subscribing to %s" % (self._sub_topic))
        # subscribe for a topic
        self._paho_mqtt.subscribe(self._sub_topic, 2)

        # just to remember that it works also as a subscriber
        self._isSubscriber = True


    def start(self):
        #manage connection to broker
        self._paho_mqtt.connect(self.broker , self.port)
        self._paho_mqtt.loop_start()

        print("subscribing to %s" % (self._sub_topic))
        # subscribe for a topic
        self._paho_mqtt.subscribe(self._sub_topic, 2)

        # just to remember that it works also as a subscriber
        self._isSubscriber = True
        self.read()

    def stop (self):
        if (self._isSubscriber):
            # remember to unsuscribe if it is working also as subscriber 
            self._paho_mqtt.unsubscribe(self._topic)

        self._paho_mqtt.loop_stop()
        self._paho_mqtt.disconnect() 
              
    def store_record(self,topic,data):
        tops = topic.split(os.path.separator())
        model = tops[1]
        audio = tops[-1]
        label = data['label']
        score = data['ts']
        self._records.append(audio,model,label,score)
        
    def print_result(self):
        #recs = self._records.drop_column(['model'])
        self._LABELS = np.array(SELF._LABELS)
        recs = self._records.GroupBy(['audio','label'])['score'].sum()
        predictions = recs.GroupBy(['audio'])['score'].max()
        accuracy = predictions[predictions == self._LABELS]
        print(f'Accuracy: {accuracy}%')
        self.stop()
        
    def preprocess(self,audio_path):
        audio_path = os.path.join(self._dataf,audio_path)
        parts = tf.strings.split(audio_path, os.path.sep)
        idx = audio_path.split(os.path.sep)[-1]
        label = parts[-2]
        label = tf.argmax(label == self._LABELS)
        label = label.numpy()
        
        audio_binary = tf.io.read_file(audio_path.replace("\n",''))
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)
        zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio,zero_padding],0)
        sample = audio.numpy()
        # Resampling
        if self._preprocess['resampling_rate']:
            sample = signal.resample_poly(sample, 1, 16000 // self._preprocess['resampling_rate'])
        
        sample = tf.convert_to_tensor(sample, dtype=tf.float32)

        # STFT
        stft = tf.signal.stft(sample, self._preprocess['frame_length'], self._preprocess['frame_step'], fft_length=self._preprocess['frame_length'])
        spectrogram = tf.abs(stft)
        
        
        if self._preprocess['mfccs'] is False and resize > 0:
            # Resize (optional)
            spectrogram = tf.reshape(spectrogram, [1, self._preprocess['num_frames'], self._preprocess['num_spectrogram_bins'], 1])
            spectrogram = tf.image.resize(spectrogram, [self._preprocess['resize'], self._preprocess['resize']])
            data = f'{spectrogram}'
        else:
            # MFCC (optional)
            mel_spectrogram = tf.tensordot(spectrogram, self._preprocess['linear_to_mel_weight_matrix'], 1)
            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
            mfccs = mfccs[..., :self._preprocess['num_coefficients']]
            mfccs = tf.reshape(mfccs, [1, self._preprocess['num_frames'], self._preprocess['num_coefficients'], 1])
            data = f'{mfccs}'
        return idx, data, label


    def read(self):
        f = open(self._test_ds, "r")
        test_set = f.readlines()
        f.close()
        f = open(self._LABELS, "r")
        self._LABELS = f.readlines()
        f.close()
        for audio_path in test_set:
            idx, data, label = self.preprocess(audio_path)
            self.myPublish(idx,json.dumps(data))
            self._ground_truth.append(label)

        
            
### Reading arguments
parser = argparse.ArgumentParser()
parser.add_argument('--id', default="speech processor", type=str, help='id of the processor')
parser.add_argument('--datadir', default="../", type=str, help='path to dataset definition')
args = parser.parse_args()
clientID = args.id
datadir = args.datadir

preprocess = {
    'sampling_rate'     :   16000, 
    'resampling_rate'   :   8000, 
    'frame_length'      :   1920, 
    'frame_step'        :   960,
    'num_mel_bins'      :   40,
    'lower_freq'        :   20,
    'upper_freq'        :   4000, 
    'num_coefficients'  :   10,
    'mfccs'             :   True,
    'resize'            :   32
}
preprocess['num_frames'] = (preprocess['resampling_rate'] - preprocess['frame_length']) // preprocess['frame_step'] + 1
preprocess['num_spectrogram_bins'] = preprocess['frame_length'] // 2 + 1
preprocess['linear_to_mel_weight_matrix'] = tf.signal.linear_to_mel_weight_matrix(
                                                        preprocess['num_mel_bins'], 
                                                        preprocess['num_spectrogram_bins'], 
                                                        preprocess['resampling_rate'], 
                                                        preprocess['lower_freq'], 
                                                        preprocess['upper_freq'])



proc = Processor(clientID,datadir,preprocess)
proc.start()

