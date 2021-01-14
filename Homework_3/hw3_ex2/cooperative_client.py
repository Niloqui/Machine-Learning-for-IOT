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
import datetime
from scipy import signal
import base64
#import tensorflow.lite as tflite
tflite = tf.lite
keras = tf.keras

class Processor():
    def __init__(self, clientID, dataf, preprocess, broker="mqtt.eclipseprojects.io", port=1883):
        self.clientID = clientID
        self.broker = broker
        self.port = port
        
        self._pub_topic = f"/PoliTO/ML4IOT/Group2/{clientID}/data/" #from any publisher any audio preprocessed
        self._sub_topic = f"/PoliTO/ML4IOT/Group2/{clientID}/results/#" # in '#' there will be the result of model '#'
        # Example: /PoliTO/ML4IOT/Group2/id1234/data/4
        #self._sub_topic = "/PoliTO/ML4IOT/Group2/+/data/#" #this model version
        self._isSubscriber = True
        self._records = pd.DataFrame(columns = ['audio','model','label','score'])
        self._test_ds = os.path.join(dataf, "kws_test_split.txt")
        self._ground_truth = []
        self._LABELS = os.path.join(dataf, "labels.txt")
        self._dataf = dataf
        self._preprocess = preprocess
        
        # create an instance of paho.mqtt.client
        self._paho_mqtt = PahoMQTT.Client(clientID, False) 
        
        # register the callback
        self._paho_mqtt.on_connect = self.myOnConnect
        self._paho_mqtt.on_message = self.myOnMessageReceived
        
        self.num_models = 2
        self.num_audio = 20 # 800
        self.num_messages_received = 0
    
    
    def myOnConnect(self, paho_mqtt, userdata, flags, rc):
        print(f"Connected to {self.broker} with result code: {rc}")
        print(f"User data: {userdata}, Flags: {flags}")
    
    ####################################
    ####################################
    ####################################
    ####################################
    def myOnMessageReceived(self, paho_mqtt, userdata, msg):
        # A new message is received
        #print(f'Received {msg.topic} from {paho_mqtt}-{userdata}')
        if msg.topic != self._pub_topic:
            self.store_record(msg.topic, msg.payload)
            print(self.num_messages_received, flush=True)
            
        #### TO-DO: Gestire qui la fine della ricezione dei messaggi
        # Quando sono arrivati tutti i messaggi, print_result(...)
    ####################################
    ####################################
    ####################################
    ####################################
    
    
    '''
    def myPublish (self, status, topic, msg):
        # if needed, you can do some computation or error-check before publishing
        topic = "".join([self._pub_topic, topic])
        print(f"publishing topic {topic}        {status}%",flush=True)
        # publish a message with a certain topic
        self._paho_mqtt.publish(topic, msg, 2)
    '''
    def myPublish (self, msg):
        #topic = "".join([self._pub_topic, topic])
        #print(f"publishing topic {self._pub_topic}",flush=True)
        # publish a message with a certain topic
        self._paho_mqtt.publish(self._pub_topic, msg, 2)
    
    '''
    def mySubscribe (self, topic=None):
        if topic:
            self._sub_topic = topic
        # if needed, you can do some computation or error-check before subscribing
        print("subscribing to %s" % (self._sub_topic))
        # subscribe for a topic
        self._paho_mqtt.subscribe(self._sub_topic, 2)
        
        # just to remember that it works also as a subscriber
        self._isSubscriber = True
    '''
    
    
    def start(self):
        #manage connection to broker
        self._paho_mqtt.connect(self.broker, self.port)
        self._paho_mqtt.loop_start()
        
        # subscribe for a topic
        #print("subscribing to %s" % (self._sub_topic))
        print(f"Subscribing to {self._sub_topic}")
        self._paho_mqtt.subscribe(self._sub_topic, 2)
        
        # just to remember that it works also as a subscriber
        self._isSubscriber = True
        self.read()
        
    def stop(self):
        if (self._isSubscriber):
            # remember to unsuscribe if it is working also as subscriber 
            self._paho_mqtt.unsubscribe(self._sub_topic)
        
        self._paho_mqtt.loop_stop()
        self._paho_mqtt.disconnect()
    
    
    '''
    def store_record(self, topic, data):
        #tops = str(topic).split(os.path.separator())
        tops = str(topic).split(os.path.sep)
        model = tops[1]
        audio = tops[-1]
        label = data['label']
        score = data['ts']
        self._records.append(audio,model,label,score)
    '''
    def store_record(self, topic, data):
        # msg = {'bn': audio_id, "label": label, "ts": trust_score}
        
        #tops = str(topic).split(os.path.separator())
        tops = str(topic).split(os.path.sep)
        data = json.loads(data)
        
        row = {
            'audio': data['bn'],
            'label': data['label'],
            'score': data['ts'],
            'model': tops[-1]
        }
        
        #self._records.append([audio, model, label, score])
        #self._records.loc[self.num_messages_received] = [audio, model, label, score]
        self._records = self._records.append(row, ignore_index=True)
        self.num_messages_received += 1
    
    def print_result(self):
        #recs = self._records.drop_column(['model'])
        #self._LABELS = np.array(self._LABELS)
        
        recs = self._records.groupby(['audio','label'])['score'].sum().reset_index()
        
        idx = recs.groupby(['audio'])['score'].transform(max) == recs['score']
        predictions = recs[idx].sort_values(by=['audio'])['label'].to_numpy()
        
        print(predictions)
        print(self._ground_truth)
        
        accuracy = predictions[predictions == self._ground_truth]
        print(f'Accuracy: {len(accuracy) / len(predictions)}%')
        
        self.stop()
    
    def preprocess(self, audio_path):
        audio_path = os.path.join(self._dataf, audio_path)
        parts = tf.strings.split(audio_path, os.path.sep)
        #idx = audio_path.split(os.path.sep)[-1]
        idx = parts[-1] # Filename
        label = parts[-2]
        label = tf.argmax(label == self._LABELS)
        label = label.numpy()
        resize = self._preprocess['resize']
        
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
            data = spectrogram
        else:
            # MFCC (optional)
            mel_spectrogram = tf.tensordot(spectrogram, self._preprocess['linear_to_mel_weight_matrix'], 1)
            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
            mfccs = mfccs[..., :self._preprocess['num_coefficients']]
            mfccs = tf.reshape(mfccs, [1, self._preprocess['num_frames'], self._preprocess['num_coefficients'], 1])
            data = mfccs
        
        #data = {'data': base64.b64encode(data), 'shape': data.shape}
        #data = json.dumps(data)
        return idx, data, label
    
    
    def read(self):
        #### TO-DO: mandare un messaggio per richiedere il numero di modelli
        ####
        
        f = open(self._LABELS, "r")
        self._LABELS = f.read().split(' ')
        self._LABELS = np.array(self._LABELS)
        f.close()
        
        f = open(self._test_ds, "r")
        test_set = f.readlines()
        #self.num_audio = len(test_set)
        f.close()
        
        for i, audio_path in enumerate(test_set):
            now = datetime.datetime.now()
            timestamp = now.timestamp()
            
            #idx, data, label = self.preprocess(audio_path)
            _, data, label = self.preprocess(audio_path)
            
            data_64 = base64.b64encode(data)
            data_string = data_64.decode()
            
            msg = {
                #'bn': idx,
                'bn': i,
                'bt': timestamp,
                'e': [
                    {'n': 'audio', 'u': '/', 't': 0, 'vd': data_string},
                    {'n': 'shape', 'u': '/', 't': 0, 'v': list(data.shape)}
                ]
            }
            msg = json.dumps(msg)
            
            #data = {'data': base64.b64encode(data), 'shape': data.shape}
            #data = json.dumps(data)
            
            #self.myPublish((i+1)*100/len(test_set), idx, data)
            self.myPublish(msg)
            self._ground_truth.append(label)
            t.sleep(0.1)
            
            if i >= self.num_audio - 1:
                break
            
            if (i + 1) % 100 == 0:
                print(f"Sent {i + 1} / {len(test_set)}")
        
        self._ground_truth = np.array(self._ground_truth)
        if len(test_set) % 100 != 0:
            print(f"Sent {len(test_set)} / {len(test_set)}")
        print(f"End publication.")
        
        #### TO-DO (forse): mandare un messaggio per indicare la fine della publicazione
        ####
        #self.myPublish('[DONE]','stop', None)
    
    def wait_all_results(self):
        # Da rifare meglio
        while self.num_messages_received < self.num_audio * self.num_models:
            t.sleep(1)
        
        pass


### Reading arguments
parser = argparse.ArgumentParser()
parser.add_argument('--id', default="id1234", type=str, help='id of the speech processor')
parser.add_argument('--datadir', default="../", type=str, help='path to dataset definition')
args = parser.parse_args()
clientID = args.id
datadir = args.datadir

#
# TO-DO: Da rivedere questi parametri
#
'''
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
'''
preprocess = {
    'sampling_rate'     :   16000,
    'resampling_rate'   :   16000,
    'frame_length'      :   640,
    'frame_step'        :   320,
    'num_mel_bins'      :   80,
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




#proc = Processor(clientID,datadir,preprocess,broker='192.168.1.195')
proc = Processor(clientID, datadir, preprocess)
proc.start()
proc.wait_all_results()
print("DioPorco.io:1883")
print(proc.num_messages_received)
print(proc._records)
proc.print_result()

