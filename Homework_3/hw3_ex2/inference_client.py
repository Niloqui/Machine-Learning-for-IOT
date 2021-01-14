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
import time
import base64
#import tensorflow.lite as tflite
tflite = tf.lite
keras = tf.keras

class InferEngine:
    def __init__(self, version, conf={}, broker="mqtt.eclipseprojects.io", port=1883):
        self._isSubscriber = True
        self.broker = broker
        self.port = port
        self.conf = conf
        
        model = f'{version}.tflite'
        if model is not None:
            self.model = model
            self.interpreter = tf.lite.Interpreter(model_path=model)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        
        self.clientID = "inference-" + model
        #self._sub_topic = "/+/data/+" 
        #self._pub_topic = f"/{self.clientID}/data/"
        
        self._sub_topic = "/PoliTO/ML4IOT/Group2/+/data/" # from any publisher any audio preprocessed
        #self._sub_topic = f"/PoliTO/ML4IOT/Group2/id1234/data" # from any publisher any audio preprocessed
        # _pub_topic must be calculated when needed
        #self._pub_topic = f"/PoliTO/ML4IOT/Group2/+/data/{version}" # this model version
        # self._pub_topic_base.replace('+', 'sensorID')
        self._pub_topic_base = f"/PoliTO/ML4IOT/Group2/+/results/{version}"
        
        # create an instance of paho.mqtt.client
        self._paho_mqtt = PahoMQTT.Client(self.clientID, False) 
        
        # register the callback
        self._paho_mqtt.on_connect = self.myOnConnect
        self._paho_mqtt.on_message = self.myOnMessageReceived
        self._paho_mqtt.notify = self.myNotify
    
    def myNotify(self, topic, msg):
        print(topic)
    
    def myOnConnect(self, paho_mqtt, userdata, flags, rc):
        print("Connected to %s with result code: %d" % (self.broker, rc))
    
    def myOnMessageReceived(self, paho_mqtt, userdata, msg):
        # A new message is received
        self._paho_mqtt.notify(msg.topic, msg.payload)
        #print(f'received {msg.topic} from {paho_mqtt}-{userdata}')
        
        ### TO-DO: forse da rifare perchè ho cambiato un po' le cose
        '''
        if(msg.topic.endswith('stop')):
            self._paho_mqtt.unsubscribe(self._sub_topic)
            return
        '''
        self.run_inference(msg.topic, msg.payload)
    
    
    def myPublish(self, topic, msg):
        # if needed, you can do some computation or error-check before publishing
        #topic = "/".join(self._pub_topic, topic)
        #print("Publishing '%s' with topic '%s'" % (msg, topic))
        # publish a message with a certain topic
        self._paho_mqtt.publish(topic, msg, 2)
    
    def mySubscribe(self, topic):
        # if needed, you can do some computation or error-check before subscribing
        print(f"Subscribing to {topic}")
        self._paho_mqtt.subscribe(topic, 2)
        print("done")
        
        # just to remember that it works also as a subscriber
        #self._isSubscriber = True
        #self._sub_topic = topic
    
    def start(self):
        # Manage connection to broker
        print("connecting to:", self.broker, self.port)
        self._paho_mqtt.connect(self.broker, self.port)
        self._paho_mqtt.loop_start()
        
        self.mySubscribe(self._sub_topic)
        #self._paho_mqtt.subscribe(self._sub_topic, 2)
        
        ###############################################
        #while (input()!=''):
        #    time.sleep(1)
        #self.stop()
    
    
    def stop (self):
        if (self._isSubscriber):
            # remember to unsuscribe if it is working also as subscriber 
            self._paho_mqtt.unsubscribe(self._sub_topic)
        
        self._paho_mqtt.loop_stop()
        self._paho_mqtt.disconnect()
     
    def run_inference(self, topic, data):
        sensorID = topic.split("/")[4]
        pubtopic = self._pub_topic_base.replace('+', sensorID)
        
        msg = json.loads(data)
        #print(msg)
        
        '''
        msg = {
                #'bn': idx,
                'bn': i,
                'bt': timestamp,
                'e': [
                    {'n': 'audio', 'u': '/', 't': 0, 'vd': data_string},
                    {'n': 'shape', 'u': '/', 't': 0, 'v': list(data.shape)}
                ]
            }
        '''
        
        audio_id = msg['bn']
        shape = msg['e'][1]['v']
        audio = msg['e'][0]['vd']
        audio = base64.b64decode(audio)
        audio = tf.io.decode_raw(audio, tf.float32)
        audio = tf.reshape(audio, shape)
        
        print("\n", "\n", shape, sep='\n', flush=True)
        print(sensorID, flush=True)
        print(pubtopic, flush=True)
        
        input_tensor = audio
        if self.interpreter:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            #start_inf = time.time()
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        print(output_data, flush=True)
        output_data = output_data[0]
        nump_sorted = np.argsort(output_data)
        label = nump_sorted[0]
        second = nump_sorted[1]
        #second = 0
        
        score_margin = output_data[label] - output_data[second]
        try:
            trust_score = score_margin * self.conf[label]
        except:
            trust_score = score_margin * 1.0
        
        #end = time.time()
        #tot_latency.append(end - start)
        
        #if args.model is None:
        #    start_inf = end
        
        #inf_latency.append(end - start_inf)
        #time.sleep(0.1)
        
        #print('Inference Latency {:.2f}ms'.format(np.mean(inf_latency)*1000.))
        #print(f"Predicted label {label}: confidence {self.conf[label]}, score margin {score_margin}, trust score: {trust_score}")
        print(f"Predicted label {label}: score margin {score_margin}, trust score: {trust_score}")
        
        msg = {'bn': audio_id, "label": int(str(label)), "ts": trust_score}
        msg = json.dumps(msg)
        self.myPublish(pubtopic, msg)

### Reading arguments
parser = argparse.ArgumentParser()
parser.add_argument('--version', default='1', type=str, help='Model version')
args = parser.parse_args()
version = args.version
#model_name = f'{version}.tflite'

#inf = InferEngine(model_name, broker='192.168.1.195')
inf = InferEngine(version)

try:
    inf.start()
    
    while (input()!=''):
        time.sleep(1)
    
    inf.stop()
    
except Exception as e:
    print(e)
