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
#import tensorflow.lite as tflite
tflite = tf.lite
keras = tf.keras

class InferEngine:
    def __init__(self, model, conf={}, broker="mqtt.eclipseprojects.io", port=1883):
        self.broker = broker
        self.port = port
        self.notifier = self
        self.conf = conf
        if model is not None:
            self.model = model
            self.interpreter = tf.lite.Interpreter(model_path=model)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()

        self.clientID = "inference-"+model
        self._sub_topic = "/+/data/+" #from any publisher any audio preprocessed
        self._pub_topic = f"/{self.clientID}/data/" #this model version
        self._isSubscriber = True

        # create an instance of paho.mqtt.client
        self._paho_mqtt = PahoMQTT.Client(self.clientID, False) 

        # register the callback
        self._paho_mqtt.on_connect = self.myOnConnect
        self._paho_mqtt.on_message = self.myOnMessageReceived


    def myOnConnect(self, paho_mqtt, userdata, flags, rc):
        print("Connected to %s with result code: %d" % (self.broker, rc))

    def myOnMessageReceived(self, paho_mqtt , userdata, msg):
        # A new message is received
        self._paho_mqtt.notify(msg.topic, msg.payload)
        print(f'received {msg.topic} from {paho_mqtt}-{userdata}')
        if(msg.topic.endswith('stop')):
            self.stop()
            return
        self.run_inference(msg.topic,msg.payload['data'])


    def myPublish (self, topic, msg):
        # if needed, you can do some computation or error-check before publishing
        topic = "/".join(self._pub_topic,topic)
        print("publishing '%s' with topic '%s'" % (msg, topic))
        # publish a message with a certain topic
        self._paho_mqtt.publish(topic, msg, 2)

    def mySubscribe (self, topic):
        # if needed, you can do some computation or error-check before subscribing
        print("subscribing to %s" % (topic))
        # subscribe for a topic
        self._paho_mqtt.subscribe(topic, 2)

        # just to remember that it works also as a subscriber
        self._isSubscriber = True
        self._sub_topic = topic
        print("done")

    def start(self):
        #manage connection to broker
        self._paho_mqtt.connect(self.broker , self.port)
        self._paho_mqtt.loop_start()
        self.mySubscribe(self._sub_topic)

    def stop (self):
        if (self._isSubscriber):
            # remember to unsuscribe if it is working also as subscriber 
            self._paho_mqtt.unsubscribe(self._topic)

        self._paho_mqtt.loop_stop()
        self._paho_mqtt.disconnect() 
              
    def run_inference(self,topic,data):
        pubtopic = topic.split(os.path.separator)[-1]
        input_tensor = data
        if self.interpreter:
            self.interpreter.set_tensor(input_details[0]['index'], input_tensor)
            start_inf = time.time()
            self.interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
        label, second = np.argsort(output_data)
        score_margin = output_data[label] - output_data[second]
        trust_score = score_margin * self.conf[label]
        
        end = time.time()
        tot_latency.append(end - start)

        if args.model is None:
            start_inf = end

        inf_latency.append(end - start_inf)
        time.sleep(0.1)

        print('Inference Latency {:.2f}ms'.format(np.mean(inf_latency)*1000.))
        print(f"Predicted label {label}: confidence {self.conf[label]}, score margin {scm}, trust score: {trust_score}")
        msg = {"label":label,"ts":trust_score}
        self.myPublish(pubtopic,msg)

### Reading arguments
parser = argparse.ArgumentParser()
parser.add_argument('--version', default=0, type=int, help='Model version')
args = parser.parse_args()
version = args.version
model_name = f'{version}.tflite'
inf = InferEngine(model_name,broker="192.168.1.195")
try:
    inf.start()
except Exception as e:
    print(e)
