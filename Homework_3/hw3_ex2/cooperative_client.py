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

class Processor():
    def __init__(self, clientID, test_ds, LABELS, broker="mqtt.eclipseprojects.io", port=1883, notifier=self):
        self.broker = broker
        self.port = port
        self.notifier = notifier
        self.conf = conf
        self.clientID = clientID
        self._pub_topic = f"/{clientID}/data/+" #from any publisher any audio preprocessed
        self._sub_topic = "/+/data/+" #this model version
        self._isSubscriber = True
        self._records = pd.DataFrame(culumn_names = ['audio','model','label','score'])
        self._LABELS = LABELS
        
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
        self._topic = topic

    def start(self):
        #manage connection to broker
        self._paho_mqtt.connect(self.broker , self.port)
        self._paho_mqtt.loop_start()
        self.mySubscribe()
        print("subscribing to %s" % (self._sub_topic))
        # subscribe for a topic
        self._paho_mqtt.subscribe(self._sub_topic, 2)

        # just to remember that it works also as a subscriber
        self._isSubscriber = True

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
        
    def print_result():
        #recs = self._records.drop_column(['model'])
        recs = self._records.GroupBy(['audio','label'])['score'].sum()
        predictions = recs.GroupBy(['audio'])['score'].max()
        accuracy = predictions[predictions == self._LABELS]
        print(f'Accuracy: {accuracy}%')
    
seed = args.seed
# Setting seed for random number generation
tf.random.set_seed(seed)
np.random.seed(seed)



inf = Processor(model_name)
inf.start()

