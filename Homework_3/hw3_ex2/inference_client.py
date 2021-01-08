import paho.mqtt.client as PahoMQTT
import argparse
import os
import shutil
import time as t

import numpy as np
import pandas as pd
import tensorflow_model_optimization as tfmot
import tensorflow as tf
#import tensorflow.lite as tflite
tflite = tf.lite
keras = tf.keras

class MyMQTT:
    def __init__(self, clientID, broker, port, notifier):
        self.broker = broker
        self.port = port
        self.notifier = notifier
        self.clientID = clientID

        self._topic = ""
        self._isSubscriber = True

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


    def myPublish (self, topic, msg):
        # if needed, you can do some computation or error-check before publishing
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

    def stop (self):
        if (self._isSubscriber):
            # remember to unsuscribe if it is working also as subscriber 
            self._paho_mqtt.unsubscribe(self._topic)

        self._paho_mqtt.loop_stop()
        self._paho_mqtt.disconnect() 
        
        
        
class InferEngine():
    def __init__(self, clientID, model):
        # create an instance of MyMQTT class
        self.clientID = clientID
        # Give as last parameter the object itself so MyQTT.py client can invoke the notify method
        self.myMqttClient = MyMQTT(self.clientID, "mqtt.eclipseprojects.io", 1883, self) 
        if model is not None:
            self.model = model
            self.interpreter = tf.lite.Interpreter(model_path=args.model)
            self.interpreter.allocate_tensors()

            self.input_details = interpreter.get_input_details()
            self.output_details = interpreter.get_output_details()


    def run(self):
        # if needed, perform some other actions befor starting the mqtt communication
        print("running %s" % (self.clientID))
        self.myMqttClient.start()

    def end(self):
        # if needed, perform some other actions befor ending the software
        print("ending %s" % (self.clientID))
        self.myMqttClient.stop ()

    # As said in MyMQTT.py this class must have a notify method
    # in which we process our messages.
    def notify(self, topic, msg):
        # manage here your received message. You can perform some error-check here  
        print("received '%s' under topic '%s'" % (msg, topic))
        
    def subs():
        self.myMqttClient.mySubscribe(f"/devID/data/{audio_index}")
        
        
    def run_inference():
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
        print(f"Predicted label {label}: confidence {self.conf[label]}, score margin {scm}, trust score: {trust_score}"

### Reading arguments
parser = argparse.ArgumentParser()
parser.add_argument('--version', default=0, type=int, help='Model version')

model_name = f'{version}.tflite'
seed = args.seed

# Setting seed for random number generation
tf.random.set_seed(seed)
np.random.seed(seed)



inf = InferEngine(f"inference model-{version}")
inf.run()

