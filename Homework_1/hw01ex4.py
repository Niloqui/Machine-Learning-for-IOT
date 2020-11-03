import argparse
import pandas as pd
import tensorflow as tf
import time
from datetime import datetime
import numpy as np

def _bytes_feature(value):                                                      
  """Returns a bytes_list from a string / byte."""                              
  if isinstance(value, type(tf.constant(0))):                                   
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))   

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input path", type=str)
parser.add_argument("--output", help="output path", type=str)
args = parser.parse_args()

out_filename = args.output
in_filename = args.input

df = pd.read_csv(in_filename+'/samples.csv', header = None)

with tf.io.TFRecordWriter(out_filename) as writer:
    for i in range(df.iloc[:,0].size):
        raw_date = ",".join([df.iloc[i,0],df.iloc[i,1]])
        date = datetime.strptime(raw_date, '%d/%m/%y,%H:%M:%S')
        posix_date = time.mktime(date.timetuple())      

        audio = tf.io.read_file(in_filename+'/'+df.iloc[i,4])
        # Convert to a string tensor.
        #wav_encoded = tf.audio.encode_wav(audio, 48000)
        
        datetime_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(posix_date)])) 
        temperature = tf.train.Feature(int64_list=tf.train.Int64List(value=[df.iloc[i,2]]))
        humidity = tf.train.Feature(int64_list=tf.train.Int64List(value=[df.iloc[i,3]]))

        mapping = {'datetime': datetime_feature,
                   'temperature': temperature,
                   'humidity': humidity,
                   'audio': _bytes_feature(audio)}
        example = tf.train.Example(features=tf.train.Features(feature=mapping))
        writer.write(example.SerializeToString())