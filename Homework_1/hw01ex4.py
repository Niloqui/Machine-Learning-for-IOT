import argparse
import pandas as pd
import tensorflow as tf
import time
from datetime import datetime
import os

def _bytes_feature(value):                                                      
  """Returns a bytes_list from a string / byte."""                              
  if isinstance(value, type(tf.constant(0))):                                   
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  # ~ return tf.train.Feature(int64_list=tf.train.Int64List(value=audio.numpy().astype(int).flatten().tolist()))   

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input path", type=str)
parser.add_argument("--output", help="output path", type=str)
parser.add_argument('-v',nargs='?', default=False, const=True)
args = parser.parse_args()

out_filename = args.output
in_filename = args.input
verbose = args.v

df = pd.read_csv(in_filename+'/samples.csv', header = None)

with tf.io.TFRecordWriter(out_filename) as writer:
    for i in range(df.iloc[:,0].size):
        raw_date = ",".join([df.iloc[i,0],df.iloc[i,1]])
        date = datetime.strptime(raw_date, '%d/%m/%y,%H:%M:%S')
        posix_date = time.mktime(date.timetuple())      

        raw_audio = tf.io.read_file(in_filename+'/'+df.iloc[i,4])
        audio = raw_audio
        # ~ audio, sample_rate = tf.audio.decode_wav(
                    # ~ raw_audio,
                    # ~ desired_channels=1,  # mono
                    # ~ desired_samples=48000 * 1)
        
        datetime_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(posix_date)])) 
        temperature = tf.train.Feature(int64_list=tf.train.Int64List(value=[df.iloc[i,2]]))
        humidity = tf.train.Feature(int64_list=tf.train.Int64List(value=[df.iloc[i,3]]))

        mapping = {'datetime': datetime_feature,
                   'temperature': temperature,
                   'humidity': humidity,
                   'audio': _bytes_feature(audio)}
        example = tf.train.Example(features=tf.train.Features(feature=mapping))
        writer.write(example.SerializeToString())
        
if verbose == True:     
    print(os.path.getsize(out_filename))
            
