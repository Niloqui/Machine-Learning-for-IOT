import tensorflow as tf

raw_dataset = tf.data.TFRecordDataset("output") #output is the name you gave to the tfrecords file 

def read_tfrecord(serialized_example):
  feature_description = {
          'datetime': tf.io.FixedLenFeature((), tf.int64),
          'temperature': tf.io.FixedLenFeature((), tf.int64),
          'humidity': tf.io.FixedLenFeature((), tf.int64),
          'audio': tf.io.FixedLenFeature((), tf.string)}
  example = tf.io.parse_single_example(serialized_example, feature_description)
  
  audio = example['audio']
  
  return audio

parsed_dataset = raw_dataset.map(read_tfrecord)

for data in parsed_dataset.take(1):
  tf.io.write_file("sperem.wav", data)

