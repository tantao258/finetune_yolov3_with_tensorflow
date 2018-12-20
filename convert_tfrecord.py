import numpy as np
import tensorflow as tf


def make_tfrecord():
    A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    B = np.array([[9, 10], [11, 12], [13, 14], [15, 16]])
    C = [A, B]

    tfrecord_file = "./test.tfrecord"
    with tf.python_io.TFRecordWriter(tfrecord_file) as record_writer:
        for item in C:
            item = item.tostring()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item])),
                }
            ))
            record_writer.write(example.SerializeToString())

# make_tfrecord()

def parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'x':  tf.FixedLenFeature([], dtype=tf.string),
        }
    )
    x = tf.decode_raw(features['x'], tf.int64)
    x = tf.cast(x, tf.int64)
    return x


dataset = tf.data.TFRecordDataset(filenames="./test.tfrecord")
dataset = dataset.map(parser, num_parallel_calls=10)
dataset = dataset.repeat().batch(1)
iterator = dataset.make_one_shot_iterator()


with tf.Session() as sess:
    for i in range(2):
        x = sess.run(iterator.get_next())
        print(x)
        print("-------")