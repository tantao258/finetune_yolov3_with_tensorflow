import cfg
from PIL import Image
from core import utils
import tensorflow as tf


from core.utils import ImageDataGenerator
train_iterator = ImageDataGenerator(batch_size=1, tfrecord_file="./test.tfrecord")

with tf.Session() as sess:
    for i in range(1):
        image, y_true_13, y_true_26, y_true_52 = sess.run(train_iterator.iterator.get_next())

        print(image)
        print(y_true_13.shape)
        print(y_true_26.shape)
        print(y_true_52.shape)
        print("------------------")