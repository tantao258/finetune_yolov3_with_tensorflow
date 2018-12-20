from __future__ import print_function

import os
import cfg
import wget
import numpy as np
from PIL import Image
import tensorflow as tf
from core import utils
from core.yolov3 import YOLO_V3

"""
Configuration Part.
"""
# Parameters
tf.app.flags.DEFINE_string("ckpt_file", "./checkpoint/yolov3.ckpt", "")
tf.app.flags.DEFINE_bool("convert", True, "if True, conver yolov3.weights to ckpt model")
tf.app.flags.DEFINE_bool("freeze", False, "")
FLAGS = tf.app.flags.FLAGS

classes = utils.read_coco_names(cfg.classes_names_file)
print("=> the input image size is [{}, {}]".format(cfg.input_size, cfg.input_size))

# input
img = Image.open("./data/demo_data/road.jpeg")
img_resized = np.array(img.resize(size=(cfg.input_size, cfg.input_size)), dtype=np.float32)
img_resized = img_resized / 255.

# initialize the yolov3 model
model = YOLO_V3()
feature_map_1, feature_map_2, feature_map_3 = model.feature_maps_val
prediction = model.prediction
print("feature_map_1: {}".format(feature_map_1))
print("feature_map_2: {}".format(feature_map_2))
print("feature_map_3: {}".format(feature_map_3))
print("predicton: {}".format(prediction))
print('------------------------------------------')


sess = tf.Session()

sess.run(tf.global_variables_initializer())

# Loading pre_trained weights
print("Loading weights ...")
sess.run(utils.load_weights(tf.global_variables(scope='yolov3'), cfg.weights_file))

saver = tf.train.Saver()

if FLAGS.convert:
    if not os.path.exists(cfg.weights_file):
        url = 'https://pjreddie.com/media/files/yolov3.weights'
        if not os.path.exists("./checkpoint/yolov3.weights"):
            print("{} does not exists ! ".format(cfg.weights_file))
        print("=> It will take a while to download it from {}".format(url))
        print('=> Downloading yolov3 weights ... ')
        wget.download(url, cfg.weights_file)
        sess.run(utils.load_weights(tf.global_variables(scope='yolov3'), cfg.weights_file))
    saver.save(sess, save_path=FLAGS.ckpt_file)
    print('ckpt model havs been saved in path: {}'.format(FLAGS.ckpt_file))

if FLAGS.freeze:
    saver.restore(sess, FLAGS.ckpt_file)
    print('=> checkpoint file restored from ', FLAGS.ckpt_file)
    utils.freeze_graph(sess, './checkpoint/yolov3_prediction.pb', ["concat_10", "concat_11", "concat_12"])