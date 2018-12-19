import os
import sys
import wget
import time
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from core import yolov3, utils


"""
Configuration Part.
"""
# Parameters
tf.app.flags.DEFINE_string("ckpt_file", "./checkpoint/yolov3.ckpt" , "")
tf.app.flags.DEFINE_string("weights_path", "./checkpoint/yolov3.weights", "")
tf.app.flags.DEFINE_bool("convert", True, "if True, conver yolov3.weights to ckpt model")
tf.app.flags.DEFINE_bool("freeze", True, "")
tf.app.flags.DEFINE_integer("image_size", 416, "one of [416, 608]")
tf.app.flags.DEFINE_float("iou_threshold", 0.3, "")
tf.app.flags.DEFINE_float("score_threshold", 0.2, "The score_threshold for gpu nms")
FLAGS = tf.app.flags.FLAGS

classes = utils.read_coco_names("./data/coco.names")
num_classes = len(classes)
print("=> the input image size is [{}, {}]".format(FLAGS.image_size, FLAGS.image_size))

# initialize the yolov3 model
model = yolov3.yolov3(num_classes)

with tf.Graph().as_default() as graph:
    sess = tf.Session(graph=graph)
    inputs = tf.placeholder(tf.float32, [1, FLAGS.image_size, FLAGS.image_size, 3])

    with tf.variable_scope('yolov3'):
        feature_map = model.forward(inputs)

    feature_map_1, feature_map_2, feature_map_3 = feature_map
    print("feature_map_1: {}".format(feature_map_1))
    print("feature_map_2: {}".format(feature_map_2))
    print("feature_map_3: {}".format(feature_map_3))
    print('------------------------------------------')

    for i, k in enumerate(tf.global_variables(), start=1):
        print(i, k.name)
    classes = utils.read_coco_names("./data/coco.names")
    num_classes = len(classes)
    img = Image.open("./data/demo_data/road.jpeg")
    img_resized = np.array(img.resize(size=(FLAGS.image_size, FLAGS.image_size)), dtype=np.float32)
    img_resized = img_resized / 255.

    feature_map_1 = feature_map[0]
    import numpy as np
    pp = sess.run(feature_map_1, feed_dict={inputs: np.expand_dims(img_resized, 0)})
    print(pp)


    # boxes, confs, probs = model.predict(feature_map)
    # scores = confs * probs
    #
    # print("boxes: {}".format(boxes))
    # print("scores: {}".format(scores))
    # print('------------------------------------------')
    #
    # boxes, scores, labels = utils.gpu_nms(boxes, scores, num_classes, 20, FLAGS.score_threshold, FLAGS.iou_threshold)
    # print("boxes: {}".format(boxes))
    # print("scores: {}".format(scores))
    # print("labels: {}".format(labels))
    # print('------------------------------------------')



    saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

    if FLAGS.convert:
        if not os.path.exists(FLAGS.weights_path):
            url = 'https://pjreddie.com/media/files/yolov3.weights'
            if not  os.path.exists("./checkpoint/yolov3.weights"):
                print("{} does not exists ! ".format(FLAGS.weights_path))
            print("=> It will take a while to download it from {}".format(url))
            print('=> Downloading yolov3 weights ... ')
            wget.download(url, FLAGS.weights_path)

        load_ops = utils.load_weights(tf.global_variables(scope='yolov3'), FLAGS.weights_path)
        sess.run(load_ops)
        saver.save(sess, save_path=FLAGS.ckpt_file)
        print('ckpt model havs been saved in path: {}'.format(FLAGS.ckpt_file))

    if FLAGS.freeze:
        saver.restore(sess, FLAGS.ckpt_file)
        print('=> checkpoint file restored from ', FLAGS.ckpt_file)
        utils.freeze_graph(sess, './checkpoint/yolov3_cpu_nms.pb', ["concat_9", "mul_9"])
        utils.freeze_graph(sess, './checkpoint/yolov3_gpu_nms.pb', ["concat_10", "concat_11", "concat_12"])
        utils.freeze_graph(sess, './checkpoint/yolov3_feature.pb', ["yolov3/yolo-v3/feature_map_1",
                                                                    "yolov3/yolo-v3/feature_map_2",
                                                                    "yolov3/yolo-v3/feature_map_3",])