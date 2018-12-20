import os
import cfg
import time
import numpy as np
import tensorflow as tf
from core import utils
from core.yolov3 import YOLO_V3
from core.utils import ImageDataGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'



"""
Main Part of the finetuning Script.
"""
# Load data on the cpu
print("Loading data...")
with tf.device('/cpu:0'):
    train_iterator = ImageDataGenerator(batch_size=1, shuffle=True)
    images, true_boxes, true_labels = train_iterator.iterator.get_next()
    anchors = utils.get_anchors(cfg.anchors_path)


# Initialize model
model = YOLO_V3()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Loading pre_trained weights
    print("Laoding weights ...")
    sess.run(utils.load_weights(tf.global_variables(scope='yolov3'), cfg.weights_file))

    step = 0
    while True:
        # train loop
        image, bboxes, labels = sess.run(train_iterator.iterator.get_next())
        y_true = utils.preprocess_true_boxes(bboxes, labels, anchors, cfg.num_classes)
        _, loss = sess.run([model.train_op, model.loss], feed_dict={
                                                            model.x_input: image,
                                                            model.y_input_13: y_true[0],
                                                            model.y_input_26: y_true[1],
                                                            model.y_input_52: y_true[2],
                                                            model.learning_rate: 0.001
                                                            })
        print("step: {} loss: {}".format(step, loss))
        step += 1

