import cfg
import tensorflow as tf
from core import utils
from core.common import conv2d_fixed_padding
from core.forward_function import forward
from core.loss_function import compute_loss
from core.detect_function import predict

slim = tf.contrib.slim


class DarkNet_53(object):
    def __init__(self, inputs):
        inputs = conv2d_fixed_padding(inputs, 32, 3, strides=1)
        inputs = conv2d_fixed_padding(inputs, 64, 3, strides=2)
        inputs = self.darknet53_block(inputs, 32)
        inputs = conv2d_fixed_padding(inputs, 128, 3, strides=2)

        for i in range(2):
            inputs = self.darknet53_block(inputs, 64)

        inputs = conv2d_fixed_padding(inputs, 256, 3, strides=2)

        for i in range(8):
            inputs = self.darknet53_block(inputs, 128)

        self.route_1 = inputs
        inputs = conv2d_fixed_padding(inputs, 512, 3, strides=2)

        for i in range(8):
            inputs = self.darknet53_block(inputs, 256)

        self.route_2 = inputs
        inputs = conv2d_fixed_padding(inputs, 1024, 3, strides=2)

        for i in range(4):
            inputs = self.darknet53_block(inputs, 512)

        self.route_3 = inputs

        self.outputs = [self.route_1, self.route_2, self.route_3]

    def darknet53_block(self, inputs, filters):
        shortcut = inputs
        inputs = conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = conv2d_fixed_padding(inputs, filters * 2, 3)

        inputs = inputs + shortcut
        return inputs


class YOLO_V3(object):
    def __init__(self):
        self.anchors = utils.get_anchors(cfg.anchors_path)

        with tf.name_scope("input"):
            self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, cfg.input_size, cfg.input_size, 3], name="x_input")
            self.y_input_13 = tf.placeholder(dtype=tf.float32,
                                             shape=[None, cfg.input_size/32, cfg.input_size/32, 3, 5 + cfg.num_classes],
                                             name="y_input_13")
            self.y_input_26 = tf.placeholder(dtype=tf.float32,
                                             shape=[None, cfg.input_size/16, cfg.input_size/16, 3, 5 + cfg.num_classes],
                                             name="y_input_13")
            self.y_input_52 = tf.placeholder(dtype=tf.float32,
                                             shape=[None, cfg.input_size/8, cfg.input_size/8, 3, 5 + cfg.num_classes],
                                             name="y_input_13")
            self.boxes_true = [self.y_input_13, self.y_input_26, self.y_input_52]
            self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

        with tf.variable_scope('yolov3', reuse=tf.AUTO_REUSE):
            # train
            self.feature_maps = forward(self.x_input, self.anchors, is_training=True)
            # validation
            self.feature_maps_val = forward(self.x_input, self.anchors, is_training=False)

        with tf.variable_scope("loss"):
            loss = compute_loss(self.feature_maps, self.boxes_true, self.anchors)
            self.loss = sum(loss)

        with tf.variable_scope("train"):
            optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)

        with tf.name_scope("prediction"):
            self.prediction = predict(self.feature_maps_val, self.anchors)
