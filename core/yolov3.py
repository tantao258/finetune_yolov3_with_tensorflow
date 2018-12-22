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
    def __init__(self, multi_gpu=False, num_gpu=4):
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

        if multi_gpu == False:
            with tf.variable_scope('yolov3', reuse=tf.AUTO_REUSE):
                # train
                self.feature_maps = forward(self.x_input, self.anchors, is_training=True)
                # validation
                self.feature_maps_val = forward(self.x_input, self.anchors, is_training=False)

            with tf.variable_scope("loss"):
                self.loss = compute_loss(self.feature_maps, self.boxes_true, self.anchors)
                self.loss_coord = self.loss[0]
                self.loss_sizes = self.loss[1]
                self.loss_confs = self.loss[2]
                self.loss_class = self.loss[3]

            with tf.variable_scope("train"):
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                var_list = [v for v in tf.trainable_variables()]
                gradients = tf.gradients(sum(self.loss), var_list)
                self.grads_and_vars = list(zip(gradients, var_list))
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)

                with tf.control_dependencies(update_ops):
                    self.train_op = optimizer.apply_gradients(grads_and_vars=self.grads_and_vars,
                                                              global_step=self.global_step)

            with tf.name_scope("prediction"):
                self.prediction = predict(self.feature_maps_val, self.anchors)

        else:
            with tf.variable_scope("train"):
                self.batch_size = tf.shape(self.x_input)[0] // num_gpu
                with tf.variable_scope(tf.get_variable_scope()):
                    self.global_step = tf.Variable(0, name="global_step", trainable=False)
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    var_list = [v for v in tf.trainable_variables()]

                    optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)

                    tower_grads = []
                    loss_list = []
                    for i in range(num_gpu):
                        with tf.device("/gpu:%d" % i):
                            with tf.name_scope("tower_%d" % i):
                                x_input = self.x_input[i * self.batch_size:(i + 1) * self.batch_size]
                                boxes_true = self.boxes_true[i * self.batch_size:(i + 1) * self.batch_size]
                                feature_maps = forward(x_input, self.anchors, is_training=True)
                                loss = compute_loss(feature_maps, boxes_true, self.anchors)

                                gradients = optimizer.compute_gradients(sum(loss), var_list)
                                tower_grads.append(gradients)
                                loss_list.append(loss)

                    self.loss = tf.reduce_mean(loss_list, axis=0)
                    self.loss_coord = self.loss[0]
                    self.loss_sizes = self.loss[1]
                    self.loss_confs = self.loss[2]
                    self.loss_class = self.loss[3]

                    gradients = utils.average_gradients(tower_grads)
                    self.grads_and_vars = list(zip(gradients, var_list))
                    with tf.control_dependencies(update_ops):
                        self.train_op = optimizer.apply_gradients(grads_and_vars=self.grads_and_vars,
                                                                  global_step=self.global_step)