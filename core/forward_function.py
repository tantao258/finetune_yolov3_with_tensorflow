import cfg
import tensorflow as tf
from core.common import conv2d_fixed_padding

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


def forward(inputs, anchors, is_training=False):
    batch_norm_params = {
            'decay': cfg.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
    }
    with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        biases_initializer=None,
                        activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=cfg.leaky_relu)):
        with tf.variable_scope('darknet-53'):
            route_1, route_2, inputs = DarkNet_53(inputs).outputs

        with tf.variable_scope('yolo-v3'):
            route, inputs = yolo_block(inputs, 512)
            feature_map_1 = detection_layer(inputs, anchors[6:9])
            feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

            inputs = conv2d_fixed_padding(route, 256, 1)
            upsample_size = route_2.get_shape().as_list()
            inputs = up_sample(inputs, upsample_size)
            inputs = tf.concat([inputs, route_2], axis=3)

            route, inputs = yolo_block(inputs, 256)
            feature_map_2 = detection_layer(inputs, anchors[3:6])
            feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

            inputs = conv2d_fixed_padding(route, 128, 1)
            upsample_size = route_1.get_shape().as_list()
            inputs = up_sample(inputs, upsample_size)
            inputs = tf.concat([inputs, route_1], axis=3)

            route, inputs = yolo_block(inputs, 128)
            feature_map_3 = detection_layer(inputs, anchors[0:3])
            feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            feature_maps = [feature_map_1, feature_map_2, feature_map_3]
            return feature_maps


def yolo_block(inputs, filters):
    inputs = conv2d_fixed_padding(inputs, filters * 1, 1)
    inputs = conv2d_fixed_padding(inputs, filters * 2, 3)
    inputs = conv2d_fixed_padding(inputs, filters * 1, 1)
    inputs = conv2d_fixed_padding(inputs, filters * 2, 3)
    inputs = conv2d_fixed_padding(inputs, filters * 1, 1)
    route = inputs
    inputs = conv2d_fixed_padding(inputs, filters * 2, 3)
    return route, inputs


def up_sample(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
    inputs = tf.identity(inputs, name='up_sampled')
    return inputs


def detection_layer(inputs, anchors):
    num_anchors = len(anchors)
    feature_map = slim.conv2d(inputs,
                              num_anchors * (5 + cfg.num_classes), 1,
                              stride=1, normalizer_fn=None,
                              activation_fn=None,
                              biases_initializer=tf.zeros_initializer())
    return feature_map