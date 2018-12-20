import cfg
import numpy as np
from PIL import Image
import tensorflow as tf


def resize_image_correct_bbox(image, bboxes, input_shape):
    """
    Parameters:
    -----------
    :param image: the type of `PIL.JpegImagePlugin.JpegImageFile`
    :param input_shape: the shape of input image to the yolov3 network, [416, 416]
    :param bboxes: numpy.ndarray of shape [N,4], N: the number of boxes in one image
                                                 4: x1, y1, x2, y2

    Returns:
    ----------
    image: the type of `PIL.JpegImagePlugin.JpegImageFile`
    bboxes: numpy.ndarray of shape [N,4], N: the number of boxes in one image
    """
    image_size = image.size
    # resize image to the input shape
    image = image.resize(tuple(input_shape))
    # correct bbox
    bboxes[:, 0] = bboxes[:, 0] * input_shape[0] / image_size[0]
    bboxes[:, 1] = bboxes[:, 1] * input_shape[1] / image_size[1]
    bboxes[:, 2] = bboxes[:, 2] * input_shape[0] / image_size[0]
    bboxes[:, 3] = bboxes[:, 3] * input_shape[1] / image_size[1]

    return image, bboxes


def read_coco_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def freeze_graph(sess, output_file, output_node_names):
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        output_node_names,
    )

    with tf.gfile.GFile(output_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("=> {} ops written to {}.".format(len(output_graph_def.node), output_file))


def read_pb_return_tensors(graph, pb_file, return_elements):
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
        input_tensor, output_tensors = return_elements[0], return_elements[1:]

    return input_tensor, output_tensors


def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    :param var_list: list of network variables.
    :param weights_file: name of the binary file.
    :return: list of assign ops
    """
    with open(weights_file, "rb") as fp:
        np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'Conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'Conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                           bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(
                tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


def preprocess_true_boxes(true_boxes, true_labels, anchors, num_classes):
    """
    Preprocess true boxes to training input format
    Parameters:
    -----------
    :param true_boxes: numpy.ndarray of shape [N, T, 4]
                        N: the number of images,
                        T: the number of boxes in each image.
                        4: coordinate => x_min, y_min, x_max, y_max
    :param true_labels: class id
    :param input_shape: the shape of input image to the yolov3 network, [416, 416]
    :param anchors: array, shape=[9,2], 9: the number of anchors, 2: width, height
    :param num_classes: integer, for coco dataset, it is 80
    Returns:
    ----------
    y_true: list(3 array), shape like yolo_outputs, [N, 13, 13, 3, 85]
                           13:cell szie,
                           3:number of anchors
                           85: box_centers, box_sizes, confidence, probability
    """

    input_shape = np.array([cfg.input_size, cfg.input_size], dtype=np.int32)
    num_images = true_boxes.shape[0]
    num_layers = len(anchors) // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    grid_sizes = [input_shape // 32, input_shape // 16, input_shape // 8]

    # trans (x_min, y_min, x_max, y_max) to center and wh
    boxes_centers = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2  # the center of box
    boxes_sizes = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # the width and height of box

    # the boxes centers and sizes with origin scale
    true_boxes[..., 0:2] = boxes_centers
    true_boxes[..., 2:4] = boxes_sizes

    y_true_13 = np.zeros(shape=[num_images, grid_sizes[0][0], grid_sizes[0][1], 3, 5 + num_classes], dtype=np.float32)
    y_true_26 = np.zeros(shape=[num_images, grid_sizes[1][0], grid_sizes[1][1], 3, 5 + num_classes], dtype=np.float32)
    y_true_52 = np.zeros(shape=[num_images, grid_sizes[2][0], grid_sizes[2][1], 3, 5 + num_classes], dtype=np.float32)

    y_true = [y_true_13, y_true_26, y_true_52]
    anchors = np.expand_dims(anchors, 0)  # expand_dims: (9, 2)  -----> (1, 9, 2)
    anchors_max = anchors / 2.
    anchors_min = -anchors_max
    valid_mask = boxes_sizes[..., 0] > 0

    for b in range(num_images):  # for each image, do:
        # Discard zero rows.
        wh = boxes_sizes[b, valid_mask[b]]
        if len(wh) == 0:
            continue

        # compute IOU between true box and anchors
        wh = np.expand_dims(wh, -2)
        boxes_max = wh / 2.  # ( w/2,  h/2)
        boxes_min = -boxes_max  # (-w/2, -h/2)

        intersect_mins = np.maximum(boxes_min, anchors_min)
        intersect_maxs = np.minimum(boxes_max, anchors_max)
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor, start=0):

            for l in range(num_layers):

                if n not in anchor_mask[l]:
                    continue

                i = np.floor(true_boxes[b, t, 1] / input_shape[::-1] * grid_sizes[l][0]).astype('int32') - 1
                j = np.floor(true_boxes[b, t, 0] / input_shape[::-1] * grid_sizes[l][1]).astype('int32') - 1
                k = anchor_mask[l].index(n)
                c = true_labels[b, t].astype('int32')

                y_true[l][b, i, j, k, 0:4] = true_boxes[b, t, 0:4]
                y_true[l][b, i, j, k, 4] = 1
                y_true[l][b, i, j, k, 5+c] = 1

        return i


def read_image_box_from_text(text_path):
    """
    :param text_path
    :returns : {image_path:(bboxes, labels)}
                bboxes -> [N,4],(x1, y1, x2, y2)
                labels -> [N,]
    """
    data = {}
    with open(text_path, 'r') as f:
        boxes_num_list = []
        for line in f.readlines():
            example = line.split(' ')
            boxes_num = len(example[1:]) // 5
            boxes_num_list.append(boxes_num)

        max_boxes_num = np.max(boxes_num_list)

    with open(text_path, 'r') as f:
        data_count =0               # count total number of image
        error_data_count = 0        # count error number of image

        for line in f.readlines():
            example = line.split(' ')
            image_path = example[0]
            image = Image.open(image_path)

            boxes_num = len(example[1:]) // 5
            bboxes = np.zeros([boxes_num, 4], dtype=np.float32)
            labels = np.zeros([boxes_num, ], dtype=np.int64)
            for i in range(boxes_num):
                # judge the correction of data by coordinates and the wh of image
                if float(example[i * 5 + 2]) < 0. or float(example[i * 5 + 3]) < 0. or \
                        float(example[i * 5 + 4]) > float(image.size[0]) or float(example[i * 5 + 5]) > float(image.size[1]):
                    error_data_count += 1
                    data_count += 1
                    continue
                labels[i] = example[1 + i * 5]
                bboxes[i] = example[2 + i * 5: 6 + i * 5]
            data[image_path] = bboxes, labels
            data_count += 1
        print("total image: {}, error image: {}".format(data_count, error_data_count))
        return data


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(-1, 2)


class ImageDataGenerator(object):
    def __init__(self, batch_size=1, shuffle=True):
        self.anchors = tf.constant(get_anchors(cfg.anchors_path), dtype=tf.float32)
        data_set = tf.data.TFRecordDataset(filenames="./test.tfrecord")
        data_set = data_set.map(self.parser, num_parallel_calls=10)
        if shuffle:
            data_set = data_set.shuffle(buffer_size=1000)
        data_set = data_set.repeat().batch(batch_size=batch_size)
        self.iterator = data_set.make_one_shot_iterator()

    def parser(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature(shape=(), dtype=tf.string),
                'bboxes': tf.FixedLenFeature(shape=(), dtype=tf.string),
                'labels': tf.VarLenFeature(dtype=tf.int64),
            })

        with tf.name_scope("image_process"):
            image = tf.image.decode_jpeg(features['image'], channels=3)
            image = tf.image.convert_image_dtype(image, tf.uint8)
            image_size = tf.to_float(tf.shape(image))[::-1]     # [channel, width, height]
            image = tf.image.resize_images(image, size=(cfg.input_size, cfg.input_size))

        with tf.name_scope("y_true_process"):
            bboxes = tf.decode_raw(features['bboxes'], tf.float32)
            bboxes = tf.reshape(bboxes, (-1, 4))
            labels = features["labels"].values

            xx1 = tf.reshape(bboxes[..., 0] / tf.cast((image_size[1] / cfg.input_size), tf.float32), (-1, 1))  # image_shape = [h, w]
            yy1 = tf.reshape(bboxes[..., 1] / tf.cast((image_size[2] / cfg.input_size), tf.float32), (-1, 1))
            xx2 = tf.reshape(bboxes[..., 2] / tf.cast((image_size[1] / cfg.input_size), tf.float32), (-1, 1))
            yy2 = tf.reshape(bboxes[..., 3] / tf.cast((image_size[2] / cfg.input_size), tf.float32), (-1, 1))
            bboxes = tf.concat([xx1, yy1, xx2, yy2], axis=1)

        return image, bboxes, labels, image_size
