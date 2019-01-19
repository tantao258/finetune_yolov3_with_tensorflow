import cfg
import tensorflow as tf
from PIL import ImageFont, ImageDraw
import colorsys
import numpy as np


def predict(feature_maps, anchors):
    feature_map_1, feature_map_2, feature_map_3 = feature_maps
    feature_map_anchors = [(feature_map_1, anchors[6:9]),
                           (feature_map_2, anchors[3:6]),
                           (feature_map_3, anchors[0:3])]

    results = [get_boxes_confs_scores(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

    boxes_list, confs_list, probs_list = [], [], []
    for result in results:
        boxes, confs, probs = result
        boxes_list.append(boxes)
        confs_list.append(confs)
        probs_list.append(probs)

    boxes = tf.concat(boxes_list, axis=1)
    confs = tf.concat(confs_list, axis=1)
    probs = tf.concat(probs_list, axis=1)

    center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)

    x0 = center_x - width / 2
    y0 = center_y - height / 2
    x1 = center_x + width / 2
    y1 = center_y + height / 2

    boxes = tf.concat([x0, y0, x1, y1], axis=-1)

    # nms
    scores = confs * probs
    boxes, scores, labels = tf_nms(boxes, scores)

    return boxes, scores, labels


# 对特征图解码
def get_boxes_confs_scores(feature_map, anchors, compute_loss=False):

    num_anchors = len(anchors)              # num_anchors=3
    grid_size = tf.shape(feature_map)[1:3]

    scale = tf.cast([cfg.input_size, cfg.input_size] / grid_size, tf.float32)
    rescaled_anchors = [(a[0] / scale[1], a[1] / scale[0]) for a in anchors]                          # scale anchors

    feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], num_anchors, 5 + cfg.num_classes])
    box_centers, box_sizes, confs, probs = tf.split(feature_map, [2, 2, 1, cfg.num_classes], axis=-1)

    box_centers = tf.nn.sigmoid(box_centers)

    grid_x = tf.range(grid_size[0], dtype=tf.int32)
    grid_y = tf.range(grid_size[1], dtype=tf.int32)
    a, b = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])
    x_y_offset = tf.cast(x_y_offset, tf.float32)

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * scale[::-1]
    box_sizes = tf.exp(box_sizes) * rescaled_anchors
    box_sizes = box_sizes * scale[::-1]

    boxes = tf.concat([box_centers, box_sizes], axis=-1)

    if compute_loss:
        return x_y_offset, boxes, confs, probs

    confs = tf.nn.sigmoid(confs)
    probs = tf.nn.sigmoid(probs)

    boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
    confs = tf.reshape(confs, [-1, grid_size[0] * grid_size[1] * 3, 1])
    probs = tf.reshape(probs, [-1, grid_size[0] * grid_size[1] * 3, cfg.num_classes])

    return boxes, confs, probs


# 非极大抑制tf实现
def tf_nms(boxes, scores):
    """
    Note:Applies Non-max suppression (NMS) to set of boxes. Prunes away boxes that have high
                intersection-over-union (IOU) overlap with previously selected boxes.
    :param boxes: tensor of shape [1, 10647, 4] # 10647 boxes
    :param scores: tensor of shape [1, 10647, num_classes], scores of boxes
    :param num_classes: the return value of function `read_coco_names`
    :param max_boxes: integer, maximum number of predicted boxes you'd like, default is 20
    :param score_thresh: real value, if [ highest class probability score < score_threshold]
                       then get rid of the corresponding box
    :param iou_thresh: real value, "intersection over union" threshold used for NMS filtering
    :return:
    """

    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(cfg.max_boxes, dtype='int32')
    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1, 4])               # [10647, 4]
    score = tf.reshape(scores, [-1, cfg.num_classes])    # [10647, 80]
    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(score, tf.constant(cfg.score_thresh))
    # Step 2: Do non_max_suppression for each class
    for i in range(cfg.num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
        filter_boxes = tf.boolean_mask(boxes, mask[:, i])
        filter_score = tf.boolean_mask(score[:, i], mask[:, i])
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=cfg.iou_thresh,
                                                   name='nms_indices')
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32') * i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)

    return boxes, score, label


# 非极大抑制python实现
def py_nms(boxes, scores):
    """
    /*----------------------------------- NMS on cpu ---------------------------------------*/
    Arguments:
        boxes ==> shape [1, 10647, 4]
        scores ==> shape [1, 10647, num_classes]
    """

    boxes = tf.reshape(boxes, [-1, 4])
    scores = tf.reshape(scores, [-1, cfg.num_classes])
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(cfg.num_classes):
        indices = np.where(scores[:, i] >= cfg.score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:, i][indices]

        if len(filter_boxes) == 0:
            continue

        # do non_max_suppression on the cpu
        x1 = filter_boxes[:, 0]
        y1 = filter_boxes[:, 1]
        x2 = filter_boxes[:, 2]
        y2 = filter_boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= cfg.iou_thresh)[0]
            order = order[inds + 1]

        indices = keep[:cfg.max_boxes]

        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32') * i)
    if len(picked_boxes) == 0:
        return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label


# 画框
def draw_boxes(image, boxes, scores, labels, classes, show=True):
    """
    :param boxes, shape of  [num, 4]
    :param scores, shape of [num, ]
    :param labels, shape of [num, ]
    :param image,
    :param classes, the return list from the function `read_coco_names`
    """
    detection_size = [cfg.input_size, cfg.input_size]
    if boxes is None:
        return image
    draw = ImageDraw.Draw(image)

    # draw settings
    font = ImageFont.truetype(font=cfg.font_path, size=np.floor(2e-2 * image.size[1]).astype('int32'))
    hsv_tuples = [(x / len(classes), 0.9, 1.0) for x in range(len(classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    for i in range(len(labels)):  # for each bounding box, do:
        bbox, score, label = boxes[i], scores[i], classes[labels[i]]
        bbox_text = "%s %.2f" % (label, score)
        text_size = draw.textsize(bbox_text, font)

        # convert_to_original_size
        detection_size, original_size = np.array(detection_size), np.array(image.size)
        ratio = original_size / detection_size
        bbox = list((bbox.reshape(2, 2) * ratio).reshape(-1))

        draw.rectangle(bbox, outline=colors[labels[i]])
        text_origin = bbox[:2]-np.array([0, text_size[1]])
        draw.rectangle([tuple(text_origin), tuple(text_origin+text_size)], fill=colors[labels[i]])

        # draw bbox
        draw.text(tuple(text_origin), bbox_text, fill=(0, 0, 0), font=font)

    image.show() if show else None
    return image


def detection_detail(scores, labels, classes):
    result = {}
    for i in range(len(labels)):
        if classes[labels[i]] not in result.keys():
            result[classes[labels[i]]] = []
            result[classes[labels[i]]].append(scores[i])
        else:
            result[classes[labels[i]]].append(scores[i])
    for (key, value) in result.items():
        print("{}: {} scores: {}".format(key, len(value), value))

