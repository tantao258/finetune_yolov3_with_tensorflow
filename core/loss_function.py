import cfg
import tensorflow as tf
from core.detect_function import get_boxes_confs_scores


def compute_loss(feature_maps, boxes_true, anchors):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    anchors = [anchors[6:9], anchors[3:6], anchors[0:3]]
    loss = 0
    for i, feature_map in enumerate(feature_maps):
        loss += loss_layer(feature_map, boxes_true[i], anchors[i])

    return loss


def loss_layer(feature_map_i, y_true_i, anchors_i, ignore_thresh=.5):
    """

    :param feature_map_i: the feature_map with shape [None, 13, 13, 5, 80] or [None, 26, 26, 5, 80] or [None, 52, 52, 5, 80]
    :param y_true_i: the y_true_i with origin size   [None, 13, 13, 5, 80] or [None, 26, 26, 5, 80] or [None, 52, 52, 5, 80]
    :param anchors_i: anchors_i  with origin size    [3, 2]
    :param ignore_thresh:
    :return:
    """

    grid_size = tf.shape(feature_map_i)[1:3]   # (13, 13)
    scale = tf.cast([cfg.input_size, cfg.input_size] // grid_size, tf.float32)    # [416//13, 416//13]

    # predict with feature_map_i, the prediction result has been translated to origin size
    pred_result = get_boxes_confs_scores(feature_map_i, anchors_i, cfg.num_classes, compute_loss=True)
    xy_offset, pred_boxes, pred_boxes_confs, pred_boxes_class = pred_result

    true_boxes_xy = y_true_i[..., 0:2]   # true center xy origin scale
    true_boxes_wh = y_true_i[..., 2:4]  # true width and height origin scale

    pred_boxes_xy = pred_boxes[..., 0:2]  # predict center xy origin scale
    pred_boxes_wh = pred_boxes[..., 2:4]  # predict width and height origin scale

    # caculate iou between true boxes and pred boxes
    intersect_xy1 = tf.maximum(true_boxes_xy - true_boxes_wh / 2.0,
                               pred_boxes_xy - pred_boxes_wh / 2.0)     # ===> (intersect_x1, intersect_y1)
    intersect_xy2 = tf.minimum(true_boxes_xy + true_boxes_wh / 2.0,
                               pred_boxes_xy + pred_boxes_wh / 2.0)     # ===> (intersect_x2, intersect_y2)
    intersect_wh = tf.maximum(intersect_xy2 - intersect_xy1, 0.)        # ===> (intersect_w,  intersect_h)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]
    pred_area = pred_boxes_wh[..., 0] * pred_boxes_wh[..., 1]

    union_area = true_area + pred_area - intersect_area
    iou_scores = tf.truediv(intersect_area, union_area)     # iou_scores.shape [None, 13, 13, 3]
    iou_scores = tf.expand_dims(iou_scores, axis=-1)        # [None, 13, 13, 3] ===>>> [None, 13, 13, 3, 1]

    true_boxes_conf = y_true_i[..., 4:5]

    conf_mask = tf.to_float(iou_scores < 0.6) * (1 - y_true_i[..., 4:5]) * cfg.NO_OBJECT_SCALE + y_true_i[..., 4:5] * cfg.OBJECT_SCALE
    # if iou_scores > 0.6 and no object?
    # cell have object       conf = 1 * OBJECT_SCALE
    # cell have no object:  iou_scores < 0.6     conf = 1 * NO_OBJECT_SCALE
    #                       iou_scores > 0.6     conf = 0 ?

    # adjust x and y => relative position to the containing cell
    true_boxes_xy_scaled = true_boxes_xy / scale - xy_offset        # true_boxes_xy_scaled  [0--1]
    pred_boxes_xy_scaled = pred_boxes_xy / scale - xy_offset

    # adjust w and h => relative size to the containing cell
    true_boxes_wh_scaled = true_boxes_wh / (anchors_i * scale)
    pred_boxes_wh_scaled = pred_boxes_wh / (anchors_i * scale)

    true_boxes_wh_scaled = tf.where(condition=tf.equal(true_boxes_wh_scaled, 0),
                                    x=tf.ones_like(true_boxes_wh_scaled),
                                    y=true_boxes_wh_scaled)

    pred_boxes_wh_scaled = tf.where(condition=tf.equal(pred_boxes_wh_scaled, 0),
                                    x=tf.ones_like(pred_boxes_wh_scaled),
                                    y=pred_boxes_wh_scaled)

    true_boxes_wh_logit = tf.log(true_boxes_wh_scaled)             # true_boxes_wh_logit   [0--1]
    pred_boxes_wh_logit = tf.log(pred_boxes_wh_scaled)

    # adjust class probabilities
    class_mask = y_true_i[..., 4:5] * cfg.CLASS_SCALE
    # class mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = y_true_i[..., 4:5] * cfg.COORD_SCALE

    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))     # compute the number of bbox have object
    nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

    loss_coord = tf.reduce_sum(tf.square(true_boxes_xy_scaled - pred_boxes_xy_scaled) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_sizes = tf.reduce_sum(tf.square(tf.sqrt(true_boxes_wh_logit) - tf.sqrt(pred_boxes_wh_logit)) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_confs = tf.reduce_sum(tf.square(true_boxes_conf - pred_boxes_confs) * conf_mask) / (nb_conf_box + 1e-6) / 2.
    loss_class = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_i[..., 5:], logits=pred_boxes_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

    loss = loss_coord + loss_sizes + loss_confs + loss_class

    return loss





























