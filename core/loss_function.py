import cfg
import tensorflow as tf
from core.detect_function import get_boxes_confs_scores


def compute_loss(feature_maps, y_true, anchors):
    '''
    param:
        feature_maps: returned feature_map list by `forward` function: [feature_map_1, feature_map_2, feature_map_3]
        y_true: input y_true by the tf.data pipeline
    '''
    loss_coord, loss_sizes, loss_confs, loss_class = 0., 0., 0., 0.
    anchors = [anchors[6:9], anchors[3:6], anchors[0:3]]

    for i in range(len(feature_maps)):
        loss = loss_layer(feature_maps[i], y_true[i], anchors[i])
        loss_coord += loss[0]
        loss_sizes += loss[1]
        loss_confs += loss[2]
        loss_class += loss[3]

    return [loss_coord, loss_sizes, loss_confs, loss_class]


def loss_layer(feature_map_i, y_true, anchors):

    grid_size = tf.shape(feature_map_i)[1:3]
    scale = tf.cast([cfg.input_size, cfg.input_size] / grid_size, dtype=tf.float32)

    pred_result = get_boxes_confs_scores(feature_map_i, anchors, compute_loss=True)
    xy_offset, pred_boxes, pred_box_conf, pred_box_class = pred_result

    ###########
    # get mask
    ###########
    # shape: take 416x416 input image and 13*13 feature_map for example:
    # [batch_size, 13, 13, 3, 1]
    object_mask = y_true[..., 4:5]
    # shape: [N, 13, 13, 3, 4] & [N, 13, 13, 3] ==> [V, 4]
    # V: num of true gt box
    valid_true_boxes = tf.boolean_mask(y_true[..., 0:4], tf.cast(object_mask[..., 0], 'bool'))

    # shape: [V, 2]
    valid_true_box_xy = valid_true_boxes[:, 0:2]
    valid_true_box_wh = valid_true_boxes[:, 2:4]
    # shape: [N, 13, 13, 3, 2]
    pred_box_xy = pred_boxes[..., 0:2]
    pred_box_wh = pred_boxes[..., 2:4]

    # calc iou
    # shape: [N, 13, 13, 3, V]
    iou = broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)

    # shape: [N, 13, 13, 3]
    best_iou = tf.reduce_max(iou, axis=-1)

    # get_ignore_mask
    ignore_mask = tf.cast(best_iou < 0.5, tf.float32)
    # shape: [N, 13, 13, 3, 1]
    ignore_mask = tf.expand_dims(ignore_mask, -1)

    # get xy coordinates in one cell from the feature_map
    # numerical range: 0 ~ 1
    # shape: [N, 13, 13, 3, 2]
    true_xy = y_true[..., 0:2] / scale[::-1] - xy_offset
    pred_xy = pred_box_xy / scale[::-1] - xy_offset

    # get_tw_th
    # numerical range: 0 ~ 1
    # shape: [N, 13, 13, 3, 2]
    true_tw_th = y_true[..., 2:4] / anchors
    pred_tw_th = pred_box_wh / anchors
    # for numerical stability
    true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                          x=tf.ones_like(true_tw_th),
                          y=true_tw_th)
    pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                          x=tf.ones_like(pred_tw_th),
                          y=pred_tw_th)
    true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
    pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

    # box size punishment:
    # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
    # shape: [N, 13, 13, 3, 1]
    box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(cfg.input_size, tf.float32)) * (y_true[..., 3:4] / tf.cast(cfg.input_size, tf.float32))

    ############
    # loss_part
    ############
    # shape: [N, 13, 13, 3, 1]
    xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale) / cfg.batch_size
    wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale) / cfg.batch_size

    # shape: [N, 13, 13, 3, 1]
    conf_pos_mask = object_mask
    conf_neg_mask = (1 - object_mask) * ignore_mask
    conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_box_conf)
    conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_box_conf)
    conf_loss = tf.reduce_sum(conf_loss_pos + conf_loss_neg) / cfg.batch_size

    # shape: [N, 13, 13, 3, 1]
    class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:], logits=pred_box_class)
    class_loss = tf.reduce_sum(class_loss) / cfg.batch_size

    return xy_loss, wh_loss, conf_loss, class_loss


def broadcast_iou(true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
    '''
    maintain an efficient way to calculate the ios matrix between ground truth true boxes and the predicted boxes
    note: here we only care about the size match
    '''
    # shape:
    # true_box_??: [V, 2]
    # pred_box_??: [N, 13, 13, 3, 2]

    # shape: [N, 13, 13, 3, 1, 2]
    pred_box_xy = tf.expand_dims(pred_box_xy, -2)
    pred_box_wh = tf.expand_dims(pred_box_wh, -2)

    # shape: [1, V, 2]
    true_box_xy = tf.expand_dims(true_box_xy, 0)
    true_box_wh = tf.expand_dims(true_box_wh, 0)

    # [N, 13, 13, 3, 1, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2]
    intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                true_box_xy - true_box_wh / 2.)
    intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                true_box_xy + true_box_wh / 2.)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

    # shape: [N, 13, 13, 3, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [N, 13, 13, 3, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    # shape: [1, V]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]

    # [N, 13, 13, 3, V]
    iou = intersect_area / (pred_box_area + true_box_area - intersect_area)

    return iou