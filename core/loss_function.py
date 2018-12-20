import cfg
import tensorflow as tf
from core.detect_function import get_boxes_confs_scores


def compute_loss(feature_maps, y_true, anchors):
    """
    Note: compute the loss
    Arguments: y_pred, list -> [feature_map_1, feature_map_2, feature_map_3]  the shape of [None, 13, 13, 3*85]. etc
    """
    loss_coord, loss_sizes, loss_confs, loss_class = 0., 0., 0., 0.
    anchors = [anchors[6:9], anchors[3:6], anchors[0:3]]

    for i in range(len(feature_maps)):
        loss = loss_layer(feature_maps[i], y_true[i], anchors[i])
        loss_coord += loss[0]
        loss_sizes += loss[1]
        loss_confs += loss[2]
        loss_class += loss[3]

    loss = tf.Print(loss, [loss_coord], message='loss coord:\t', summarize=1000)
    loss = tf.Print(loss, [loss_sizes], message='loss sizes:\t', summarize=1000)
    loss = tf.Print(loss, [loss_confs], message='loss confs:\t', summarize=1000)
    loss = tf.Print(loss, [loss_class], message='loss class:\t', summarize=1000)

    return [loss_coord, loss_sizes, loss_confs, loss_class]


def loss_layer(feature_map_i, y_true, anchors):

    grid_size = tf.shape(feature_map_i)[1:3]
    stride = tf.cast([416, 416] // grid_size, dtype=tf.float32)

    pred_result = get_boxes_confs_scores(feature_map_i, anchors, compute_loss=True)
    xy_offset, pred_box, pred_box_conf, pred_box_class = pred_result

    true_box_xy = y_true[..., :2]       # absolute coordinate
    true_box_wh = y_true[..., 2:4]      # absolute size

    pred_box_xy = pred_box[..., :2]     # absolute coordinate
    pred_box_wh = pred_box[..., 2:4]    # absolute size

    # caculate iou between true boxes and pred boxes
    intersect_xy1 = tf.maximum(true_box_xy - true_box_wh / 2.0,
                               pred_box_xy - pred_box_xy / 2.0)
    intersect_xy2 = tf.minimum(true_box_xy + true_box_wh / 2.0,
                               pred_box_xy + pred_box_wh / 2.0)
    intersect_wh = tf.maximum(intersect_xy2 - intersect_xy1, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_area = true_area + pred_area - intersect_area
    iou_scores = tf.truediv(intersect_area, union_area)
    iou_scores = tf.expand_dims(iou_scores, axis=-1)

    true_box_conf = y_true[..., 4:5]

    conf_mask = tf.to_float(iou_scores < 0.6) * (1 - y_true[..., 4:5]) * cfg.NO_OBJECT_SCALE
    conf_mask = conf_mask + y_true[..., 4:5] * cfg.OBJECT_SCALE

    true_box_xy = true_box_xy / stride - xy_offset
    pred_box_xy = pred_box_xy / stride - xy_offset

    true_box_wh_logit = true_box_wh / (anchors * stride)
    pred_box_wh_logit = pred_box_wh / (anchors * stride)

    true_box_wh_logit = tf.where(condition=tf.equal(true_box_wh_logit, 0),
                                 x=tf.ones_like(true_box_wh_logit),
                                 y=true_box_wh_logit)
    pred_box_wh_logit = tf.where(condition=tf.equal(pred_box_wh_logit, 0),
                                 x=tf.ones_like(pred_box_wh_logit),
                                 y=pred_box_wh_logit)

    true_box_wh = tf.log(true_box_wh_logit)
    pred_box_wh = tf.log(pred_box_wh_logit)

    class_mask = y_true[..., 4:5] * cfg.CLASS_SCALE
    coord_mask = y_true[..., 4:5] * cfg.COORD_SCALE

    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

    loss_coord = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_sizes = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_confs = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
    loss_class = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:], logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

    return loss_coord, loss_sizes, loss_confs, loss_class

























