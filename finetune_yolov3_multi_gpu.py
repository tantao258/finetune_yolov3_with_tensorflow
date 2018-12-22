import os
import cfg
import time
import datetime
import tensorflow as tf
from core import utils
from core.yolov3 import YOLO_V3_MULTI_GPU
from core.utils import ImageDataGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

"""
Configuration Part.
"""
# Parameters
tf.app.flags.DEFINE_integer("num_checkpoints", 2, "num_checkpoints(default:3)")
tf.app.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 400, "Save model after this many steps (default: 100)")
FLAGS = tf.app.flags.FLAGS

"""
Main Part of the finetuning Script.
"""
# Load data on the cpu
print("Loading data...")
with tf.device('/cpu:0'):
    train_iterator = ImageDataGenerator(batch_size=20,
                                        tfrecord_file="./data/train_data/tfrecord/train.tfrecord",
                                        )
    next_batch_train = train_iterator.iterator.get_next()

# Initialize model
yolov3 = YOLO_V3_MULTI_GPU()

with tf.Session() as sess:
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "yolov3", timestamp))
    print("Writing to {}\n".format(out_dir))

    # define summary
    grad_summaries = []
    for g, v in yolov3.grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    loss_summaries = []
    loss_coord_summary = tf.summary.scalar("loss_coord", yolov3.loss_coord)
    loss_sizes_summary = tf.summary.scalar("loss_sizes", yolov3.loss_sizes)
    loss_confs_summary = tf.summary.scalar("loss_confs", yolov3.loss_confs)
    loss_class_summary = tf.summary.scalar("loss_class", yolov3.loss_class)
    loss_summaries.append(loss_coord_summary)
    loss_summaries.append(loss_sizes_summary)
    loss_summaries.append(loss_confs_summary)
    loss_summaries.append(loss_class_summary)
    loss_summaries_merged = tf.summary.merge(loss_summaries)
    # acc_summary = tf.summary.scalar("accuracy", yolov3.accuracy)

    # merge all the train summary
    train_summary_merged = tf.summary.merge([loss_summaries_merged, grad_summaries_merged])
    train_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "train"), graph=sess.graph)
    # merge all the dev summary
    val_summary_merged = tf.summary.merge([loss_summaries_merged])
    val_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "val"), graph=sess.graph)

    # checkPoint saver
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "ckpt"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    sess.run(tf.global_variables_initializer())

    # Loading pre_trained weights
    print("Laoding weights ...")
    sess.run(utils.load_weights(tf.global_variables(scope='yolov3'), cfg.weights_file))

    while True:
        # train loop
        image, y_true_13, y_true_26, y_true_52 = sess.run(next_batch_train)
        _, step, train_summaries, loss = sess.run(
            [yolov3.train_op, yolov3.global_step, train_summary_merged, yolov3.loss], feed_dict={yolov3.x_input: image,
                                                                                                 yolov3.y_input_13: y_true_13,
                                                                                                 yolov3.y_input_26: y_true_26,
                                                                                                 yolov3.y_input_52: y_true_52,
                                                                                                 yolov3.learning_rate: 0.001
                                                                                                 })
        train_summary_writer.add_summary(train_summaries, step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step: {}, loss_total: {:g}, loss_coord: {:g}, loss_sizes: {:g}, loss_confs: {:g}, loss_class: {:g}" \
              .format(time_str, step, sum(loss), loss[0], loss[1], loss[2], loss[3]))

        current_step = tf.train.global_step(sess, yolov3.global_step)

        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))
        if current_step == 30:
            exit()



