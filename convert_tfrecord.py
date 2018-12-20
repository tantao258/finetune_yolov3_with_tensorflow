import cfg
from PIL import Image
from core import utils
import tensorflow as tf


def make_tfrecord(tfrecord_path="./test.tfrecord"):
    dataset = utils.read_image_box_from_text(cfg.dataset_txt)
    image_paths = list(dataset.keys())
    images_num = len(image_paths)
    print(">> Processing %d images" % images_num)

    with tf.python_io.TFRecordWriter(tfrecord_path) as record_writer:

        for img_path in image_paths:
            image = tf.gfile.FastGFile(img_path, 'rb').read()
            bboxes, labels = dataset[img_path]
            bboxes = bboxes.tostring()

            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    'bboxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bboxes])),
                    'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
                }
            ))
            record_writer.write(example.SerializeToString())
        print("making tfrecord file completed.")

#
# if __name__ == '__main__':
#
#     tfrecord_path = "./test.tfrecord"
#     make_tfrecord(tfrecord_path=tfrecord_path)



from core.utils import ImageDataGenerator
train_iterator = ImageDataGenerator(batch_size=1, shuffle=True)
anchors = utils.get_anchors(cfg.anchors_path)

with tf.Session() as sess:
    for i in range(1):
        image, bboxes, labels, image_size = sess.run(train_iterator.iterator.get_next())
        y_true = utils.preprocess_true_boxes(bboxes, labels, anchors, cfg.num_classes)
