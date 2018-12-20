import cfg
import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils
from core.detect_function import draw_boxes, detection_detail
from core.yolov3 import YOLO_V3

classes = utils.read_coco_names(cfg.classes_names_file)

# with tf.variable_scope('yolov3'):
model = YOLO_V3()

sess = tf.Session()

for i, k in enumerate(tf.global_variables("yolov3"), start=1):
    print(i, k.name)
sess.run(tf.global_variables_initializer())

# Loading pre_trained weights
print("Loading weights ...")
sess.run(utils.load_weights(tf.global_variables(scope='yolov3'), cfg.weights_file))

while True:
    print("=============================================")

    flag = 0
    while flag == 0:
        img_path = input("input the image path:")
        try:
            img = Image.open(img_path)   # input RGB format
            flag = 1
        except FileNotFoundError:
            print("{} does not exists.".format(img_path))

    img_resized = np.array(img.resize(size=(cfg.input_size, cfg.input_size)), dtype=np.float32)
    img_resized = img_resized / 255.
    print("image shape: {}".format(img_resized.shape))

    boxes, scores, labels = sess.run(model.prediction, feed_dict={model.x_input: np.expand_dims(img_resized, 0)})
    print("nms select: {} boxes".format(len(boxes)))

    detection_detail(scores, labels, classes)
    image = draw_boxes(img, boxes, scores, labels, classes, show=True)
    image.save('./data/demo_data/road_result.jpg')