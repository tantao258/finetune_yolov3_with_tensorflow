import cv2
import cfg
import time
import numpy as np
from core import utils
from PIL import Image
import tensorflow as tf
from core.yolov3 import YOLO_V3
from core.detect_function import predict, draw_boxes


classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)
anchors = utils.get_anchors(anchors_path=cfg.anchors_path)


model = YOLO_V3(80)
feature_maps = model.feature_maps_val


sess = tf.Session()

for i, k in enumerate(tf.global_variables(), start=1):
    print(i, k.name)
sess.run(tf.global_variables_initializer())

# Loading pre_trained weights
print("Laoding weights ...")
sess.run(utils.load_weights(tf.global_variables(scope='yolov3'), cfg.weights_file))

# 捕获摄像头
cap = cv2.VideoCapture(cfg.video_path)

# Prepare for saving the detected video
sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'mpeg')
out = cv2.VideoWriter('./', fourcc, 20.0, sz, True)
counter = 1
while True:

    return_value, frame = cap.read()
    if return_value:
        print(counter)
        counter += 1
        image = Image.fromarray(frame)
    else:
        raise ValueError("No image!")

    img_resized = np.array(image.resize(size=[cfg.input_size, cfg.input_size]), dtype=np.float32)
    img_resized = img_resized / 255.

    boxes, scores, labels = sess.run(predict(model.feature_maps_val, anchors, num_classes),
                                     feed_dict={
                                         model.x_input: np.expand_dims(img_resized, 0)
                                     })
    image = draw_boxes(image, boxes, scores, labels, classes, show=False)
    out.write(np.array(image))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()


