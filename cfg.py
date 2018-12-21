# default:
input_size = 416    # 608

# default:
num_classes = 80

# default:
max_boxes_num = 20                                   # max number of bboxes in one image
# default:
anchors_path = "./data/yolo_anchors.txt"

# default:
weights_file = "./checkpoint/yolov3.weights"

# default:
font_path = "./data/font/FiraMono-Medium.otf"

# default:
classes_names_file = "./data/coco.names"




# generate dataSet parameter
# default:
dataset_txt = "./data/train_data/quick_train_data.txt"

# default:
tfrecord_save_path = "./data/train_data/"



# img detection
sample_file = "./data/demo_data/road.jpeg"



# video detection
video_path = "./data/demo_data/test_video.mp4"
# video_path = 0           # use camera



# train parameter
leaky_relu = 0.1
batch_norm_decay = 0.9
NO_OBJECT_SCALE = 1.0
OBJECT_SCALE = 5.0
COORD_SCALE = 1.0
CLASS_SCALE = 1.0


# nms parameter
score_thresh = 0.2               # default: 0.4    score低于这个值得框被丢掉，减小该值会得到更多的框
iou_thresh = 0.3                 # default: 0.5    两个框的IOU超过这个值才认为是一个框， 增大数值大小将得到更多的框
max_boxes = 20