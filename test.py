import cfg
from PIL import Image
from core import utils
import tensorflow as tf
from PIL import Image



from core.utils import ImageDataGenerator
train_iterator = ImageDataGenerator(batch_size=1, tfrecord_file="./data/train_data/tfrecord/train.tfrecord")

with tf.Session() as sess:
    for i in range(1):
        image, y_true_13, y_true_26, y_true_52 = sess.run(train_iterator.iterator.get_next())
        # img = image[0, ...]
        # print(img)
        # img1 = Image.fromarray(img)
        # img1.show()
        print(image)
        print(type(image))
        print(image.shape)
        img = image[0, ...]
        print(img)
        print(type(img))
        img1 = Image.fromarray(img, "RGB")
        img1.show()


        print("------------------")