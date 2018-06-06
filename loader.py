import sys
from tensorpack.dataflow import imgaug
import cv2

from tensorpack.tfutils import TowerContext

import argparse
import numpy as np
import os
import scipy

from tensorpack import imgaug

import tensorflow as tf

from imagenet_resnet import Model


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGENET_VAL_PATH = os.path.join(os.environ['INET_DIR'], 'val')
VAL_PATH = 'val.txt'

with open(VAL_PATH) as f:
    val_labels = [int(i.split(' ')[1]) for i in f.read().strip().split('\n')]

# load_img: returns image and label corresponding to an imagenet validation number
#   num: val # 

def load_img(num, img_size=224):
    path_to_img = os.path.join(IMAGENET_VAL_PATH, 'ILSVRC2012_val_%08d.JPEG' % num)
    img = scipy.ndimage.imread(path_to_img, mode='RGB')

    label = val_labels[num - 1]
    img = preprocess(img, image_size=img_size)
    return img.astype(np.float32)/255., np.array(label).astype(np.int32)

_has_been_loaded = False

def _optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

# get model: returns logits, xent of a resnet model of your desired parameters
#   sess: tf session
#   x: 4d batch of images of shape [b, image_size, image_size, 3]
#   y: 1d batch of integer labels of shape [b]
#   checkpoint: path to train **directory**
#   image size: either 75 or 224
#   depth: in the set of {18, 34, 50, 101, 152}
# for details of resnet model, visit:
#   https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet

def get_model(sess, x, y, checkpoint, image_size, depth = 50):
    global _has_been_loaded

    true_shape = [image_size, image_size, 3]
    if len(x.shape) == 2:
        x = tf.reshape(x, [-1] + true_shape)

    # stupid tensor dimensionality inconsistencies
    concordant = [(k.value if not type(k) is int else k) == true_shape[i]
                  for i, k in enumerate(x.shape[1:])]

    assert all(concordant)
    assert len(y.shape) == 1

    assert x.dtype == tf.float32 
    assert y.dtype == tf.int32 

    # NCHW bullshit
    x = tf.reverse(x, axis=[3])

    variables = set(tf.global_variables())
    with TowerContext(tower_name='tower', is_training=False):
        with tf.variable_scope('', reuse=_has_been_loaded):
            resnet = Model(image_size, depth)
            logits, xent = resnet._build_graph((x,y), attack=True)

    model_variables = set(tf.global_variables()) - variables
    if not _has_been_loaded:
        sess.run(tf.variables_initializer(list(model_variables)))
        if os.path.isdir(checkpoint):
            checkpoint = tf.train.latest_checkpoint(checkpoint)

        _optimistic_restore(sess, checkpoint)

    _has_been_loaded = True

    return logits, xent

# preprocess: preprocesses images in resnet style
#   img: [None, None, 3] batch of images
#   image size: either 75 or 224
def preprocess(img, image_size):
    augmentors = [
        imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
        imgaug.CenterCrop((224, 224)),
    ]

    if image_size != 224:
        augmentors.append(imgaug.ResizeShortestEdge(image_size, cv2.INTER_CUBIC))

    for a in augmentors:
        img = a.augment(img)

    return img

