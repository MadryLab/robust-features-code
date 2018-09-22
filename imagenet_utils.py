#!/usr/bin/env python
# File: imagenet_utils.py

import cv2
import numpy as np
import multiprocessing
import tensorflow as tf
from abc import abstractmethod

from tensorpack import imgaug, dataset, ModelDesc, InputDesc
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary

LABEL_RANGES = [(151, 268), (281, 285), (30, 32), (33, 37), (80, 100), (365, 382),
              (389, 397), (118, 121), (300,319)]

# horrendously sloppy coding
# dont think too hard about this
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7')
parser.add_argument('--eps', default=0.0, type=float)
parser.add_argument('--data', help='ILSVRC dataset dir', default='foo')
parser.add_argument('--load', help='load model')
parser.add_argument('--image-size', type=int, choices=[224,75],
                    help='size of images to send to resnet')
parser.add_argument('--checkpoint-dir')
parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
parser.add_argument('--data-format', help='specify NCHW or NHWC',
                    type=str, default='NCHW')
parser.add_argument('-d', '--depth', help='resnet depth',
                    type=int, default=50, choices=[18, 34, 50, 101, 152])
parser.add_argument('--eval', action='store_true')
parser.add_argument('--lp', choices=['2', 'inf'])
parser.add_argument('--mode', choices=['resnet', 'preact', 'se'],
                    help='variants of resnet to use', default='resnet')
args = parser.parse_known_args()[0]

class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """
    def __init__(self, crop_area_fraction=0.08,
                 aspect_ratio_low=0.75, aspect_ratio_high=1.333,
                 target_shape=None):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        num_tries = 10
        for _ in range(num_tries):
            targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
            aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (self.target_shape, self.target_shape),
                                 interpolation=cv2.INTER_CUBIC)
                return out
        # otherwise just take center crop
        out = imgaug.ResizeShortestEdge(self.target_shape, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(self.target_shape).augment(out)
        return out

def fbresnet_augmentor(isTrain, target_shape):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if isTrain:
        augmentors = [
            GoogleNetResize(target_shape=target_shape),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True)
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224))
        ]

        if target_shape != 224:
            augmentors.append(imgaug.ResizeShortestEdge(target_shape, cv2.INTER_CUBIC))

    return augmentors


def get_imagenet_dataflow(
        datadir, name, batch_size,
        augmentors):
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    assert name in ['train', 'val', 'test']
    assert datadir is not None
    assert isinstance(augmentors, list)
    isTrain = name == 'train'
    cpu = min(30, multiprocessing.cpu_count())
    meta_dir = './ilsvrc_metadata'
    if isTrain:
        ds = dataset.ILSVRC12(datadir, name,
                              meta_dir=meta_dir,
                              shuffle=True)
        ds = AugmentImageComponent(ds, augmentors, copy=False)
        ds = PrefetchDataZMQ(ds, cpu)
        ds = BatchData(ds, batch_size, remainder=False)
    else:
        ds = dataset.ILSVRC12Files(datadir, name, meta_dir=meta_dir,
                                   shuffle=False)
        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, cls
        ds = MultiThreadMapData(ds, cpu, mapf, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = PrefetchDataZMQ(ds, 1)
    return ds


def eval_on_ILSVRC12(model, sessinit, dataflow):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    pred = SimpleDatasetPredictor(pred_config, dataflow)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for top1, top5 in pred.get_result():
        batch_size = top1.shape[0]
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))

try:
    EPS = args.eps
    NUM_ITERATIONS = 8
    STEP_SIZE = EPS/NUM_ITERATIONS * 2 # args.eps/3
except:
    pass


class ImageNetModel(ModelDesc):
    weight_decay = 1e-4

    """
    uint8 instead of float32 is used as input type to reduce copy overhead.
    It might hurt the performance a liiiitle bit.
    The pretrained models were trained with float32.
    """
    image_dtype = tf.uint8

    def __init__(self, target_shape, data_format='NCHW', attack_inline=True):
        if data_format == 'NCHW':
            assert tf.test.is_gpu_available()

        self.data_format = data_format
        self.attack_inline = attack_inline
        self.image_shape = target_shape

    def _get_inputs(self):
        return [InputDesc(self.image_dtype, [None, self.image_shape, self.image_shape, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _adv(self, x, y):
        def full_logits(img):
            img = self.image_preprocess(img, bgr=True, attack=True)
            if self.data_format == 'NCHW':
                img = tf.transpose(img, [0, 3, 1, 2])
            with tf.variable_scope('', reuse=True):
                logits = self.get_logits(img)

            return logits
        
        i_0 = tf.constant(0)
        cond = lambda i, _: tf.less(i, NUM_ITERATIONS)

        if args.lp == '2':
            def norm_divisor(v):
                norms = (tf.reduce_sum(v**2, axis=[1,2,3])**(1/2.))[..., None, None, None]
                return norms

            def l2_linf_project(v):
                v = tf.clip_by_value(v, 0., 1.)
                diff = v - x
                norms = norm_divisor(diff)
                normalized = diff/norms * tf.minimum(EPS, norms)
                return x + normalized

            random_point = tf.random_normal(shape=tf.shape(x))
            random_point = random_point/norm_divisor(random_point)
            start_adv = l2_linf_project(x + random_point * EPS)

            initial_vars = [i_0, start_adv]
            def body(i, adv):
                logits = full_logits(adv)
                losses = ImageNetModel.compute_loss_and_error(logits, y, attack=True)
                g, = tf.gradients(losses, adv)
                g = g/norm_divisor(g)
                adv = tf.stop_gradient(l2_linf_project(adv + g * STEP_SIZE)) # g * STEP_SIZE)
                return i + 1, adv

        elif args.lp == 'inf':
            unif = tf.random_uniform(minval=-EPS, maxval=EPS, shape=tf.shape(x))
            start_adv = tf.clip_by_value(x + unif, 0., 1.)
            def linf_project(v):
                v = tf.clip_by_value(v, 0., 1.)
                v = tf.clip_by_value(v, x - EPS, x + EPS)
                return v

            initial_vars = [i_0, start_adv]
            def body(i, adv):
                logits = full_logits(adv)
                losses = ImageNetModel.compute_loss_and_error(logits, y, attack=True)
                g, = tf.gradients(losses, adv)
                g = tf.sign(g)
                adv = tf.stop_gradient(linf_project(adv + g * STEP_SIZE))
                return i + 1, adv

        _, adv = tf.while_loop(cond, body, initial_vars, back_prop=False,
                               parallel_iterations=1)

        return tf.stop_gradient(adv)

    def _build_graph(self, inputs, attack=False, inputs_preprocessed=False):
        name_scope = tf.get_default_graph().get_name_scope()

        # IMG::[[[[24 47 34]]]...]
        image, label = inputs

        if name_scope == 'tower0' or args.eval:
            self.get_logits(tf.cast(tf.transpose(image, [0, 3, 1, 2]), tf.float32))

        if self.attack_inline and attack == False and args.eps > 0: #and 'Inference' in name_scope
            with tf.variable_scope('', reuse=True):
                image = tf.cast(image, dtype=tf.float32)/tf.constant(255.0, dtype=tf.float32)
                image = self._adv(image, label) * 255.0

        # img in [0,255]
        image = self.image_preprocess(image, bgr=True, attack=attack)

        assert image.shape[1] == image.shape[2]

        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        with tf.variable_scope('', reuse=(not attack)):
            logits = self.get_logits(image)

        loss = ImageNetModel.compute_loss_and_error(logits, label, attack=attack)
        if not attack:
            wd = tf.contrib.layers.l2_regularizer(self.weight_decay)
            wd_loss = regularize_cost('.*/W', wd, name='l2_regularize_loss')

            add_moving_summary(loss, wd_loss)
            self.cost = tf.add_n([loss, wd_loss], name='cost')

        num_labels = len(LABEL_RANGES)
        zeroed = tf.sign(tf.maximum((tf.range(1000) - num_labels + 1), 0))
        neg_logits = tf.cast(zeroed, dtype=tf.float32) * tf.constant(-1000.0, dtype=tf.float32)
        return logits + neg_logits, loss

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of 224x224 in ``self.data_format``

        Returns:
            Nx1000 logits

        """

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    def image_preprocess(self, image, bgr=True, attack=False):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)

            if not attack:
                image = image * (1.0 / 255)

            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32)
            image_std = tf.constant(std, dtype=tf.float32)
            image = (image - image_mean) / image_std
            return image

    @staticmethod
    def compute_loss_and_error(logits, label, attack=False):
        all_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=label)
        if not attack:
            loss = tf.reduce_mean(all_loss, name='xentropy-loss')
            def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
                with tf.name_scope('prediction_incorrect'):
                    x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
                return tf.cast(x, tf.float32, name=name)

            wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
            add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

            wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
            add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
            return loss

        return all_loss
