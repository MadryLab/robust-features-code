#!/usr/bin/env python
# File: imagenet-resnet.py

import sys
import argparse
import numpy as np
import os

import tensorflow as tf

os.environ['TENSORPACK_TRAIN_API'] = 'v2'   # will become default soon
from tensorpack import logger, QueueInput
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.train import (
    TrainConfig, SyncMultiGPUTrainerParameterServer, launch_train_with_config)
from tensorpack.dataflow import imgaug, FakeData
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_nr_gpu

from imagenet_utils import (
    fbresnet_augmentor, get_imagenet_dataflow, ImageNetModel,
    eval_on_ILSVRC12)
from resnet_model import (
    preresnet_group, preresnet_basicblock, preresnet_bottleneck,
    resnet_group, resnet_basicblock, resnet_bottleneck, se_resnet_bottleneck,
    resnet_backbone)

TOTAL_BATCH_SIZE = 256

class Model(ImageNetModel):
    def __init__(self, target_shape, depth, data_format='NCHW', mode='resnet',
                 attack_inline=True):
        super(Model, self).__init__(target_shape, data_format=data_format,
                                    attack_inline=attack_inline)

        self.mode = mode
        basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'se': se_resnet_bottleneck}[mode]
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]

    def get_logits(self, image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            return resnet_backbone(
                image, self.num_blocks,
                preresnet_group if self.mode == 'preact' else resnet_group, self.block_func)


def get_data(name, batch, target_shape):
    isTrain = name == 'train'
    augmentors = fbresnet_augmentor(isTrain, target_shape)
    return get_imagenet_dataflow(
        args.data, name, batch, augmentors)


def get_config(model, checkpoint_dir, target_shape, fake=False):
    nr_tower = max(get_nr_gpu(), 1)
    batch = TOTAL_BATCH_SIZE // nr_tower

    if fake:
        logger.info("For benchmark, batch size is fixed to 64 per tower.")
        dataset_train = FakeData(
            [[64, target_shape, target_shape, 3], [64]], 1000, random=False, dtype='uint8')
        callbacks = []
    else:
        logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
        dataset_train = get_data('train', batch, target_shape)
        dataset_val = get_data('val', batch, target_shape)
        callbacks = [
            ModelSaver(checkpoint_dir=checkpoint_dir),
            ScheduledHyperParamSetter('learning_rate',
                                      [(30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5), (105, 1e-6)]),
            HumanHyperParamSetter('learning_rate'),
        ]
        infs = [ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]
        if nr_tower == 1:
            # single-GPU inference with queue prefetch
            callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
        else:
            # multi-GPU inference (with mandatory queue prefetch)
            callbacks.append(DataParallelInferenceRunner(
                dataset_val, infs, list(range(nr_tower))))
# 7.5 it / sec testing
    return TrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=100 if args.fake else 300, #5000 
        max_epoch=110,
        nr_tower=nr_tower
    )


if __name__ == '__main__':
    inet_dir = os.environ['INET_DIR']
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--eps', default=0.0, type=float)
    parser.add_argument('--data', help='ILSVRC dataset dir', default=inet_dir)
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
    parser.add_argument('--mode', choices=['resnet', 'preact', 'se'],
                        help='variants of resnet to use', default='resnet')
    parser.add_argument('--lp', choices=['2', 'inf'])
 
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.mode == 'se':
        assert args.depth >= 50

    nr_tower = max(get_nr_gpu(), 1)
    batch_size = TOTAL_BATCH_SIZE // nr_tower

    model = Model(args.image_size, args.depth, args.data_format, args.mode)
    if args.eval:
        batch = 128 # something that can run on one gpu
        ds = get_data('val', batch, args.image_size)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    else:
        logger.set_logger_dir(args.checkpoint_dir)
        config = get_config(model, args.checkpoint_dir, args.image_size, fake=args.fake)
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SyncMultiGPUTrainerParameterServer(max(get_nr_gpu(), 1))
        launch_train_with_config(config, trainer)
