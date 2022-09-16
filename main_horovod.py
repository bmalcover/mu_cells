import os
import json
import colorsys
import random
from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
import sys

import cv2 
import skimage
import skimage.io
import skimage.color
import skimage.transform
import numpy as np
import pandas as pd
import imgaug as ia
import imgaug.augmenters as iaa
from numpy.random import seed
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from matplotlib import patches,  lines

from tensorflow.keras import backend as K
import tensorflow.keras.layers as keras_layer
import tensorflow as tf
from tensorflow.keras import utils as KU
import horovod.tensorflow.keras as hvd 

# Llibraries pròpies
from u_rpn.data import unet as u_data
from u_rpn.data import rpn as rpn_data
from u_rpn.data import datasets as rpn_datasets
from u_rpn import model as u_model
from u_rpn import configurations as u_configs
from u_rpn.common import data as common_data
from u_rpn import layers as own_layers
from u_rpn.losses import bce
from u_rpn import losses as rpn_losses
from u_rpn.common import utils as rpn_utils
from u_rpn.common import metrics as rpn_metrics


seed(1)


hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# ============================================
# Optimisation Flags - Do not remove
# ============================================

os.environ['CUDA_CACHE_DISABLE'] = '0'

os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'

os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

os.environ['TF_ADJUST_HUE_FUSED'] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

os.environ['TF_SYNC_ON_FINISH'] = '0'
os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
os.environ['TF_DISABLE_NVTX_RANGES'] = '1'


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def draw_bboxes(img, bboxes, thickness=3):
    img = np.copy(img.astype(np.uint8))
    colors = random_colors(len(bboxes))

    for bbox, color in zip(bboxes, colors):
        color = np.array(color) * 255
        img = cv2.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, thickness)
    
    return img

def make_masks(mask, slice_mask):
    if isinstance(slice_mask, list):
        if len(slice_mask) != 2:
            raise ValueError
        
        mask = np.sum(mask[:,:, slice_mask[0]:slice_mask[1]], axis=-1)
    elif isinstance(slice_mask, list):
        mask = mask[:, :, slice_mask]
    else:
        raise ValueError
    
    return mask


MULTI_CLASS = False
PYRAMID = False
TRANSFER = False

config = u_configs.CellConfig()

if MULTI_CLASS:
    NUM_CLASSES = 1 + 3  # Background + 3 classes
else:
    NUM_CLASSES = 1 + 1  # Background + 3 classes
        
config.IMAGE_SHAPE = np.array([128,128, 3])
config.BATCH_SIZE = 2
config.DO_MERGE_BRANCH = False
config.DO_MASK_CLASS = False


# Validation dataset
dataset_val = rpn_datasets.ErithocytesDataset([("cell", 1, "cell")], "bboxes.json")
dataset_val.load_cell("./in/eritocitos_augmented/", rpn_datasets.Subset.VALIDATION)
dataset_val.prepare()

val_generator = rpn_data.DataGenerator(2, dataset_val, config, shuffle=False)
anchors = val_generator.anchors.tolist()

dataset = rpn_datasets.ErithocytesPreDataset(
    "./in/jumbo2_mini/train", "data.json", divisor=1
)
dataset.prepare()

generator = rpn_data.DataGenerator(
    50,
    dataset,
    pre_calculated=True,
    config=config,
    phantom_output=True,
    shuffle=False,
    size_anchors=dataset.anchors,
)


urpn = u_model.u_rpn.URPN(
        config=config,
        mode=u_model.keras_rpn.NeuralMode.TRAIN,
        anchors=anchors,
        input_size=(128, 128, 3),
        decoder_output=u_model.decoder.SuperDecoderOutput,
        input_sd=(128, 128),
        cell_shape=(9, 9),
        filters=32,
        decoder_output_size=3
    )

opt = tf.keras.optimizers.Adadelta(config.LEARNING_RATE * hvd.size())

urpn.compile(optimizer=hvd.DistributedOptimizer(opt),
             run_eagerly=True)

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./check/checkpoint.h5', save_best_only=True))


urpn.internal_model.fit(
    generator,
    epochs=1000,
    steps_per_epoch=config.STEPS_PER_EPOCH,
    callbacks=callbacks
)

