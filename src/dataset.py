#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Dataset and transformations
'''

import cv2
import os
import numpy as np
from tensorflow.keras.utils import Sequence


class DataLoader(Sequence):
    """Implements a data loader class

    Args:
        root (str): root image folder
        samples (list<tuple>): list of image path and label
        batch_size (int, optional): batch size
        spatial_shape (tuple<int>, optional): spatial size of the image
        n_channels (int, optional): number of color channels
        shuffle (bool, optional): should the extract be shuffled
        transform_fn (callable, optional): transformation to apply to each sample

    """
    def __init__(self, root, samples, batch_size=32, vrange=slice(70, -20), spatial_shape=(64, 64), n_channels=3,
                 shuffle=True, transform_fn=None):
        self.root = root
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.data = samples
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.vrange = vrange
        self.transform_fn = transform_fn
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        _samples = [self.data[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(_samples)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_samples):
        """Generate data containing self.batch_size samples"""
        # Initialization
        X = np.empty((self.batch_size, *self.spatial_shape, self.n_channels), dtype=np.uint8)
        y = np.empty((self.batch_size), dtype=np.float32)

        # Generate data
        for idx, (_imgpath, _steering) in enumerate(batch_samples):
            # Store sample (BGR)
            X[idx, :] = cv2.resize(cv2.imread(os.path.join(self.root, _imgpath))[self.vrange, :, ::-1],
                                   self.spatial_shape[::-1])

            # Store class
            y[idx] = _steering

        if isinstance(self.transform_fn, list):
            for t in self.transform_fn:
                X, y = t(X, y)
        elif callable(self.transform_fn):
            X, y = self.transform_fn(X, y)

        return X, y


def h_flip(X, y, prob=0.5):
    """Horizontal flip transformation

    Args:
        X (numpy.ndarray[N, H, W, C]): input samples
        y (numpy.ndarray[N]): label samples
        prob (float, optional): probability of data augmentation for each sample
    """

    # Augmentations
    selection = np.random.rand(X.shape[0]) >= prob
    # Horizontal flip
    X[selection, ...] = X[selection, :, ::-1, :]
    y[selection] = -y[selection]

    return X, y


def shadow(X, y):
    """Overlay randomly positioned shadow

    Args:
        X (numpy.ndarray[N, H, W, C]): input samples
        y (numpy.ndarray[N]): label samples
        prob (float, optional): probability of data augmentation for each sample
    """

    if X.ndim != 4:
        raise AssertionError("expected `X` to be 4-dimensional")

    for _idx in range(X.shape[0]):
        X[_idx, ...] = _shadow(X[_idx, ...])

    return X, y


def _shadow(X):
    """Overlay randomly positioned shadow

    Args:
        X (numpy.ndarray[N, H, W, C]): input samples
        y (numpy.ndarray[N]): label samples
        prob (float, optional): probability of data augmentation for each sample
    """

    if X.ndim != 3:
        raise AssertionError("expected `X` to be 4-dimensional")

    intercepts = np.random.choice(X.shape[1], 2, replace=False)
    slopes = X.shape[1] / (intercepts[1] - intercepts[0])
    biases = - slopes * intercepts[0]

    for y in range(X.shape[0]):
        shadow_edge = ((y - biases) / slopes).astype(int)
        X[y, :shadow_edge, :] //= 2

    return X
