#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Behavioral cloning model
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda


def normalize_tensor(X):
    """Normalize uint8 encoded tensors

    Args:
        X (numpy.ndarray): image tensor encoded with uint8

    Returns:
        numpy.ndarray: normalized float tensor
    """

    X /= 255.0
    X -= 0.5

    return X


def lenet5(dropout_prob=0.2, input_shape=(64, 64, 3)):
    """Implements a LeNet5 architecture from http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    with dropout

    Args:
        dropout_prob (float): dropout probability
        input_shape (tuple<int>): input tensor shape

    Returns:
        tensorflow.keras.models.Sequential: LeNet5 model
    """

    return Sequential([
        Lambda(normalize_tensor, input_shape=input_shape),
        Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid',
               activation='relu', data_format='channels_last', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(16, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(dropout_prob),
        Dense(120, activation='relu'),
        Dropout(dropout_prob),
        Dense(84, activation='relu'),
        Dense(1)
    ])


def nvidia(dropout_prob=0.2, input_shape=(160, 320, 3)):
    """Implements a motion control architecture as described in https://arxiv.org/abs/1604.07316

    Args:
        dropout_prob (float): dropout probability
        input_shape (tuple<int>): input tensor shape

    Returns:
        tensorflow.keras.models.Sequential: LeNet5 model
    """

    return Sequential([
        Lambda(normalize_tensor, input_shape=input_shape),
        Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='valid',
               activation='relu', data_format='channels_last', input_shape=input_shape),
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dropout(dropout_prob),
        Dense(100, activation='relu'),
        Dropout(dropout_prob),
        Dense(50, activation='relu'),
        Dropout(dropout_prob / 2),
        Dense(10, activation='relu'),
        Dense(1)
    ])


def babypilot(dropout_prob=0.5, input_shape=(64, 64, 3)):
    """Custom motion control architecture with dropout

    Args:
        dropout_prob (float): dropout probability
        input_shape (tuple<int>): input tensor shape

    Returns:
        tensorflow.keras.models.Sequential: LeNet5 model
    """

    return Sequential([
        Lambda(normalize_tensor, input_shape=input_shape),
        Conv2D(filters=16, kernel_size=(3, 3), padding='valid', kernel_initializer='he_normal',
               activation='relu', data_format='channels_last', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), kernel_initializer='he_normal', activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), kernel_initializer='he_normal', activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(dropout_prob),
        Dense(128, activation='relu', kernel_initializer='he_uniform'),
        Dropout(dropout_prob),
        Dense(64, activation='relu', kernel_initializer='he_uniform'),
        Dropout(dropout_prob / 2),
        Dense(16, activation='relu', kernel_initializer='he_uniform'),
        Dense(1, kernel_initializer='he_uniform')
    ])
