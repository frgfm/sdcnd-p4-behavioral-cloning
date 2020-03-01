#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Behavioral cloning training
'''

import argparse
import os
from shutil import rmtree
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from dataset import DataLoader, h_flip, shadow
import models


def main(args):

    data_folder = Path(args.folder)
    if not data_folder.is_dir():
        raise FileNotFoundError(f"unable to access {data_folder}")

    # Read the driving log
    samples = []
    with open(data_folder.joinpath('driving_log.csv'), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip the headers
        for sample in reader:
            steering = float(sample[3].strip())
            if args.resampling is None or (steering != 0) or (random.random() > args.resampling):
                for _idx, _imgpath in enumerate(sample[:3]):
                    # Correct left and right cameras
                    samples.append((_imgpath.strip().rpartition('/')[-1], steering - (_idx - 1) * args.correction))

    # Split the samples in train and validation sets
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # Train and valid the model using the generator function
    spatial_shape = (args.size, args.size)
    train_loader = DataLoader(data_folder.joinpath('IMG'), train_samples, spatial_shape=spatial_shape,
                              batch_size=args.batch_size, transform_fn=[h_flip, shadow])
    validation_loader = DataLoader(data_folder.joinpath('IMG'), validation_samples,
                                   spatial_shape=spatial_shape, batch_size=args.batch_size)

    model = models.__dict__[args.arch](input_shape=(*spatial_shape, 3))
    print(model.summary())

    # Tensorboard
    log_dir = Path('logs')
    if not log_dir.is_dir():
        log_dir.mkdir(parents=True)

    log_file = f"{args.arch}_{datetime.datetime.now().strftime('%Y%m%d')}"
    if log_dir.joinpath(log_file).is_dir():
        rmtree(log_dir.joinpath(log_file))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir.joinpath(log_file), histogram_freq=1)

    # Scheduler
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3,
                                                       min_lr=args.lr * 1e-3, min_delta=1e-3, verbose=1)

    # Compile and train the model
    optimizer = Adam(lr=args.lr)
    model.compile(loss='mse', optimizer=optimizer)
    model.fit(train_loader, epochs=args.epochs, validation_data=validation_loader, workers=4,
              callbacks=[tensorboard_callback, lr_callback])

    # Save the model
    model.save(args.output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Driving behavior cloning training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--arch", type=str, default='babypilot', help="Model architecture")
    parser.add_argument("--folder", type=str, default='./data', help="Path to data folder")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--size", type=int, default=64, help="Image resizing")
    parser.add_argument("--output", type=str, default='model.h5', help="Path where trained model will be saved")
    parser.add_argument("--correction", type=float, default=0.25, help="Default steering correction")
    parser.add_argument("--dropout", type=float, default=0., help="Dropout probability")
    parser.add_argument("--resampling", type=float, default=0.5,
                        help="Probability of keeping samples equals to mode")

    args = parser.parse_args()
    main(args)
