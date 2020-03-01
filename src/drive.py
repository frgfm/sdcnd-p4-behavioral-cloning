#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Autonomous driving simulation
'''

import argparse
import base64
from datetime import datetime
import os
from pathlib import Path

import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import models


def preprocess(img, vrange=slice(70, -20), spatial_shape=(64, 64)):
    """Preprocess image arrays for keras model

    Args:
        img (numpy.ndarray[H, W, 3]): input image
        vrange (slide): vertical range to keep (cropping)
        spatial_shape (tuple<int>): target shape

    Returns:
        numpy.ndarray[*spatial_shape, 3]: processed image
    """
    return cv2.resize(img[vrange, ...], spatial_shape[::-1])


class SimplePIController:
    """Implements a PI Controller

    Args:
        Kp (float): error coefficient
        Ki (float): integral coefficient
    """
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


def main(args):

    # Model loading
    spatial_shape = (args.size, args.size)
    model = models.__dict__[args.arch](input_shape=(*spatial_shape, 3))
    model.load_weights(args.model)

    if args.img_folder:
        output_folder = Path(args.img_folder)
        if not output_folder.is_dir():
            output_folder.mkdir(parents=True)
        print(f"Driving session recorded at: {output_folder}")
    else:
        print(f"Unrecorded session")

    # Controller
    controller = SimplePIController(0.1, 0.002)
    controller.set_desired(args.speed)

    # Web Server
    sio = socketio.Server()

    def send_control(steering_angle, throttle):
        sio.emit(
            "steer",
            data={
                'steering_angle': steering_angle.__str__(),
                'throttle': throttle.__str__()
            },
            skip_sid=True)

    @sio.on('telemetry')
    def telemetry(sid, data):
        if data:
            # The current speed of the car
            speed = data["speed"]
            # The current image from the center camera of the car
            image = Image.open(BytesIO(base64.b64decode(data["image"])))
            image_array = preprocess(np.asarray(image), spatial_shape=spatial_shape)
            # Add batch size and forward
            steering_angle = float(model.predict(image_array[None, ...], batch_size=1))

            # Adjust speed to match desired speed
            throttle = controller.update(float(speed))

            print(steering_angle, throttle)
            send_control(steering_angle, throttle)

            # save frame
            if args.img_folder:
                timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                image.save(output_folder.joinpath(f"{timestamp}.jpg"))
        else:
            # NOTE: DON'T EDIT THIS.
            sio.emit('manual', data={}, skip_sid=True)

    @sio.on('connect')
    def connect(sid, environ):
        print('====================================')
        print(f"\tAutopilot engaged\n(SID: {sid})")
        print('====================================')
        send_control(0, 0)

    @sio.on('disconnect')
    def disconnect(sid):
        print('====================================')
        print(f"\tAutopilot disengaged\n(SID {sid})")
        print('====================================')

    # wrap Flask application with engineio's middleware
    app = Flask(__name__)
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autonomous driving simulation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', type=str, help='Path to model h5 file. Model should be on the same path.')
    parser.add_argument('--arch', type=str, default='babypilot', help='Architecture used for simulation.')
    parser.add_argument("--size", type=int, default=64, help="Image resizing")
    parser.add_argument('--speed', type=int, default=30, help='Max allowed speed.')
    parser.add_argument('--img-folder', type=str, nargs='?', default=None,
                        help='Path to image folder. This is where the images from the run will be saved.')
    args = parser.parse_args()
    main(args)
