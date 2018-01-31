
'''
Calculates stats on dataset, performs image augmentation,
and generates images/steerings to train, validate and test model
Original By: dolaameng Revd By: cgundling
*Note that steering shifts for left/right cameras is currently
commented out. Change line 323 to add back in.
'''

from __future__ import print_function

import numpy as np
import pandas as pd
import csv
import random

from collections import defaultdict
from os import path

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from skimage.transform import resize, rotate
from skimage.io import imread

'''
Generator assumes the dataset folder has the following structure:
camera.csv  center/  left/  right/  steering.csv
Bag files processed by:
[rwightman/udacity-driving-reader](https://github.com/rwightman/udacity-driving-reader)
'''

# The following are all the functions used for data processing/augmentation
# ----------------------------------------------------------------------------------

def read_steerings(steering_log, time_scale):
    steerings = defaultdict(list)
    speeds = defaultdict(list)
    with open(steering_log) as f:
        for line in f.readlines()[1:]:
            fields = line.split(",")
            nanosecond, angle, speed = int(fields[0]), float(fields[1]), float(fields[3])
            timestamp = nanosecond / time_scale
            steerings[timestamp].append(angle)
            speeds[timestamp].append(speed)
    return steerings, speeds


def camera_adjust(angle,speed,camera):

    # Left camera -20 inches, right camera +20 inches (x-direction)
    # Steering should be correction + current steering for center camera

    # Chose a constant speed
    speed = 10.0  # Speed

    # Reaction time - Time to return to center
    # The literature seems to prefer 2.0s (probably really depends on speed)
    if speed < 1.0:
        reaction_time = 0
        angle = angle
    else:
        reaction_time = 2.0 # Seconds

        # Trig to find angle to steer to get to center of lane in 2s
        opposite = 20.0 # inches
        adjacent = speed*reaction_time*12.0 # inches (ft/s)*s*(12 in/ft) = inches (y-direction)
        angle_adj = np.arctan(float(opposite)/adjacent) # radians

        # Adjust based on camera being used and steering angle for center camera
        if camera == 'left':
            angle_adj = -angle_adj
        angle = angle_adj + angle

    return angle


def read_image_stamps(image_log, camera, time_scale):
    timestamps = defaultdict(list)
    with open(image_log) as f:
        for line in f.readlines()[1:]:
            if camera not in line:
                continue
            fields = line.split(",")
            nanosecond = int(fields[0])
            timestamp = nanosecond / time_scale
            timestamps[timestamp].append(nanosecond)
    return timestamps

# read a batch of images, id is file name
def read_images(image_folder, camera, ids, image_size):
    prefix = path.join(image_folder, camera)
    imgs = []
    for id in ids:
        # Uncomment to view cropped images and sizes
        img = imread(path.join(prefix, "{}.jpg".format(id))) # (480,640,3) np array

        # Cropping and resizing
        crop_img = img[200:,:]
        img = resize(crop_img, image_size)

        imgs.append(img)
    if len(imgs) < 1:
        print('Error no image at timestamp')
        print(ids)
    img_block = np.stack(imgs, axis=0)
    if K.image_dim_ordering() == 'th':   # tensorflow = tf, theono = th
        img_block = np.transpose(img_block, axes = (0, 3, 1, 2))
    return img_block


def read_images_augment(image_folder, camera, ids, image_size):
    prefix = path.join(image_folder, camera)
    imgs = []
    j = 0
    for id in ids:
        # Uncomment to view cropped images and sizes
        img = imread(path.join(prefix, "{}.jpg".format(id))) # (480,640,3) np array

        # Flip image
        img = np.fliplr(img)  # flip left right

        # Cropping and resizing
        crop_img = img[200:,:]
        img = resize(crop_img, image_size)

        # Rotate randomly by small amount (not a viewpoint transform)
        img = rotate(img, random.uniform(-1, 1))

        imgs.append(img)
    if len(imgs) < 1:
        print('Error no image at timestamp')
        print(ids)
    img_block = np.stack(imgs, axis=0)
    if K.image_dim_ordering() == 'th':
        img_block = np.transpose(img_block, axes = (0, 3, 1, 2))
    return img_block


def normalize_input(x):
    return x / 255.


def exact_output(y):
    return y


def preprocess_input_InceptionV3(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


# Data generator (output mean steering angles and image arrays)
# ---------------------------------------------------------------------------------
def data_generator(steering_log, image_log, image_folder, gen_type='train',
                   camera='center', batch_size=32, fps=10, image_size=(128,128),
                   timestamp_start=None, timestamp_end=None, shuffle=True,
                   preprocess_input=normalize_input,
                   preprocess_output=exact_output):

    # Constants
    # -----------------------------------------------------------------------------
    minmax = lambda xs: (min(xs), max(xs))
    time_scale = int(1e9) / fps

    # Read the image stamps for each camera
    # -----------------------------------------------------------------------------
    if gen_type == 'train':
        image_stamps = read_image_stamps(image_log, camera[0], time_scale)
        image_stamps_r = read_image_stamps(image_log, camera[1], time_scale)
        image_stamps_l = read_image_stamps(image_log, camera[2], time_scale)
    else:
        image_stamps = read_image_stamps(image_log, camera, time_scale)

    # Read all steering angles and speeds
    # -----------------------------------------------------------------------------
    steerings, speeds = read_steerings(steering_log, time_scale)


    # More data exploration stats
    # -----------------------------------------------------------------------------
    print('timestamp range for all steerings: %d, %d' % minmax(steerings.keys()))
    print('timestamp range for all images: %d, %d' % minmax(image_stamps.keys()))
    print('min and max # of steerings per time unit: %d, %d' % minmax(map(len, steerings.values())))
    print('min and max # of images per time unit: %d, %d' % minmax(map(len, image_stamps.values())))

    # Generate images and steerings within one time unit
    # (Mean steering angles used when mulitple steering angels within a single time unit)
    # -----------------------------------------------------------------------------
    start = max(min(steerings.keys()), min(image_stamps.keys()))
    if timestamp_start:
        start = max(start, timestamp_start)
    end = min(max(steerings.keys()), max(image_stamps.keys()))
    if timestamp_end:
        end = min(end, timestamp_end)
    print("sampling data from timestamp %d to %d" % (start, end))

    # unique timestamps between start and end
    time_list = image_stamps.keys()
    unique_list = [t for t in time_list if t <= end and t>=start]
    unique_set = set(unique_list)

    # While loop for data generator
    # -----------------------------------------------------------------------------
    i = start - 1
    x_buffer, y_buffer, buffer_size = [], [], 0
    if gen_type =='train':
        camera_select = camera[0]
    else:
        camera_select = 'center'

    while True:
        # update i for next iteration
        if shuffle:
            i = int(random.choice(unique_list)) # unique_timestamp_list
            # this i must be between start and end
        else:
            i += 1
            while i not in unique_set:
                i += 1
                if i > end:
                    i = start

        coin = random.choice([1,2])
        if steerings[i] and image_stamps[i]:
            if camera_select == 'right':
                images = read_images(image_folder, camera_select, image_stamps_r[i], image_size)
            elif camera_select == 'left':
                images = read_images(image_folder, camera_select, image_stamps_l[i], image_size)
            elif camera_select == 'center':
                if gen_type == 'train':
                    if coin == 1:
                        images = read_images(image_folder, camera_select, image_stamps[i], image_size)
                    else:
                        images = read_images_augment(image_folder, camera_select, image_stamps[i], image_size)
                else:
                    images = read_images(image_folder, camera_select, image_stamps[i], image_size)

            # Mean angle with a timestamp
            angle = np.repeat([np.mean(steerings[i])], images.shape[0]) # It's likely that number the angles and images are not the same, use the average angle for all images in that timestamp
            # Adjust steering angle for horizontal flipping
            if gen_type == 'train' and coin == 2:
                angle = -angle
            # speed = np.repeat([np.mean(speeds[i])], images.shape[0])
            # Adjust the steerings of the offcenter cameras
            # if camera_select != 'center':
            #     angle = camera_adjust(angle[0], speed[0], camera_select)
            #     angle = np.repeat([angle], images.shape[0])
            x_buffer.append(images)
            y_buffer.append(angle)
            buffer_size += images.shape[0]
            if buffer_size >= batch_size:
                indx = range(buffer_size)
                if gen_type == 'train':
                    np.random.shuffle(indx)
                x = np.concatenate(x_buffer, axis=0)[indx[:batch_size], ...] # np.concatenate [[2,2],[3,3],[4,4]] => [2,2,3,3,4,4]
                y = np.concatenate(y_buffer, axis=0)[indx[:batch_size], ...]
                x_buffer, y_buffer, buffer_size = [], [], 0
                yield preprocess_input(x.astype(np.float32)), preprocess_output(y)
