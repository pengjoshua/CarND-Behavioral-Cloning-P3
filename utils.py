import json
import os
import cv2
import errno
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli
from scipy.misc import imread, imresize


def random_shear(image, steering, shear_range=200):
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    points1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    points2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(points1, points2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering += dsteering
    return image, steering

def crop(image, top_percent, bot_percent):
    top = int(np.ceil(image.shape[0] * top_percent))
    bot = image.shape[0] - int(np.ceil(image.shape[0] * bot_percent))
    return image[top:bot, :]

def random_flip(image, steering):
    coin = bernoulli.rvs(0.5)
    if coin:
        return np.fliplr(image), -1 * steering
    else:
        return image, steering

def random_gamma(image):
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)

def random_brightness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4 * (2 * np.random.uniform() - 1.0)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

def resize(image, shape=(64, 64)):
    return imresize(image, shape)

def modify_image(image, steering, crop_top_percent=0.35, crop_bot_percent=0.1, shear_prob=0.9):
    coin = bernoulli.rvs(shear_prob)
    if coin == 1:
        image, steering = random_shear(image, steering)
    image = crop(image, crop_top_percent, crop_bot_percent)
    image, steering = random_flip(image, steering)
    image = random_brightness(image)
    image = resize(image)
    return image, steering

def batch_random_view(batch_size=64, correction=0.225):
    data = pd.read_csv('./data/driving_log.csv')
    num_images = len(data)
    random_indices = np.random.randint(0, num_images, batch_size)

    batch = []
    for index in random_indices:
        view = np.random.randint(0, 3)
        if view == 0:
            img = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + correction
            batch.append((img, angle))
        elif view == 1:
            img = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            batch.append((img, angle))
        else:
            img = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - correction
            batch.append((img, angle))
    return batch

def generate_batch(batch_size=64):
    while 1:
        X_batch = []
        y_batch = []
        images = batch_random_view(batch_size)
        for image_file, angle in images:
            raw_image = plt.imread('./data/' + image_file)
            raw_angle = angle
            new_image, new_angle = modify_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)
        yield np.array(X_batch), np.array(y_batch)
