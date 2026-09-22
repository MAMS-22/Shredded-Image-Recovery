import numpy as np
from numpy import array
import cv2
import os
import shutil
import random
from random import shuffle
from math import sqrt
import re

root_path = './'

if os.path.exists(root_path + 'COCO_dataset'):
    COCO_dataset_path = root_path + 'COCO_dataset/val2017/'
else:
    COCO_dataset_path = root_path + 'COCO dataset/val2017/'

cwd = root_path + 'project/'
dataset_path = cwd + 'image_dataset/'
train_path = dataset_path + 'train/'
test_path = dataset_path + 'test/'

initial_weights_path = root_path + '2024102912.weights.h5'
save_weights_path = root_path + 'new.weights.h5'


def get_fraction(DIR, fraction, index):
    files = os.listdir(DIR)
    files.sort()
    #files = np.array(files)
    #np.random.shuffle(files)
    files = files[ index*int(len(files)*fraction) : (index+1)*int(len(files)*fraction) ]
    return files


def get_n_files(DIR, number, start_index=0):
    files = os.listdir(DIR)
    files.sort()
    #files = np.array(files)
    #np.random.shuffle(files)
    files = files[start_index:start_index+number]
    return files


def resize_and_make_border(im, desired_size=100):
    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - im.shape[1]
    delta_h = desired_size - im.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return np.reshape(new_im, (desired_size,desired_size,3))


def extract_pos_from_filenames(files):
    return [int(re.search(r'_(\d+)\.jpg', filename).group(1)) for filename in files]

def get_row_image(images, row):  # for debugging
    row_image = np.concatenate([images[i] for i in row], axis=1)  # concatenate images in row to make a single image
    return row_image

def display_assembled_image(images, rows, sequence, display, save_path):
    img = np.concatenate([get_row_image(images, rows[row_ind]) for row_ind in sequence], axis=0)
    if display:
        cv2.imshow('img', img)
        cv2.waitKey(4000)
        cv2.destroyAllWindows()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_path, 'after.jpg'), img)
        shuffled_images = list(images)
        shuffle(shuffled_images)
        shuffled_img = np.concatenate([get_row_image(shuffled_images, rows[row_ind]) for row_ind in sequence], axis=0)
        cv2.imwrite(os.path.join(save_path, 'before.jpg'), shuffled_img)