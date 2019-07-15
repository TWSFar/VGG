import cv2
import math
import random
import numpy as np


def normalize(img, mean=[0.485, 0.456, 0.406], std=[0.299, 0.224, 0.225]):
    ''' 
    norm = (x - mean) / std
    '''
    img = img / 255.0
    mean = np.array(mean)
    std = np.array(std)
    img = (img - mean[:, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis]
    return img.astype(np.float32)


def letterbox(img, img_size=224, mode="train", color=(127.5, 127.5, 127.5)):
    '''  
    resize a rectangular image to a padded square
    '''
    shape = img.shape[:2]
    if mode == 'train':
        dw = random.randint(0, max(shape) - shape[1])
        dh = random.randint(0, max(shape) - shape[0])
        top, bottom = dh, max(shape) - shape[0] - dh
        left, right = dw, max(shape) - shape[1] - dw
    else:
        dw = (max(shape) - shape[1]) / 2
        dh = (max(shape) - shape[0]) / 2
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    interp = np.random.randint(0, 5)
    img = cv2.resize(img, (img_size, img_size), interpolation=interp)

    return img

'''
img = cv2.imread("data/001.jpg")
letterbox(img)
'''