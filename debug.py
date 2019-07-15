import os
import math
import cv2
import random
import numpy as np



def letterbox(img, height=416, mode='test', color=(127.5, 127.5, 127.5)):
    """
    resize a rectangular image to a padded square
    """
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    if mode == 'test':
        dw = (max(shape) - shape[1]) / 2  # width padding
        dh = (max(shape) - shape[0]) / 2  # height padding
        left, right = round(dw - 0.1), round(dw + 0.1)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
    else:
        dw = random.randint(0, max(shape) - shape[1])
        dh = random.randint(0, max(shape) - shape[0])
        left, right = dw, max(shape) - shape[1] - dw
        top, bottom = dh, max(shape) - shape[0] - dh

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square

    interp = np.random.randint(0, 5)
    img = cv2.resize(img, (height, height), interpolation=interp)  # resized, no border
   

    return img


img = cv2.imread("data/001.jpg")
letterbox(img)