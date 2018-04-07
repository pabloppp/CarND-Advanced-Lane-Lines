import numpy as np
import cv2


def minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def threshold(img, thresh=(0, 255)):
    mask = np.zeros_like(img)
    mask[(img > thresh[0]) & (img < thresh[1])] = 1
    return mask


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel = np.absolute(sobel)
    sobel = np.uint8(255 * minmax(sobel))
    mask = np.zeros_like(sobel)
    mask[(sobel > thresh[0]) & (sobel < thresh[1])] = 1
    return mask


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel_xy = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel = np.uint8(255 * minmax(sobel_xy))
    mask = np.zeros_like(sobel)
    mask[(sobel > thresh[0]) & (sobel < thresh[1])] = 1
    return mask


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    direction = np.arctan2(abs_sobel_y, abs_sobel_x)
    mask = np.zeros_like(direction)
    mask[(direction > thresh[0]) & (direction < thresh[1])] = 1
    return mask
