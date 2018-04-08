import cv2
import numpy as np

from src.utils.calibration import setup_undistort
from src.utils.gradients import abs_sobel_thresh, threshold, dir_threshold, mag_thresh

undistort = setup_undistort("calibration_matrix.p")


def birds_eye_view(image, debug=False):
    src = np.float32([
        [215, 706],
        [580, 460],
        [704, 460],
        [1094, 706],
    ])

    dst = np.float32([
        [320, 720],
        [320, 0],
        [960, 0],
        [960, 720],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = warp(image, M)
    unwarped = warp(warped, Minv)

    debug_pre = None
    debug_post = None
    if debug:
        vrx = (np.array(src, np.int32)).reshape((-1, 1, 2))
        image_bgr = cv2.cvtColor(image * 255, cv2.COLOR_GRAY2BGR)
        debug_pre = cv2.polylines(image_bgr, [vrx], True, (255, 0, 0), 3)
        vrx = (np.array(dst, np.int32)).reshape((-1, 1, 2))
        warped_bgr = cv2.cvtColor(warped * 255, cv2.COLOR_GRAY2BGR)
        debug_post = cv2.polylines(warped_bgr, [vrx], True, (255, 0, 0), 3)

    return warped, unwarped, M, Minv, debug_pre, debug_post


def warp(image, M):
    return cv2.warpPerspective(image, M, image.shape[1::-1])


def pipeline(image, debug=False):
    image_undistorted = undistort(image)
    thresholded = thresholded_binary_image(image_undistorted)
    birdseye = birds_eye_view(thresholded, debug)
    return birdseye


def thresholded_binary_image(image):
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    image_lab_l = image_lab[:, :, 0]
    image_hls_h = image_hls[:, :, 0]
    image_hls_l = image_hls[:, :, 1]
    image_hls_s = image_hls[:, :, 2]

    image_lab_l_sobel_x = abs_sobel_thresh(image_lab_l, orient='x', sobel_kernel=15, thresh=(20, 60))

    image_lab_l_sobel_y = abs_sobel_thresh(image_lab_l, orient='y', sobel_kernel=15, thresh=(20, 120))
    image_lab_l_sobel_mag = mag_thresh(image_lab_l, sobel_kernel=15, thresh=(110, 130))
    image_lab_l_sobel_dir = dir_threshold(image_lab_l, sobel_kernel=15, thresh=(np.pi / 4, np.pi / 2))

    image_hls_h_white = threshold(image_hls_h, thresh=(0, 255))
    image_hls_l_white = threshold(image_hls_l, thresh=(200, 255))
    image_hls_s_white = threshold(image_hls_s, thresh=(0, 255))

    image_hls_h_yellow = threshold(image_hls_h, thresh=(15, 35))
    image_hls_l_yellow = threshold(image_hls_l, thresh=(30, 205))
    image_hls_s_yellow = threshold(image_hls_s, thresh=(85, 255))

    color_binary = np.zeros_like(image_lab_l_sobel_x)
    color_binary[
        (image_lab_l_sobel_x == 1) |
        ((image_lab_l_sobel_y == 1) & (image_lab_l_sobel_mag == 1) & (image_lab_l_sobel_dir == 1)) |
        ((image_hls_h_white == 1) & (image_hls_l_white == 1) & (image_hls_s_white == 1)) |
        ((image_hls_h_yellow == 1) & (image_hls_l_yellow == 1) & (image_hls_s_yellow == 1))
        ] = 1

    return color_binary
