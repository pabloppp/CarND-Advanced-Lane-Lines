import cv2
import numpy as np

from src.utils.calibration import setup_undistort
from src.utils.gradients import abs_sobel_thresh, threshold, dir_threshold

undistort = setup_undistort("calibration_matrix.p")


def birds_eye_view(image, debug=False):
    src = np.float32([
        [215, 706],
        [605, 444],
        [676, 444],
        [1093, 706],
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
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    image_y = image_yuv[:, :, 0]
    image_u = image_yuv[:, :, 1]
    image_v = image_yuv[:, :, 2]
    image_h = image_hls[:, :, 0]
    image_l = image_hls[:, :, 1]
    image_s = image_hls[:, :, 2]

    image_y_threshold = threshold(image_y, thresh=(205, 255))
    image_y_sobel_threshold = abs_sobel_thresh(image_y, orient="x", thresh=(110, 255))

    image_u_threshold = threshold(image_u, thresh=(50, 115))

    image_v_threshold = threshold(image_v, thresh=(150, 255))
    image_v_low_threshold = threshold(image_v, thresh=(0, 125))

    # image_h_low_threshold = threshold(image_h, thresh=(0, 160))

    image_l_threshold = threshold(image_l, thresh=(200, 255))

    image_s_threshold = threshold(image_s, thresh=(140, 255))

    color_binary = np.zeros_like(image_s_threshold)
    color_binary[
        ((image_s_threshold >= 1) | (image_y_threshold >= 1) | (image_y_sobel_threshold >= 1) |
         (image_u_threshold >= 1) | (image_v_threshold >= 1) | (image_l_threshold >= 1)) &
        (image_v_low_threshold < 1)  # & (image_h_low_threshold >= 1)
        ] = 1

    return color_binary
