import numpy as np
import cv2


def _window_mask(width, height, img, center, level):
    output = np.zeros_like(img)
    output[int(img.shape[0] - (level + 1) * height):int(img.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img.shape[1])] = 1
    return output


def _find_window_centroids(image, window_width, window_height, margin, vertical_threshold):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

    # Add what we found for the first layer
    # window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(0, int(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 4
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_value = np.argmax(conv_signal[l_min_index:l_max_index])
        if l_value >= vertical_threshold:
            l_center = l_value + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_value = np.argmax(conv_signal[r_min_index:r_max_index])
        if r_value >= vertical_threshold:
            r_center = r_value + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center, l_value, r_value, conv_signal))

    return window_centroids


def _plot_masks(l_center, r_center):
    leftx = [i[0] for i in l_center]
    lefty = [i[1] for i in l_center]
    rightx = [i[0] for i in r_center]
    righty = [i[1] for i in r_center]

    left_fit = np.polyfit(leftx, lefty, 2)
    right_fit = np.polyfit(rightx, righty, 2)

    left_curve = np.poly1d(left_fit)
    right_curve = np.poly1d(right_fit)

    return left_fit, right_fit, left_curve, right_curve


def _lane_shape(image_width, left_curve, right_curve):
    left_fitx = np.empty([image_width])
    right_fitx = np.empty([image_width])
    ploty = np.empty([image_width])
    for i in range(0, image_width):
        ploty[i] = i
        left_fitx[i] = left_curve(i)
        right_fitx[i] = right_curve(i)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    return np.hstack((pts_left, pts_right))


def _curve_radius(y_eval, l_center, r_center):
    leftx = [i[0] for i in l_center]
    lefty = [i[1] for i in l_center]
    rightx = [i[0] for i in r_center]
    righty = [i[1] for i in r_center]

    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 640

    left_fit_cr = np.polyfit(np.multiply(leftx, ym_per_pix), np.multiply(lefty, xm_per_pix), 2)
    right_fit_cr = np.polyfit(np.multiply(rightx, ym_per_pix), np.multiply(righty, xm_per_pix), 2)

    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])

    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad


def _offcenter_position(y_eval, image_width, left_curve, right_curve):
    xm_per_pix = 3.7 / 640
    return ((left_curve(y_eval) + (right_curve(y_eval) - left_curve(y_eval)) / 2) - (image_width / 2)) * xm_per_pix


def detect_lanes(image, window_dims=(40, 45), margin=180, shape_only=False):
    vertical_threshold = 2500 / window_dims[1]

    # image = image[:, :, 0]
    window_centroids = _find_window_centroids(image, window_dims[0], window_dims[1], margin, vertical_threshold)

    if len(window_centroids) > 0:
        l_points = np.zeros_like(image)
        r_points = np.zeros_like(image)

        l_center = []
        r_center = []
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            if window_centroids[level][2] >= vertical_threshold:
                l_mask = _window_mask(window_dims[0], window_dims[1], image, window_centroids[level][0], level)
                l_points[(l_points == 255) | (l_mask == 1)] = 255
                l_center.append((image.shape[0] - ((level + 0.5) * window_dims[1]), window_centroids[level][0]))

            if window_centroids[level][3] >= vertical_threshold:
                r_mask = _window_mask(window_dims[0], window_dims[1], image, window_centroids[level][1], level)
                r_points[(r_points == 255) | (r_mask == 1)] = 255
                r_center.append((image.shape[0] - ((level + 0.5) * window_dims[1]), window_centroids[level][1]))

        left_fit, right_fit, left_curve, right_curve = _plot_masks(np.array(l_center), np.array(r_center))
        for y in range(0, image.shape[0]):
            l_points[y, int(left_curve(y)) - 5:int(left_curve(y)) + 5] = 255
            r_points[y, int(right_curve(y)) - 5:int(right_curve(y)) + 5] = 255

        zero_channel = np.zeros_like(image)  # create a zero color channel
        template = np.array(cv2.merge((r_points, l_points, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.dstack((image, image, image)) * 255  # making the original road pixels 3 color channels

        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

        if shape_only:
            output = np.dstack((zero_channel, zero_channel, zero_channel))
            pts = _lane_shape(image.shape[0], left_curve, right_curve)
            cv2.fillPoly(output, np.int32([pts]), (0, 255, 0))

        curvatures = _curve_radius(image.shape[0] - 1, np.array(l_center, dtype=np.float32),
                                   np.array(r_center, dtype=np.float32))
        offcenter = _offcenter_position(image.shape[0] - 1, image.shape[1], left_curve, right_curve)
    else:
        curvatures = (0.0, 0.0)
        offcenter = 0.0
        output = np.array(cv2.merge((image, image, image)), np.uint8)

    return output, curvatures, offcenter
