import glob
import cv2
import numpy as np
import pickle


def _initialize_object_points(n_horizontal, n_vertical):
    objp = np.zeros((n_horizontal * n_vertical, 3), np.float32)
    objp[:, :2] = np.mgrid[0:n_horizontal, 0:n_vertical].T.reshape(-1, 2)
    return objp


def get_distortion_matrix(input_path, image_dims, grid_shape=(9, 6)):
    objp = _initialize_object_points(grid_shape[0], grid_shape[1])

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(input_path)

    for index, file_name in enumerate(images):
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (grid_shape[0], grid_shape[1]), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image_dims, None, None)

    return mtx, dist


def setup_undistort(calibration_matrix_path):
    distortion_matrix = pickle.load(open(calibration_matrix_path, "rb"))
    mtx = distortion_matrix["mtx"]
    dist = distortion_matrix["dist"]

    return lambda img: cv2.undistort(img, mtx, dist, None, mtx)
