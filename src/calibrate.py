import pickle
import cv2
from src.utils.calibration import get_distortion_matrix
import matplotlib.pyplot as plt

print("# load test distorted image")
test_img = cv2.imread('../camera_cal/calibration1.jpg')
img_shape = (test_img.shape[1], test_img.shape[0])

print("# calculate calibration matrix from calibration samples")
mtx, dist = get_distortion_matrix("../camera_cal/calibration*.jpg", img_shape, (9, 6))

print("# generate undistorted image and save it")
test_img_undistorted = cv2.undistort(test_img, mtx, dist, None, mtx)
cv2.imwrite('../output_images/calibration1_undistorted.jpg', test_img_undistorted)

print("# save calibration data for further use")
distortion_mtx = {"mtx": mtx, "dist": dist}
pickle.dump(distortion_mtx, open("calibration_matrix.p", "wb"))

print("# plot the two images side by side")
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(test_img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(test_img_undistorted)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()
