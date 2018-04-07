import cv2
from src.utils.calibration import setup_undistort
import matplotlib.pyplot as plt

undistort = setup_undistort("calibration_matrix.p")

test_img = cv2.imread('../test_images/test4.jpg')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img_undistorted = undistort(test_img)

cv2.imwrite('../output_images/test4_undistorted.jpg', cv2.cvtColor(test_img_undistorted, cv2.COLOR_RGB2BGR))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(test_img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(test_img_undistorted)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()
