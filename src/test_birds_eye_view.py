import cv2
import matplotlib.pyplot as plt

from src.utils.image_pipeline import undistort, birds_eye_view, pipeline

test_img = cv2.imread('../test_images/test1.jpg')

output_image, unwarped, M, Minv, _, _ = birds_eye_view(undistort(test_img))
# cv2.imwrite('../output_images/test4_birdseye.jpg', output_image)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(test_img)
ax1.set_title('Original Image', fontsize=10)
ax2.imshow(unwarped, cmap="gray")
ax2.set_title('Thresholded Binary Image', fontsize=10)
plt.show()