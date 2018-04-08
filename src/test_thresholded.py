import cv2
import matplotlib.pyplot as plt

from src.utils.image_pipeline import thresholded_binary_image, undistort

test_img = cv2.imread('../test_images/test4.jpg')
output_image = thresholded_binary_image(undistort(test_img)) * 255

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image', fontsize=10)
ax2.imshow(output_image, cmap="gray")
ax2.set_title('Thresholded Binary Image', fontsize=10)
plt.show()

# cv2.imwrite('../output_images/test8_binary_threshold.jpg', output_image)
