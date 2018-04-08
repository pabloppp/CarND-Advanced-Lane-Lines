import cv2
import matplotlib.pyplot as plt

from src.utils.image_pipeline import undistort, birds_eye_view, pipeline

test_img = cv2.imread('../test_images/test8.jpg')

output_image, unwarped, M, Minv, d_pre, d_post = pipeline(test_img, debug=True)
cv2.imwrite('../output_images/test8_pipeline.jpg', output_image * 255)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(d_pre)
ax1.set_title('Original Image', fontsize=10)
ax2.imshow(d_post, cmap="gray")
ax2.set_title('Thresholded Binary Image', fontsize=10)
plt.show()
