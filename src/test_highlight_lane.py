import cv2
import matplotlib.pyplot as plt
from src.utils.highlight_lane import highlight

test_img = cv2.imread('../test_images/test8.jpg')
combined, undistorted, lanes = highlight(test_img)

# cv2.imwrite('../output_images/test8_highlighted.jpg', cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(undistorted)
ax1.set_title('Original Image', fontsize=10)
ax2.imshow(combined)
ax2.set_title('Lanes Image', fontsize=10)
plt.show()
