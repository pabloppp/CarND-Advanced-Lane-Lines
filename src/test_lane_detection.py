import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils.image_pipeline import pipeline
from src.utils.lane_detection import detect_lanes

test_img = cv2.imread('../test_images/test8.jpg')
pipelined_img, unwarped, M, Minv, d_pre, d_post = pipeline(test_img, debug=True)
pipelined_img_lanes, pipelined_img_lanes_shape, curvatures, offcenter = detect_lanes(pipelined_img)

# cv2.imwrite('../output_images/test8_lanes_b.jpg', pipelined_img_lanes)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(pipelined_img, cmap="gray")
ax1.set_title('Original Image', fontsize=10)
ax2.imshow(pipelined_img_lanes)
ax2.set_title('Lanes Image', fontsize=10)
plt.show()

print("curvature", curvatures[0], curvatures[1], (curvatures[0] + curvatures[1]) / 2)
print("offcenter", offcenter)
