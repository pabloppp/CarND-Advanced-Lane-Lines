import cv2

from src.utils.image_pipeline import pipeline, undistort, warp
from src.utils.lane_detection import detect_lanes


def highlight(img):
    undistorted_img = cv2.cvtColor(undistort(img), cv2.COLOR_BGR2RGB)
    pipelined_img, unwarped, M, Minv, d_pre, d_post = pipeline(img, debug=True)
    _, pipelined_img_lanes, curvatures, offcenter = detect_lanes(pipelined_img)
    unwarped_pipelined_img_lanes = warp(pipelined_img_lanes, Minv)

    combined = cv2.addWeighted(undistorted_img, 1, unwarped_pipelined_img_lanes, 0.3, 0)
    return combined, undistorted_img, unwarped_pipelined_img_lanes
