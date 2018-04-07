import cv2
from src.utils.image_pipeline import pipeline, undistort, warp, thresholded_binary_image
from src.utils.lane_detection import detect_lanes


class LaneProcessor:
    def __init__(self):
        self.index = 0
        self.Minv = None
        self.lane = None

    def add_label(self, image):
        out = image
        cv2.putText(out, "Label {}".format(self.index), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        self.index += 1
        return out

    def undistort(self, image):
        return undistort(image)
    
    def threshold(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(thresholded_binary_image(image) * 255, cv2.COLOR_GRAY2RGB)

    def birdseye(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        pipelined_img, unwarped, M, Minv, _, _ = pipeline(image)
        self.Minv = Minv
        return cv2.cvtColor(pipelined_img * 255, cv2.COLOR_GRAY2RGB)

    def detect_lanes(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pipelined_img_lanes, _, _ = detect_lanes(image, shape_only=True)
        pipelined_img_lanes_borders, _, _ = detect_lanes(image)
        unwarped_pipelined_img_lanes = warp(pipelined_img_lanes, self.Minv)
        self.lane = unwarped_pipelined_img_lanes
        return pipelined_img_lanes_borders

    def draw_lanes(self, image):
        out = cv2.addWeighted(image, 1, self.lane, 0.3, 0)
        return out
