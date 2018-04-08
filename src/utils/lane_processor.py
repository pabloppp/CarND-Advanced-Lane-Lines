import cv2
from src.utils.image_pipeline import pipeline, undistort, warp, thresholded_binary_image
from src.utils.lane import Lane
from src.utils.lane_detection import detect_lanes, curv_error


class LaneProcessor:
    def __init__(self):
        self.index = 0
        self.Minv = None
        self.unwarped_lane_img = None
        self.lane = Lane()
        self.curvatures = (0.0, 0.0)
        self.offcenter = 0.0

    def undistort(self, image):
        # cv2.imwrite('../test_images/test12.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
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
        pipelined_img_lanes_borders, pipelined_img_lanes, curvatures, offcenter = detect_lanes(image, lane=self.lane)
        self.curvatures = curvatures
        self.offcenter = offcenter
        self.unwarped_lane_img = warp(pipelined_img_lanes, self.Minv)
        return pipelined_img_lanes_borders

    def draw_lanes(self, image):
        out = cv2.addWeighted(image, 1, self.unwarped_lane_img, 0.3, 0)
        avg_curvature = (self.curvatures[0] + self.curvatures[1]) / 2
        curvature_error = curv_error(self.curvatures)
        curvature_error_color = (255, 255, 255) if curvature_error < 1.8 else (255, 0, 0)
        cv2.putText(out, "{:6.1f}C ({:6.1f}L {:6.1f}R)".format(avg_curvature, self.curvatures[0], self.curvatures[1]),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.putText(out, "{:6.3f}m offcenter".format(self.offcenter),
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.putText(out, "Curv. error {:1.4f}".format(curvature_error),
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, curvature_error_color)
        return out
