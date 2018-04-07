from moviepy.editor import VideoFileClip, CompositeVideoClip

from src.utils.lane_processor import LaneProcessor

clip = VideoFileClip("../project_video.mp4")  # .subclip(0, 10)
laneProcessor = LaneProcessor()

undistorted = clip.fl_image(laneProcessor.undistort)
thresholded = undistorted.fl_image(laneProcessor.threshold)
birdseye = undistorted.fl_image(laneProcessor.birdseye)
# laneProcessor.clear_lanes()
lane = birdseye.fl_image(laneProcessor.detect_lanes)
final = undistorted.fl_image(laneProcessor.draw_lanes)

combo = CompositeVideoClip([final, thresholded.resize(0.3).set_pos((502, 10)), lane.resize(0.3).set_pos((886, 10))])
combo.write_videofile("../output_video.mp4", audio=False)
