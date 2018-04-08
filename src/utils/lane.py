import numpy as np


class Lane:
    def __init__(self):
        self.detected = False
        self.left_curve = None
        self.right_curve = None
        self.left_curve_history = []
        self.right_curve_history = []

    def left_curve_smooth(self, i, window=4):
        # trim everyting except the last N curve values to avoid a memory overload
        self.left_curve_history = self.left_curve_history[-window:]
        return np.mean([left_curve(i) for left_curve in self.left_curve_history])

    def right_curve_smooth(self, i, window=4):
        self.right_curve_history = self.right_curve_history[-window:]
        return np.mean([right_curve(i) for right_curve in self.right_curve_history])
