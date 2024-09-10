import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage

from memory_profiler import profile

class OpticalFlowVisualizer():
    def __init__(self, frame_h, frame_w):
        self.frame_w = frame_w
        self.frame_h = frame_h
        # arrow_spacing = number of pixels between each arrow drawn
        self.arrow_spacing = 50
    
    @profile
    def quivers(self, flow):
        step = 10
        fig, ax = plt.subplots()
        X, Y = np.arange(0, flow.shape[1], step), np.arange(flow.shape[0], 0, -step)
        # the optical flow is down positive, so v must be negated
        U, V = flow[::step, ::step, 0], -flow[::step, ::step, 1]
        ax.quiver(X, Y, U, V, angles='xy')
        numpy_fig = mplfig_to_npimage(fig)
        return numpy_fig 
