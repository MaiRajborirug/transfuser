import cv2 as cv 
import numpy as np
import sys

from visualize_optical_flow import OpticalFlowVisualizer
# from visualize_optical_flow import OpticalFlowVisualizer

from memory_profiler import profile

# adapted from https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
class optical_flow:
    def __init__(self, camera_height, camera_width, meters_per_pixel_x, meters_per_pixel_y): 
        # set up optical flow
        self.isFirstFrame = True
        self.hsv = None
        self.prvs = None
        self.savePath = "/home/haoming/InterFuser/optical_flow_output/"
        self.img_idx = 0
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.meters_per_pixel_x = meters_per_pixel_x
        self.meters_per_pixel_y = meters_per_pixel_y

        # viz
        self.visualize = True
        self.optical_flow_visualizer = OpticalFlowVisualizer(self.camera_height, self.camera_width)


    def unflatten(self, raw):
        # height, width, channels
        return raw.reshape((self.camera_height, self.camera_width, 3))
    
    def save_img(self, flow):
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        self.hsv[..., 0] = ang*180/np.pi/2
        self.hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(self.hsv, cv.COLOR_HSV2BGR)
        cv.imwrite(self.savePath + str(self.img_idx) + ".png", bgr)
        self.img_idx += 1
    
    #@profile
    def predict(self, frame, dt): 
        """
        frame: numpy ndarray with bgr colors.
        Dimension:
        (width,  height, 3)

        dt: delta time in seconds

        returns: numpy ndarray with optical flow at each pixel location
        dimension:
        (height, width, 2)
        """
        if self.isFirstFrame:
            self.prvs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            self.hsv = np.zeros_like(frame)
            self.hsv[..., 1] = 255
            self.isFirstFrame = False
        
        frame2 = frame
        next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        output = cv.calcOpticalFlowFarneback(self.prvs, next_frame, None, 0.5, 3, 20, 5, 5, 1.2, 0) 
        # zero out flow less than one pixel / frame
        output_x = output[:, :, 0] 
        gt_pos_one = np.greater(output_x, np.ones(output_x.shape)) 
        lt_neg_one = np.less(output_x, -1 * np.ones(output_x.shape)) 
        output_x = output_x * np.logical_or(gt_pos_one, lt_neg_one).astype(np.float)

        output_y = output[:, :, 1] 
        gt_pos_one = np.greater(output_y, np.ones(output_y.shape)) 
        lt_neg_one = np.less(output_y, -1 * np.ones(output_y.shape)) 
        output_y = output_y * np.logical_or(gt_pos_one, lt_neg_one).astype(np.float)



        # convert optical flow to meters 
        output_x = output_x * self.meters_per_pixel_x
        output_y = output_y * self.meters_per_pixel_y
        output = np.dstack((output_x, output_y))

        # divide by delta time
        output = output / dt

        self.prvs = next_frame
        # visualize
        # if self.visualize:
            # quiver_plot = self.optical_flow_visualizer.quivers(output)
            # cv.imshow("optical flow:", (quiver_plot))
            
        return output

    #@profile
    def predict_sf(self, frame1, frame2, dt): 
        """
        frame: numpy ndarray with rgb colors.
        Dimension:
        (width,  height, 3)

        dt: delta time in seconds

        returns: numpy ndarray with optical flow at each pixel location
        dimension:
        (height, width, 2)
        """
        # if self.isFirstFrame:
        #     self.prvs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #     self.hsv = np.zeros_like(frame)
        #     self.hsv[..., 1] = 255
        #     self.isFirstFrame = False

        self.prvs = cv.cvtColor(frame1, cv.COLOR_RGB2GRAY)
        self.hsv = np.zeros_like(frame1)
        self.hsv[..., 1] = 255
        
        next_frame = cv.cvtColor(frame2, cv.COLOR_RGB2GRAY)

        output = cv.calcOpticalFlowFarneback(self.prvs, next_frame, None, 0.5, 3, 20, 5, 5, 1.2, 0) 
        # zero out flow less than one pixel / frame
        output_x = output[:, :, 0] 
        gt_pos_one = np.greater(output_x, np.ones(output_x.shape)) 
        lt_neg_one = np.less(output_x, -1 * np.ones(output_x.shape)) 
        output_x = output_x * np.logical_or(gt_pos_one, lt_neg_one).astype(np.float)

        output_y = output[:, :, 1] 
        gt_pos_one = np.greater(output_y, np.ones(output_y.shape)) 
        lt_neg_one = np.less(output_y, -1 * np.ones(output_y.shape)) 
        output_y = output_y * np.logical_or(gt_pos_one, lt_neg_one).astype(np.float)



        # convert optical flow to meters 
        output_x = output_x * self.meters_per_pixel_x
        output_y = output_y * self.meters_per_pixel_y
        output = np.dstack((output_x, output_y))

        # divide by delta time
        output = output / dt

        # visualize
        # if self.visualize:
            # quiver_plot = self.optical_flow_visualizer.quivers(output)
            # cv.imshow("optical flow:", (quiver_plot))
            
        return output