

import os
import json
import datetime
import pathlib
import time
import imp
import cv2 as cv
import carla
from collections import deque

import torch
import carla
import numpy as np
from PIL import Image
from easydict import EasyDict

import torchvision.transforms as T
from leaderboard.autoagents import autonomous_agent
import math
import yaml

from optical_flow import optical_flow
from segmentation import segmentation

# from team_code_transfuser.alg1_pycuda import Algorithm1
from alg1_pycuda_copy import Algorithm1 #
from alg2 import Algorithm2
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from submission_agent import HybridAgent
from optical_flow import OpticalFlowVisualizer

from memory_profiler import profile

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

def get_entry_point():
    return "agent"

class agent(HybridAgent):
    def setup(self, path_to_conf_file):
        print(path_to_conf_file)
        print('initializing agent')
        super().setup(path_to_conf_file)
        self.track = autonomous_agent.Track.SENSORS
        self.fov = 120
        self.camera_width = 480
        self.camera_height = 240
        self.ppd = self.camera_width / self.fov # pixel:degree # assume the same for x and y
        # define ourselves # original size
        self.camera_size_x = 0.096
        self.camera_size_y = 0.048
        self.cam_config = {
            'width': 320,
            'height': 160,
            'fov': 60
        }
        
        # x and y should be equal given no distortion
        meters_per_pixel_x = self.camera_size_x / self.camera_width
        meters_per_pixel_y = self.camera_size_y / self.camera_height

        self.optical_flow = optical_flow(self.camera_height, self.camera_width, meters_per_pixel_x, meters_per_pixel_y)
        self.segmentation = segmentation()
        
        # https://github.com/carla-simulator/carla/issues/56
        # assume fov is measured horizontally
        self.focal_len = self.camera_size_x / 2 / np.tan(np.deg2rad(self.fov / 2))
        print("focal length: ", self.focal_len)
        X, Y = self.create_XY(meters_per_pixel_x, meters_per_pixel_y)
        self.X = X

        self.turning_radius = 3.7756/1.5
          
        # tune hyperparameter variables -> shape the tropazoid safety box
        x_deg_close=38
        x_deg_far=29
        y_deg=13
        const_temp = 1.23

        self.no_certification_required = np.full((self.camera_height, self.camera_width), True)
        x_mid = self.camera_width // 2
        y_mid = self.camera_height // 2

        # create tropazoid shape boundary in BEV
        temp_slope_xy = x_deg_far * self.ppd / self.camera_height # slope of BEV regtangular safety box in camera view
        temp_x = temp_slope_xy * y_deg*self.ppd

        # slope of BEV trapezoid safety box in camera view = delta x / delta y
        temp_dx = (x_deg_close/2*self.ppd - temp_x)
        temp_dy = (y_mid-y_deg*self.ppd) 
        x_y_slope = temp_dx / temp_dy
        # y_x_slope = 1/x_y_slope
        sin_theta = temp_dx / np.sqrt(temp_dx**2 + temp_dy**2) # angle of troposoid compare to a straight line
        
        for i in range(self.camera_height):
            for j in range(self.camera_width):
                # / line: -(j-x_mid) = (i-y_mid)*x_y_slope
                if -(j-x_mid) < (i-y_mid)*x_y_slope and (j-x_mid) < (i-y_mid)*x_y_slope and (i-y_mid) > (y_deg*self.ppd):  # region / \ -
                    self.no_certification_required[i,j] = False
    
        self.certification_offset = np.full((self.camera_height, self.camera_width), 0.0)
        offset_w = y_deg*self.ppd*x_y_slope # offset use to shift the line to the left and right
        
        for i in range(self.camera_height):
            for j in range(self.camera_width):
                if self.no_certification_required[i,j] == False:
                    # point distance (j,i) from line Aj + Bi + C = 0 -> D = |Aj + Bi + C|/sqrt(A^2 + B^2)
                    # distance from lines / \ -
                    temp = np.min(np.abs(np.array([(x_y_slope*i + j- x_y_slope*y_mid - x_mid)/np.sqrt(x_y_slope**2+1),
                                                                      (x_y_slope*i - j- x_y_slope*y_mid + x_mid)/np.sqrt(x_y_slope**2+1),
                                                                      (i-y_mid-y_deg*self.ppd)])))
                    self.certification_offset[i,j] = (temp ** const_temp)
                elif -(j-x_mid-2*offset_w) > (i-y_mid)*x_y_slope and (j-x_mid+2*offset_w) > (i-y_mid)*x_y_slope:  # in upper \_/ region
                    # transform distance in _ to be equal to distance in / \
                    temp = np.absolute(i-y_mid-y_deg*self.ppd) # * 2 * sin_theta 
                    self.certification_offset[i,j] = -(temp ** const_temp) * 2
                elif -(j-x_mid) < (i-y_mid)*x_y_slope: # /
                    temp = np.absolute(x_y_slope*i - j- x_y_slope*y_mid + x_mid)/np.sqrt(x_y_slope**2+1)
                    self.certification_offset[i,j] = -(temp ** const_temp) /2 / sin_theta *2
                else:
                    temp = np.absolute(x_y_slope*i + j- x_y_slope*y_mid - x_mid)/np.sqrt(x_y_slope**2+1)
                    self.certification_offset[i,j] = -(temp** const_temp) /2 / sin_theta *2
        self.certification_offset = 0.002 * self.certification_offset

        self.no_certification_required = np.full((self.camera_height, self.camera_width), False)

        is_sky = np.full((self.camera_height, self.camera_width), False)
        is_sky[0:self.camera_height // 2, :] = True
        self.no_certification_required = np.logical_or(is_sky, self.no_certification_required) # c

        # TODO test when turn-off safety certificate
        # self.no_certification_required = np.full((self.camera_height, self.camera_width), True)
        # self.certification_offset = np.full((self.camera_height, self.camera_width), -1000000)

        # calculate chi
        # algorithm 1 setup
        self.alg1_solver = Algorithm1(self.focal_len, (self.camera_width, self.camera_height), X, Y, self.certification_offset) 
        self.alg2_solver = Algorithm2()
        self.last_psi_e = None

        self.certify_threshold = 0.998

        cv.imshow("region satisfied: ", np.repeat(self.no_certification_required[:, :, np.newaxis].astype(np.uint8) * 255, 3, axis=2))

        # DEBUGGING VARIABLE
        self.current_directory = os.path.abspath(os.getcwd())
        self.data_directory = "/home/haoming/git/debug_data" # 
        self.timestep_counter = 0
        self.save_counter = 0
        self.alg_1_debug = np.zeros((self.camera_height, self.camera_width, 8, 4, 0))
        self.mask_status = np.zeros((self.camera_height, self.camera_width, 2, 0))
        self.global_counter = 0
        
        

    def create_XY(self, camera_x_idx_to_x, camera_y_idx_to_y):

        x_mid = self.camera_width // 2
        y_mid = self.camera_height // 2
        xaxis = np.linspace(-x_mid, x_mid, self.camera_width)
        yaxis = np.linspace(-y_mid, y_mid, self.camera_height)

        X, Y = np.meshgrid(xaxis, yaxis)
        X = X * camera_x_idx_to_x
        Y = Y * camera_y_idx_to_y
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)

        return X, Y

    def sensors(self):
        # add semantic segmentation
        result = super().sensors() 
        result += [
                    {
                        'type': 'sensor.camera.semantic_segmentation',
                        'x': 1.3, 'y': 0.0, 'z':2.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                        'id': 'semantics_front'
                    },
                    {
                        'type': 'sensor.camera.semantic_segmentation',
                        'x': 1.3, 'y': 0.0, 'z':2.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                        'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                        'id': 'semantics_left'
                    },
                    {
                        'type': 'sensor.camera.semantic_segmentation',
                        'x': 1.3, 'y': 0.0, 'z':2.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                        'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                        'id': 'semantics_right'
                    },
        ]
        return result
    #@profile
    def run_step(self, input_data, timestamp):
        """
        input_data: A dictionary containing sensor data for the requested sensors.
        The data has been preprocessed at sensor_interface.py, and will be given
        as numpy arrays. This dictionary is indexed by the ids defined in 
        sensor method.

        timestamp: A timestamp of the current simulation instant.

        returns: control 
        see https://carla.readthedocs.io/en/latest/python_api/#carlavehiclecontrol
        """
        self.global_counter = self.global_counter + 1

        # TODO: now the variable name is interfuser --> suppose to be transfuser
        interfuser_out = super().run_step(input_data, timestamp)
        # nominal_control = (interfuser_out.throttle, interfuser_out.steer)
        # print("nominal a: {}, nominal phi: {}".format(interfuser_out.throttle, interfuser_out.steer))
        delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
        interfuser_out.throttle = np.minimum(interfuser_out.throttle,1.0)
        
        print("interfuser raw throttle: {}, brake: {}, steer: {}".format(interfuser_out.throttle, interfuser_out.brake, interfuser_out.steer))
        # return interfuser_out
        
        v_e = input_data["speed"][1]["speed"]
        if v_e < 0.2 and interfuser_out.brake > 0:
            return interfuser_out
        interfuser_steer_normalized = interfuser_out.steer * v_e/self.turning_radius
        
        # if self.global_counter < 550:
            # return interfuser_out
        
        # process camera input
        self.bgr_ = input_data["rgb_front"][1][:, :, :3]  # rgb to rgb_front

        # scale down
        rgb_image = Image.fromarray(self.bgr_)
        new_size = (self.camera_width, self.camera_height) # H, H
        resized_image = rgb_image.resize(new_size, Image.ANTIALIAS)
        self.bgr = np.array(resized_image)
        self.bgr = self.bgr.astype(np.uint8)
        self.rgb = cv.cvtColor(self.bgr, cv.COLOR_BGR2RGB)

        # predictg output
        self.optical_flow_output = self.optical_flow.predict(self.bgr, delta_time) 

        is_road, is_nonanimated, is_terrain, is_animated = self.segmentation.predict_(self.rgb) # worked 
        # is_road, _, _, is_animated = self.segmentation.predict_gt_batch(semantic_seg)
        # is_road = self.segmentation.predict(self.rgb) # now 0,1 

        # get current vehicle state
        # m/s^2, in global xyz
        a_e_measured = np.linalg.norm(input_data["imu"][1][0:3])

        # rad/s, w.r.t z-axis
        psi_e = input_data["imu"][1][5]
        print("Real psi: {}".format(psi_e))
        print("Calculated psi: {}".format(interfuser_steer_normalized))

        # d psi_e / d t
        delta_psi_e = None
        if (self.last_psi_e is None):
            delta_psi_e = psi_e
        else:
            delta_psi_e = psi_e - self.last_psi_e
        self.last_psi_e = psi_e
        phi_e_measured = delta_psi_e / delta_time     

        # convert CARLA control actions to acceleration and steering rate
        if interfuser_out.throttle > 0:
            interfuser_acc = interfuser_out.throttle*(-2.64038-0.00917088*v_e+1.53601/v_e+0.000785623*np.exp(0.5*v_e))+interfuser_out.throttle*interfuser_out.throttle*(6.85333-0.181589*v_e+3.67525/v_e-0.000760838*np.exp(0.5*v_e))
            interfuser_acc = np.maximum(interfuser_acc,-3)
        else:
            interfuser_acc = -13
        interfuser_steering_rate = (interfuser_steer_normalized - psi_e) / delta_time
        nominal_control = (interfuser_acc, interfuser_steering_rate)
        print("interfuser acc: {}, steer_rate: {}".format(interfuser_acc,interfuser_steering_rate))
        
        return interfuser_out

    def destroy(self):
        super().destroy()
