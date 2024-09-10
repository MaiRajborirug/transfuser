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

from torchvision import transforms
from leaderboard.autoagents import autonomous_agent
import math
import yaml

from optical_flow import optical_flow
from segmentation import segmentation

from team_code_transfuser.alg1_pycuda_ import Algorithm1
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
        super().setup(path_to_conf_file)
        self.track = autonomous_agent.Track.SENSORS
        self.fov = 100
        # self.camera_width = 800
        # self.camera_height = 600
        # # meters
        # self.camera_size_x = 0.08
        # self.camera_size_y = 0.06
        self.camera_width = 960
        self.camera_height = 480
        # meters
        self.camera_size_x = 0.096
        self.camera_size_y = 0.048
        
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

        # WARNING: must be consistent with CUDA macros
        H_BAR = 0.2
        R = 1.0
        f = self.focal_len
        def condition_22_is_satisfied(X,Y):
            H_BAR_SQ = H_BAR * H_BAR
            pos_term = H_BAR_SQ * (X * X + f * f)
            neg_term = Y * Y * R * R
            return not (pos_term <= neg_term)
        
        vfunc = np.vectorize(condition_22_is_satisfied)
        self.no_certification_required = vfunc(X, Y)
        
        # new region for no certificatio required
        self.no_certification_required = np.full((self.camera_height, self.camera_width), True)
        for i in range(self.camera_height):
            for j in range(self.camera_width):
                if j < 5/6*i+150 and j > -5/6*i+650 and i > 400:
                    self.no_certification_required[i,j] = False
        self.certification_offset = np.full((self.camera_height, self.camera_width), 0.0)
        for i in range(self.camera_height):
            for j in range(self.camera_width):
                if self.no_certification_required[i,j] == False:
                    self.certification_offset[i,j] = np.min(np.array([np.absolute(5/6*i-j+150)/np.sqrt((5/6)**2+1),np.absolute(-5/6*i-j+650)/np.sqrt((5/6)**2+1),np.absolute(i-400)]))
                elif j > -6/5*i+2000/6+630:
                    self.certification_offset[i,j] = -np.absolute(5/6*i-j+150)/np.sqrt((5/6)**2+1)
                elif j < 6/5*i-2000/6+170:
                    self.certification_offset[i,j] = -np.absolute(-5/6*i-j+650)/np.sqrt((5/6)**2+1)
                elif j > -2000/6+650 and j < 2000/6+150:
                    self.certification_offset[i,j] = -np.absolute(i-400)
                elif j > 400:
                    self.certification_offset[i,j] = -np.sqrt((i-400)**2 + (j-2000/6-150)**2)
                else:
                    self.certification_offset[i,j] = -np.sqrt((i-400)**2 + (j+2000/6-650)**2)
        self.certification_offset = 0.001*self.certification_offset

        self.no_certification_required = np.full((self.camera_height, self.camera_width), False)

        is_sky = np.full(X.shape, False)
        is_sky[0:X.shape[0] // 2, :] = True
        self.no_certification_required = np.logical_or(is_sky, self.no_certification_required)

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

        print("X row:", X[0, :])
        print("Y col:", Y[:, 0])
        return X, Y

    def sensors(self):   
        return super().sensors()
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
        # discards alpha chaf
        self.bgr = input_data["rgb_front"][1][:, :, :3]  # rgb to rgb_front
        # TEMPORARY: record a reference performance
        # self.timestep_counter = self.timestep_counter + 1
        # if self.timestep_counter % 25 == 0:
            # cv.imwrite(self.data_directory + "/image_" + str(self.save_counter) + ".png", bgr)
            # self.save_counter = self.save_counter + 1
        # return interfuser_out


        self.rgb = cv.cvtColor(self.bgr, cv.COLOR_BGR2RGB)
        self.optical_flow_output = self.optical_flow.predict(self.bgr, delta_time) 

        is_road = self.segmentation.predict(self.rgb)
        # print("fraction is road:", is_road.sum() / is_road.size)

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

        # m/s
        

        # Orientation in radians. North is (0.0, -1.0, 0.0)
        self.theta = input_data["imu"][1][-1]

        # convert optical flow to mu and nu
        mu = self.optical_flow_output[:, :, 0]
        nu = self.optical_flow_output[:, :, 1]
        mu = mu.astype(np.float32)
        nu = nu.astype(np.float32)
        

        # convert CARLA control actions to acceleration and steering rate
        if interfuser_out.throttle > 0:
            interfuser_acc = interfuser_out.throttle*(-2.64038-0.00917088*v_e+1.53601/v_e+0.000785623*np.exp(0.5*v_e))+interfuser_out.throttle*interfuser_out.throttle*(6.85333-0.181589*v_e+3.67525/v_e-0.000760838*np.exp(0.5*v_e))
            interfuser_acc = np.maximum(interfuser_acc,-3)
        else:
            interfuser_acc = -13
        interfuser_steering_rate = (interfuser_steer_normalized - psi_e) / delta_time
        nominal_control = (interfuser_acc, interfuser_steering_rate)
        print("interfuser acc: {}, steer_rate: {}".format(interfuser_acc,interfuser_steering_rate))
        # steering_limit = np.minimum(4/v_e,np.pi/8*v_e)
        steering_limit = v_e/self.turning_radius
        self.alg2_solver.phi_bounds = ((-steering_limit-psi_e)/delta_time,(steering_limit-psi_e)/delta_time)


        # TODO: delete after done testing max acceleration
        # control = carla.VehicleControl()

        # steer = 0
        # brake = 0 
        # throttle = 1
        # control.steer = float(steer)
        # control.throttle = float(throttle)
        # control.brake = float(brake)

        # print(control.throttle, a_e_measured, v_e)
        # return control



        # FOR TESTING: certification status in extreme control actions
#        self.raw_data_from_alg_1 = np.zeros((self.camera_height, self.camera_width, 8, 4, 1))
#        _, self.raw_data_from_alg_1_current = self.alg1_solver.run((mu, nu, v_e, psi_e, self.alg2_solver.a_bounds[0], self.alg2_solver.phi_bounds[0]))
#        self.raw_data_from_alg_1[:,:,:,0,0] = self.raw_data_from_alg_1_current
#        _, self.raw_data_from_alg_1_current = self.alg1_solver.run((mu, nu, v_e, psi_e, self.alg2_solver.a_bounds[0], self.alg2_solver.phi_bounds[1]))
#        self.raw_data_from_alg_1[:,:,:,1,0] = self.raw_data_from_alg_1_current
#        _, self.raw_data_from_alg_1_current = self.alg1_solver.run((mu, nu, v_e, psi_e, self.alg2_solver.a_bounds[1], self.alg2_solver.phi_bounds[0]))
#        self.raw_data_from_alg_1[:,:,:,2,0] = self.raw_data_from_alg_1_current
#        _, self.raw_data_from_alg_1_current = self.alg1_solver.run((mu, nu, v_e, psi_e, self.alg2_solver.a_bounds[1], self.alg2_solver.phi_bounds[1]))
#        self.raw_data_from_alg_1[:,:,:,3,0] = self.raw_data_from_alg_1_current
#        self.alg_1_debug = np.append(self.alg_1_debug, self.raw_data_from_alg_1, axis=4)
        self.timestep_counter = self.timestep_counter + 1
#        self.mask_status_debug = np.zeros((self.camera_height, self.camera_width, 2, 1))
#        self.mask_status_debug[:,:,0,0] = self.certification_offset
#        self.mask_status_debug[:,:,1,0] = is_road
#        self.mask_status = np.append(self.mask_status, self.mask_status_debug, axis=3)
        if self.timestep_counter % 10 == 0:
            print("Saving to ")
            print(self.data_directory)
#            while os.path.exists(self.data_directory + "/debug_output_" + str(self.save_counter) + ".npy"):
#                self.save_counter = self.save_counter + 100000
#            # if os.path.exists(self.data_directory + "/image_" + str(self.save_counter) + ".npy"):
#                # os.remove(self.data_directory + "/image_" + str(self.save_counter) + ".npy")
#            # if os.path.exists(self.data_directory + "/image_" + str(self.save_counter) + ".png"):
#                # os.remove(self.data_directory + "/image_" + str(self.save_counter) + ".png")
#            # if os.path.exists(self.data_directory + "/mask_" + str(self.save_counter) + ".npy"):
#                # os.remove(self.data_directory + "/mask_" + str(self.save_counter) + ".npy")
#            with open(self.data_directory + "/debug_output_" + str(self.save_counter) + ".npy", "wb") as f:
#                np.save(f, self.alg_1_debug)
#            with open(self.data_directory + "/image_" + str(self.save_counter) + ".npy", "wb") as f:
#                np.save(f, self.bgr)
#            with open(self.data_directory + "/mask_" + str(self.save_counter) + ".npy", "wb") as f:
#                np.save(f, self.mask_status)
            cv.imwrite(self.data_directory + "/image_" + str(self.save_counter) + ".png", self.bgr)
#            self.alg_1_debug = np.zeros((self.camera_height, self.camera_width, 8, 4, 0))
#            self.mask_status = np.zeros((self.camera_height, self.camera_width, 2, 0))
            self.save_counter = self.save_counter + 1

        # certify nominal control action first
        self.nominal_pixel_is_certifieds, self._ = self.alg1_solver.run((mu,nu,v_e,psi_e,interfuser_acc,interfuser_steering_rate))
        self.nominal_pixel_is_certifieds = np.logical_or(is_road, self.nominal_pixel_is_certifieds)
        self.nominal_pixel_is_certifieds = np.logical_or(self.nominal_pixel_is_certifieds, self.no_certification_required)
        if self.nominal_pixel_is_certifieds.sum() / self.nominal_pixel_is_certifieds.size > self.certify_threshold:
            return interfuser_out

        def certify_control_action(control_action):
            a_e, phi_e = control_action
            # print("certifying:", a_e, phi_e)
            pixel_is_certifieds, _ = self.alg1_solver.run((mu, nu, v_e, psi_e, a_e, phi_e))


            # if (v_e > -0.001 and v_e < 0.001):
            #     print("certified due to being stationary")
            #     return True

            # the pixel is automatically certified if it is 
            # the road (in other words, not obstacle)
            pixel_is_certifieds = np.logical_or(is_road, pixel_is_certifieds)

            # we only want to check certification for those regions we care about
            pixel_is_certifieds = np.logical_or(pixel_is_certifieds, self.no_certification_required)

            # print("is certified: ", pixel_is_certifieds.sum() / pixel_is_certifieds.size > self.certify_threshold)
            cv.imshow("certified, post mask", np.repeat(pixel_is_certifieds[:, :, np.newaxis].astype(np.uint8) * 255, 3, axis=2))
            # return pixel_is_certifieds.sum() / pixel_is_certifieds.size > self.certify_threshold
            return pixel_is_certifieds.sum() / pixel_is_certifieds.size

        res = self.alg2_solver.run(nominal_control, certify_control_action)

        if res is None:
            print("no control action is certified!")
            return interfuser_out
        else:
            (a_control, phi_control) = res

        # TODO: find conversion between throttle/brake and a
        # TODO: find coversion between steering and phi
        # measure how acceleration reacts to throttle and brake
        # measure how phi reacts to steering 
        # or hopefully find some formula provided by carla

        # NEW: DOESN'T WORK: changed to VehicleAckermannControl
        # control = carla.VehicleAckermannControl()
        # control.steer_speed = float(phi_control)
        # control.acceleration = float(a_control)
        print("acc: {}, steer_rate: {}".format(a_control,phi_control))

        steer = psi_e + delta_time*phi_control*self.turning_radius/v_e
        # steer = interfuser_out.steer
        if a_control > 0:
            coeff_a = 6.85333-0.181589*v_e+3.67525/v_e-0.000760838*np.exp(0.5*v_e)
            coeff_b = -2.64038-0.00917088*v_e+1.53601/v_e+0.000785623*np.exp(0.5*v_e)
            coeff_c = -a_control
            sol1 = (-coeff_b-np.sqrt(coeff_b*coeff_b-4*coeff_a*coeff_c))/(2*coeff_a)
            sol2 = (-coeff_b+np.sqrt(coeff_b*coeff_b-4*coeff_a*coeff_c))/(2*coeff_a)
            if np.minimum(sol1,sol2) > 0:
                throttle = np.minimum(sol1,sol2)
            else:
                throttle = np.maximum(sol1,sol2)
            if throttle < 0:
                print("How?")
                return interfuser_out
            brake = 0
        else:
            throttle = 0
            brake = 1
        throttle = np.minimum(throttle,interfuser_out.throttle)
        brake = np.maximum(brake,interfuser_out.brake)

        
        # OLD: VehicleControl        
        # steer = phi_control
        # throttle = a_control
        # brake = 0
         #if (a_control < 0):
             #brake = a_control
             #throttle = 0

        print("throttle: {}, brake: {}, steer: {}".format(throttle, brake, steer))
        
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        return control

    def destroy(self):
        super().destroy()
