
# update alg 1, 2
import os
import json
import datetime
import pathlib
import time
import imp
import cv2
import carla
from collections import deque
import sys
import torch
import carla
import numpy as np
from PIL import Image
from easydict import EasyDict
from config import GlobalConfig

import torchvision.transforms as T
from leaderboard.autoagents import autonomous_agent
import math
import yaml
import matplotlib.pyplot as plt

from optical_flow import optical_flow
# from segmentation import segmentation
from rgb_seg import UNet, Unetpad, group_segment

# import alg1_objavoid3
from alg1_pr10 import Algorithm1 #
from alg2_10 import Algorithm2



from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
# NOTE: adjust submission_agent
from tf_04_noise import HybridAgent
# NOTE: get noise level
NOISE = float(os.environ.get('NOISE')) # to run step -> d_std
from optical_flow import OpticalFlowVisualizer
import torchvision.transforms as transforms
from memory_profiler import profile

# NOTE: for depth estimationƒ
sys.path.append('/media/haoming/970EVO/Yaguang/depth_est/monodepth2')
import networks  
sys.path.append('/media/haoming/970EVO/Yaguang/depth_est/')
from monodepth2.utils import download_model_if_doesnt_exist

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
        self.track = autonomous_agent.Track.MAP
        self.fov = 120
        self.camera_width = 960
        self.camera_height = 480
        self.meters_per_pixel_x = 1.0
        self.meters_per_pixel_y = 1.0
        self.ppd = self.camera_width / self.fov # pixel:degree # assume the same for x and y
        # define ourselves # original size
        self.focal_len = (self.camera_width /2) / np.tan(np.deg2rad(self.fov/2)) # 277.128

        
        # NOTE: for control mapping
        self.delta_time = 0.05 # CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
        self.throttle_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1, n=20) # throttle PID controller
        self.c_speed_sqrt = 0.032 # constant for throttle_control
        self.c_acc = 0.025
        self.c_w_sq = 0.04
        self.c_w = 0.01
        self._step = -1

        
        # NOTE: legacy parameters
        self.turning_radius = 2.21 # This is the minimum turning radius of the vehicle when steer = 1
        self.agent_front2back_whl = 2.9 # This property are from agent_vehicle.get_physics_control()
        self.agent_backwhl2cm = 1.75 # This property are from agent_vehicle.get_physics_control()
        self.camera_size_x = 960
        self.camera_size_y = 480
        self._set_certificate_bound(x_deg_close=45, x_deg_far=35, y_deg=12, decay_temp=1.19, offset_scale=0.004)
        turn_KI = 0.5
        turn_KP = 1.0
        turn_KD = 0.5
        turn_n = 20 # buffer size
        self.turn_controller = PIDController(K_P=turn_KP, K_I=turn_KI, K_D=turn_KD, n=turn_n)
        self.aug_degrees = [0]


        # NOTE: for certificate bound
        self.certify_threshold = 0.98
        
        # NOTE: for optical flow
        self.optical_flow = optical_flow(self.camera_height, self.camera_width, self.meters_per_pixel_x, self.meters_per_pixel_y)

        # # NOTE: for semantic segmentation -> don't need for now
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.segment_model = UNet(in_channels=3, filters=32, n_classes=29)
        # self.segment_model  = self.segment_model .to(self.device)
        # segment_state_path = '/media/haoming/970EVO/Pharuj/git/DOS/testing_codes/carla-image-segmentation-model-resized-state-dict.pth'
        # self.segment_model.load_state_dict(torch.load(segment_state_path))
        # self.transform = transforms.Compose([
        #     transforms.Resize((256, 256)),  # Resizes to 256x256. Change this line if aspect ratio needs to be preserved
        #     transforms.ToTensor(),  # Converts to a tensor and scales to [0, 1]
        #     # already convert from (H, W, 3) to (3, 256, 256)
        # ])
        
        self.d_std = NOISE # bound
        
        """
        # NOTE: for depth estimation monodepth2 ---------
        model_name = "mono+stereo_1024x320"
        download_model_if_doesnt_exist(model_name)
        encoder_path = os.path.join("models", model_name, "encoder.pth")
        depth_decoder_path = os.path.join("models", model_name, "depth.pth")
        # LOADING PRETRAINED MODEL
        self.encoder = networks.ResnetEncoder(18, False)
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(loaded_dict)

        self.encoder.eval()
        self.depth_decoder.eval()

        # tune the value of k
        self.disp_k = 0.36
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        """    

        # NOTE: test when turn-off safety certificate
        # self.no_certification_required = np.full((self.camera_height, self.camera_width), True)
        # self.certification_offset = np.full((self.camera_height, self.camera_width), -1000000)

        # NOTE: define safety certificate solver edit 09/27 add path here
        self.cuda_path = '/media/haoming/970EVO/pharuj/git/transfuser/team_code_transfuser/alg1_pr10.cu' # obstacle avoidance path
        self.alg1_solver = Algorithm1(self.focal_len, (self.camera_width, self.camera_height), self.X, self.Y, self.certification_offset, self.cuda_path)
        self.alg2_solver = Algorithm2()
    
        
        # NOTE: for cruise controller
        self.w_e_deque = deque([0 for _ in range(3)], maxlen=3)
        self.w_e_weight = [0.1, 0.3, 0.6]
        self.w_e_weight2 = [0.1, 0.3, 0]
        self.v_e_deque = deque([0 for _ in range(3)], maxlen=3)
        self.rho_e_deque = deque([0 for _ in range(3)], maxlen=3) # for rho_e = 1/turning_radius
        self.steer_deque = deque([0 for _ in range(10)], maxlen=10)
        
        self.cruise_controller = PIDController(K_P=0.25, K_I=0.5, K_D=0.2, n=20)
        self.desired_speed =5.0
        


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
        
        start_time = time.time() 
        
        # NOTE: for cruise controller -------
        self.a_e = input_data["imu"][1][0]
        self.v_e = np.maximum(0,input_data["speed"][1]["speed"])
        delta = self.desired_speed - self.v_e
        # delta = np.clip(math.sqrt(max(0,self.v_e)) * self.c_speed_sqrt + a_control * self.c_acc + self.w_e**2 * self.c_w_sq + abs(self.w_e) * self.c_w, 0.0, 0.25)
        control_signal = self.cruise_controller.step(delta)
        # throttle = np.clip(throttle+0.15, 0.0, 0.75)
        if control_signal > 0:
            throttle = np.clip(control_signal, 0.0, 1.0)
            brake = 0.0
        elif control_signal > -1.: # weak signal -> slow down not exactly brake
            throttle = 0.0
            brake = 0.0
        else: # brake
            throttle = 0.0
            brake = np.clip(-0.1*control_signal, 0.0, 1.0)
        
        control = carla.VehicleControl() 
        control.throttle = throttle
        control.brake = brake
        control.steer = 0.0
        self._step +=1
        # print(f"[{self.v_e:.4f}, {input_data['imu'][1][5]:.4f}],")
            
        #------ start old algorithm -----
        delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
        
        # to find the steer mapping
        # print(f"[{self.v_e:.5f}, {input_data['imu'][1][5]:.5f}, {control.steer:.2f}],")
        # return control
        
        prefollow_time = time.time()
        
        # for fast testing, since the vehicle only run after step ~ 50, we can skip the first 50 steps
        if self._step < 60:
            self.start_time = time.time()
            return control
        
        if self._step >1:
            print(f"step:{self._step}, op_time:{time.time()-self.start_time:.3f}")
            self.start_time = time.time()
        
        # NOTE: optical flow output
        self.bgr_ = input_data["rgb_front"][1][:, :, :3]  # rgb to rgb_front
        self.optical_flow_output = self.optical_flow.predict(self.bgr_, self.delta_time)
        
        # NOTE: gt segmentation
        pred_mask = input_data["semantics_front"][1][:, :, 2] # semantic to semantic_front
        # pred_mask = self._distort(pred_mask, value=5, n_labels=24) # remove it for target following
        
        # NOTE: CARLA segmentation, initialize gt segmentation # edit 09/27 add is_wp
        is_wp, is_road, is_nonanimated, is_terrain, is_animated, group_label = group_segment(pred_mask)
        is_obstacle = np.logical_not(is_road)
        # self.resize_visualize(group_label, 'segment label', binary_input=False)
        
        # NOTE: visualize raw input
        self.resize_visualize(self.bgr_, 'input', binary_input=False)  # NOTE: debug attempt
        cv2.waitKey(1)
        
        """
        # NOTE: monodepth2 estimation------
        input_image = Image.fromarray(cv2.cvtColor(self.bgr_, cv2.COLOR_BGR2RGB)).convert('RGB')
        original_width, original_height = input_image.size
        input_image_resized = input_image.resize((self.feed_width, self.feed_height), Image.LANCZOS)
        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
        with torch.no_grad():
            features = self.encoder(input_image_pytorch)
            outputs = self.depth_decoder(features)
        disp = outputs[("disp", 0)]
        arr = disp.squeeze().detach().cpu().numpy()
        # cv2.imshow("distance", arr)
        disp_resized = torch.nn.functional.interpolate(disp,
                                                       (original_height, original_width), mode="bilinear", align_corners=False)
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        distance = self.disp_k / disp_resized_np
        """
        
        # NOTE: depth from depth camera
        depth_bgr = input_data["depth_front"][1][:, :, :3].astype(np.float32)
        normalized = (depth_bgr[:, :, 0] * 256 * 256 + depth_bgr[:, :, 1] * 256 + depth_bgr[:, :, 2]) 
        normalized = normalized / (256 * 256* 256 -1) 
        distance = 1000 * normalized # convert to meters
        
        d_upper = distance * np.exp(self.d_std)
        d_lower = distance * np.exp(-self.d_std)
        
        # NOTE: for certificate and control mapping

        # NOTE: require parameter for both target following and obstacle avoidance        
        self.w_e_deque.append(input_data['imu'][1][5])
        self.w_e = np.average(self.w_e_deque, weights=self.w_e_weight)
        self.v_e = input_data['speed'][1]['speed']
        self.v_e_deque.append(self.v_e)
        self.alpha_e = (np.average(self.w_e_deque, weights=self.w_e_weight) - np.average(self.w_e_deque, weights=self.w_e_weight2)) / self.delta_time
        
        # get alpha_e from steer
        rho = control.steer / self.config.m_rs
        w_e_new = rho * (self.v_e + self.config.c_rs)
        self.alpha_e = (w_e_new - self.w_e) / self.delta_time
        # self.alpha_e = 0
        
        self.a_e = input_data["imu"][1][0] # accelaration in vehicle direction (m/s^2)
        self.rho_e_deque.append(input_data['imu'][1][5]/(abs(input_data['speed'][1]['speed'])+1e-3)) # add 1e-2 to avoid division by zero, rho is error when the car hitting
        control_acc = self.a_e
        control_steering_rate = self.alpha_e
        
        # convert optical flow to mu and nu
        self.mu = self.optical_flow_output[:, :, 0]
        self.nu =self.optical_flow_output[:, :, 1]
        self.mu = self.mu.astype(np.float32)
        self.nu =self.nu.astype(np.float32)
                
        
        # # NOTE: obstacle avoidance----------
        # self.nominal_pixel_is_certifieds, self._ = self.alg1_solver.run(( self.mu,self.nu,self.v_e,self.w_e,control_acc,control_steering_rate, is_animated, is_obstacle, d_upper, d_lower))
        # self.nominal_pixel_is_certifieds = np.logical_or(is_road, self.nominal_pixel_is_certifieds) # remove road pixels
        # self.nominal_pixel_is_certifieds = np.logical_or(self.nominal_pixel_is_certifieds, self.no_certification_required) # remove sky pixels        
        # self.resize_visualize(self.nominal_pixel_is_certifieds, 'objects not avoid')
        
        # if self.nominal_pixel_is_certifieds.sum() / self.nominal_pixel_is_certifieds.size > self.certify_threshold:
        #     # donotion
        #     do = 'nothing'
        #     # delta = np.clip(math.sqrt(max(0,self.v_e)) * self.c_speed_sqrt + self.a_e * self.c_acc + self.w_e**2 * self.c_w_sq + abs(self.w_e) * self.c_w, 0.0, 0.25)
        #     # _ = self.throttle_controller.step(delta)

        # else:
        #     nominal_control = (control_acc, control_steering_rate)
        #     # print("nominal control action: {:.3%}, {:.3%}".format(control_acc, control_steering_rate))
            
        #     def certify_control_action(control_action):
        #         self.a_e, self.alpha_e = control_action
        #         # print("certifying:", self.a_e, self.alpha_e)
        #         pixel_is_certifieds, _ = self.alg1_solver.run(( self.mu, self.nu, self.v_e, self.w_e, self.a_e, self.alpha_e, is_animated, is_obstacle, d_upper, d_lower))

        #         # the pixel is automatically certified if it is 
        #         # the road (in other words, not obstacle)
        #         pixel_is_certifieds = np.logical_or(is_road, pixel_is_certifieds)

        #         # we only want to check certification for those regions we care about
        #         pixel_is_certifieds = np.logical_or(pixel_is_certifieds, self.no_certification_required)
                            
        #         return pixel_is_certifieds.sum() / pixel_is_certifieds.size
        
        #     res = self.alg2_solver.run(nominal_control, certify_control_action)

        #     if res is not None:
   
        #         (a_control, alpha_control) = res
        #         # NOTE: emergency brake
        #         # control.throttle = 0.0
        #         # control.steer = 0.0
        #         # control.brake = 1.0
        #         # return control
                
        #         self.a_e = control_acc # before SC 2
                
                
        #         if a_control > -2:             
        #             delta = np.clip(math.sqrt(max(0,self.v_e)) * self.c_speed_sqrt + a_control * self.c_acc + self.w_e**2 * self.c_w_sq + abs(self.w_e) * self.c_w, 0.0, 0.25)
        #             throttle = self.throttle_controller.step(delta)
        #             throttle = np.clip(throttle, 0.2, 0.75)
        #             # throttle = np.maximum(control.throttle /2, 0.25)
        #             brake = 0.0
        #         else:
        #             throttle = 0.0
        #             brake = 1.0
                
        #         # steer mapping 2
        #         gain = 12.0 # more stechgth to steer 
        #         desire_w_e = self.w_e + gain * (self.delta_time * alpha_control) # gain
        #         desire_rho = desire_w_e / (abs(self.v_e) + self.config.c_rs) # appear when w happen without v_e
        #         self.rho_e_deque.append(desire_rho)
        #         steer = self.config.m_rs * desire_rho
        #         # steer = self.turn_controller.step(steer) # finally we want to have steer ~ 0
        #         steer = np.clip(steer, -1.0, 1.0)
                    
        #         print("avoid: ", a_control, alpha_control)
        #         print(f'avoid:{self._step}, w_e:{self.w_e:.3e}, old/new_alpha:{self.alpha_e:.3f}/{alpha_control:.3f}, old/new_steer: {control.steer:.2f}/{steer:.3f}')
        #         print(f'old/new a,{self.a_e:.3f}/{a_control:.3f}, old/new throttle: {control.throttle:.2f}/{throttle:.2f}, brake: {brake}')
                
        #         control.throttle = throttle
        #         control.steer = steer
        #         control.brake = 0.0
                
        #         # print(f"step: {self.step}, pre follow time: {prefollow_time-start_time:.3f}, "
        #         #       f"postfollow time: {time.time() - prefollow_time:.3f}")

        # # #         # NOTE: map (a, alpha) back to carla control (throttle, steering, break)
        # #         if abs(a_control) < 2 * abs(alpha_control) * self.config.m_rs and (a_control > -2): # prioritize turning arbitary comparison
        # #             # print('sc case 1')
        # #             # throttle mapping
        # #             a_control = control_acc # before SC 2              
        # #             delta = np.clip(math.sqrt(max(0,self.v_e)) * self.c_speed_sqrt + a_control * self.c_acc + self.w_e**2 * self.c_w_sq + abs(self.w_e) * self.c_w, 0.0, 0.25)
        # #             throttle = self.throttle_controller.step(delta)
        # #             throttle = np.clip(throttle, 0.2, 0.75)
        # #             # throttle = np.maximum(control.throttle /2, 0.25)
        # #             brake = 0
                    
        # #             # steer mapping 2
        # #             gain = 12.0 # more stechgth to steer 
        # #             desire_w_e = self.w_e + gain * (self.delta_time * alpha_control) # gain
        # #             desire_rho = desire_w_e / (abs(self.v_e) + self.config.c_rs) # appear when w happen without v_e
        # #             self.rho_e_deque.append(desire_rho)
        # #             steer = self.config.m_rs * desire_rho
        # #             # steer = self.turn_controller.step(steer) # finally we want to have steer ~ 0
        # #             steer = np.clip(steer, -1.0, 1.0)
                    
        # #             control.throttle = throttle
        # #             control.steer = steer
        # #             control.brake = 0.0
        # #             print('normal target vaoid')
        # #         else: 
        # #             control.throttle = 0.0
        # #             control.steer = 0.0
        # #             control.brake = 1.0
        # #             print('emergency brake')
                    
        #         self.nominal_pixel_is_certifieds, self._ = self.alg1_solver.run(( self.mu,self.nu,self.v_e,self.w_e,a_control,alpha_control, is_animated, is_obstacle, d_upper, d_lower))
        #         self.nominal_pixel_is_certifieds = np.logical_or(is_road, self.nominal_pixel_is_certifieds) # remove road pixels
        #         self.nominal_pixel_is_certifieds = np.logical_or(self.nominal_pixel_is_certifieds, self.no_certification_required) # remove sky pixels        
        #         self.resize_visualize(self.nominal_pixel_is_certifieds, 'object not avoid, adjusted', cert=False)
        #         return control
                    
        
        # breakpoint()
        # NOTE: target following -----    
        self.nominal_pixel_is_certifieds, self._ = self.alg1_solver.run((self.v_e,self.w_e,control_acc, is_animated, is_wp, is_obstacle, d_upper, d_lower))
        # self.nominal_pixel_is_certifieds = np.logical_or(1-is_wp, self.nominal_pixel_is_certifieds) # check only at wp
        # H_ = self.nominal_pixel_is_certifieds.shape[0]
        # threshold = int(0.79* H_)
        # self.nominal_pixel_is_certifieds[:threshold,:]=1    
        
        
        # NOTE: visualize target following ----
        self.resize_visualize(self.nominal_pixel_is_certifieds, 'targets not follow')
        
        
        # # return control
        # print(f"{self._step}, v:{self.v_e:.2f}, a:{self.a_e:.2f}, w:{self.w_e:.2f}, alpha:{self.alpha_e:.2f}, "
        #       f"si:{control_signal:.2f}, th:{throttle:.2f}, " 
        #       f"brake:{brake:.2f}, cert:{self.nominal_pixel_is_certifieds.sum()/self.nominal_pixel_is_certifieds.size:.4f}")
        
        if self.nominal_pixel_is_certifieds.sum() == self.nominal_pixel_is_certifieds.size: #> self.certify_threshold:
            delta = np.clip(math.sqrt(max(0,self.v_e)) * self.c_speed_sqrt + self.a_e * self.c_acc + self.w_e**2 * self.c_w_sq + abs(self.w_e) * self.c_w, 0.0, 0.25)
            _ = self.throttle_controller.step(delta)
            
            post_followtime = time.time()

            # print(f"step: {self.step}, pre follow time: {prefollow_time-start_time:.3f}, "
            #     f"postfollow time: {time.time() - prefollow_time:.3f}")
        
        else:
            nominal_control = (self.a_e, self.w_e )
            # print("nominal control action: {:.3%}, {:.3%}".format(control_acc, control_steering_rate))
            a_e = self.a_e
            alpha_e = self.w_e
            
            def certify_control_action(control_action):
                a_e, w_e = control_action
                # print("certifying:", self.a_e, self.alpha_e)
                pixel_is_certifieds, _ = self.alg1_solver.run((self.v_e, w_e, a_e, is_animated, is_wp, is_obstacle, d_upper, d_lower))


                # if (self.v_e > -0.001 and self.v_e < 0.001):
                #     print("certified due to being stationary")
                #     return True

                # the pixel is automatically certified if it is 
                # the road (in other words, not obstacle)
                # pixel_is_certifieds = np.logical_or(1-is_wp, pixel_is_certifieds)
                
                # limit focus region
                H_ = pixel_is_certifieds.shape[0]
                threshold = int(0.79* H_)
                pixel_is_certifieds[:threshold,:]=1
                
                # return pixel_is_certifieds.sum() / pixel_is_certifieds.size > self.certify_threshold
                return pixel_is_certifieds.sum() / pixel_is_certifieds.size
            
            res = self.alg2_solver.run(nominal_control, certify_control_action)
            if res is not None:
                # (a_control, alpha_control) = res     
                (a_control, w_control) = res
                
                # steer mapping 2
                
                desire_w_e = w_control # gain
                desire_rho = desire_w_e / (abs(self.v_e) + self.config.c_rs) # appear when w happen without v_e
                self.rho_e_deque.append(desire_rho)
                steer = self.config.m_rs * desire_rho
                
                # steer = self.turn_controller.step(steer) # finally we want to have steer ~ 0
                steer = np.clip(steer, -1.0, 1.0) # 1.3 is some extra gain

                print(f'follow:{self._step}, w_e:{self.w_e:.3e}, old/new_w:{self.w_e:.3f}/{w_control:.3f}, old/new_steer: {control.steer:.2f}/{steer:.3f}')
                print(f"step: {self.step}, pre follow time: {prefollow_time-start_time:.3f}, "
                      f"postfollow time: {time.time() - prefollow_time:.3f}")
                
                control.steer =  steer
                # update the target following plot
                self.nominal_pixel_is_certifieds, self.raw_data = self.alg1_solver.run((self.v_e, w_control, a_control, is_animated, is_wp, is_obstacle, d_upper, d_lower))
                """
                breakpoint()
                self.nominal_pixel_is_certifieds, self.raw_data = self.alg1_solver_follow.run((self.mu,self.nu,self.v_e,self.w_e,a_control,alpha_control, is_animated, is_wp, d_upper, d_lower))
                
                # self.nominal_pixel_is_certifieds, self._ = self.alg1_solver_follow.run((self.mu,self.nu,self.v_e,w_e,interfuser_acc,interfuser_steering_rate))
                self.nominal_pixel_is_certifieds = np.logical_or(1-is_wp, self.nominal_pixel_is_certifieds) # remove road pixels
                H_ = self.nominal_pixel_is_certifieds.shape[0]
                threshold = int(0.79* H_)
                self.nominal_pixel_is_certifieds[:threshold,:]=1
                
                # NOTE: visualize raw_data ----
                # Create a 3x2 grid of subplots
                fig, axes = plt.subplots(4, 2, figsize=(10, 12))

                # First subplot (0,0): mu_b
                data_mu_b = self.raw_data[:, :, 2]
                im1 = axes[0, 0].imshow(data_mu_b, cmap='viridis', vmin=np.min(data_mu_b), vmax=np.max(data_mu_b))
                axes[0, 0].set_title('mu_b')
                fig.colorbar(im1, ax=axes[0, 0])

                # Second subplot (0,1): nu_b
                data_nu_b = self.raw_data[:, :, 3]
                im2 = axes[0, 1].imshow(data_nu_b, cmap='viridis', vmin=np.min(data_nu_b), vmax=np.max(data_nu_b))
                axes[0, 1].set_title('nu_b')
                fig.colorbar(im2, ax=axes[0, 1])

                # Third subplot (1,0): mu_i
                data_mu_i = self.raw_data[:, :, 4]
                im3 = axes[1, 0].imshow(data_mu_i, cmap='viridis', vmin=np.min(data_mu_i), vmax=np.max(data_mu_i))
                axes[1, 0].set_title('mu_i')
                fig.colorbar(im3, ax=axes[1, 0])

                # Fourth subplot (1,1): nu_i
                data_nu_i = self.raw_data[:, :, 5]
                im4 = axes[1, 1].imshow(data_nu_i, cmap='viridis', vmin=np.min(data_nu_i), vmax=np.max(data_nu_i))
                axes[1, 1].set_title('nu_i')
                fig.colorbar(im4, ax=axes[1, 1])

                # Fifth subplot (2,0): mu_dot (same data as mu_i)
                data_mu_dot = self.raw_data[:, :, 6]
                im5 = axes[2, 0].imshow(data_mu_dot, cmap='viridis', vmin=np.min(data_mu_dot), vmax=np.max(data_mu_dot))
                axes[2, 0].set_title('mu_dot')
                fig.colorbar(im5, ax=axes[2, 0])

                # Sixth subplot (2,1): nu_dot (same data as nu_i)
                data_nu_dot = self.raw_data[:, :, 7]
                im6 = axes[2, 1].imshow(data_nu_dot, cmap='viridis', vmin=np.min(data_nu_dot), vmax=np.max(data_nu_dot))
                axes[2, 1].set_title('nu_dot')
                fig.colorbar(im6, ax=axes[2, 1])
                
                # Seventh subplot (3,0): certify
                data_certify = self.nominal_pixel_is_certifieds
                im7 = axes[3, 0].imshow(data_certify, cmap='viridis', vmin=np.min(is_wp), vmax=np.max(is_wp))
                axes[3, 0].set_title('certify')
                fig.colorbar(im7, ax=axes[3, 0])
                
                # Seventh subplot (3,1): wp
                im8 = axes[3, 1].imshow(is_wp, cmap='viridis', vmin=np.min(is_wp), vmax=np.max(is_wp))
                axes[3, 1].set_title('is_wp')
                fig.colorbar(im8, ax=axes[3, 1])

                # Adjust layout for better spacing
                plt.tight_layout()

                # Show the plot
                plt.show()
                # ------
                """
                
                # NOTE: visualize target following
                # self.nominal_pixel_is_certifieds = np.logical_or(1-is_wp, self.nominal_pixel_is_certifieds) # remove road pixels
                H_ = self.nominal_pixel_is_certifieds.shape[0]
                threshold = int(0.79* H_)
                self.nominal_pixel_is_certifieds[:threshold,:]=1
                self.resize_visualize(self.nominal_pixel_is_certifieds, 'targets not follow, adjusted', cert=False)
            
                """
                NOTE: debug attempt
                if self._step == 60:
                    breakpoint()
            
                    # Create a 2x2 subplot grid
                    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

                    # Plot mu
                    axes[0, 0].set_title('self.mu', fontsize=14)
                    c1 = axes[0, 0].imshow(self.mu, cmap='viridis', aspect='auto')
                    axes[0, 0].set_title('self.mu')
                    fig.colorbar(c1, ax=axes[0, 0])

                    # Plot nu
                    axes[0, 1].set_title('self.nu', fontsize=14)
                    c2 = axes[0, 1].imshow(self.nu, cmap='plasma', aspect='auto')
                    axes[0, 1].set_title('self.nu')
                    fig.colorbar(c2, ax=axes[0, 1])

                    # Plot self.X
                    axes[1, 0].set_title('self.X', fontsize=14)
                    c3 = axes[1, 0].imshow(self.X, cmap='inferno', aspect='auto')
                    axes[1, 0].set_title('self.X')
                    fig.colorbar(c3, ax=axes[1, 0])

                    # Plot self.Y
                    axes[1, 1].set_title('self.Y', fontsize=14)
                    c4 = axes[1, 1].imshow(self.Y, cmap='magma', aspect='auto')
                    axes[1, 1].set_title('self.Y')
                    fig.colorbar(c4, ax=axes[1, 1])

                    # Adjust layout for better spacing
                    plt.tight_layout()

                    # Show the plot
                    plt.show()
                    breakpoint()
                """   
                cv2.waitKey(1)
                return control # --> return obj avoidance control
        
        # cv2.waitKey(1)
        return control
    

    def destroy(self):
        super().destroy()
        
    def _create_XY(self, camera_x_idx_to_x, camera_y_idx_to_y):

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
    
    def _set_certificate_bound(self, x_deg_close, x_deg_far, y_deg, decay_temp, offset_scale):
        # relate to SC boundary
        self.meters_per_pixel_x = self.camera_size_x / self.camera_width
        self.meters_per_pixel_y = self.camera_size_y / self.camera_height
        
        self.optical_flow = optical_flow(self.camera_height, self.camera_width, self.meters_per_pixel_x, self.meters_per_pixel_y)
        
        # https://github.com/carla-simulator/carla/issues/56
        # assume fov is measured horizontally
        self.focal_len = self.camera_size_x / 2 / np.tan(np.deg2rad(self.fov / 2))
        print(f'focal length: {self.focal_len:.3f}')
        X, Y = self._create_XY(self.meters_per_pixel_x, self.meters_per_pixel_y)
        self.X = X
        self.Y = Y

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
                    self.certification_offset[i,j] = (temp ** decay_temp)
                elif -(j-x_mid-2*offset_w) > (i-y_mid)*x_y_slope and (j-x_mid+2*offset_w) > (i-y_mid)*x_y_slope:  # in upper \_/ region
                    # transform distance in _ to be equal to distance in / \
                    temp = np.absolute(i-y_mid-y_deg*self.ppd) # * 2 * sin_theta 
                    self.certification_offset[i,j] = -(temp ** decay_temp) * 2
                elif -(j-x_mid) < (i-y_mid)*x_y_slope: # /
                    temp = np.absolute(x_y_slope*i - j- x_y_slope*y_mid + x_mid)/np.sqrt(x_y_slope**2+1)
                    self.certification_offset[i,j] = -(temp ** decay_temp) /2 / sin_theta *2
                else:
                    temp = np.absolute(x_y_slope*i + j- x_y_slope*y_mid - x_mid)/np.sqrt(x_y_slope**2+1)
                    self.certification_offset[i,j] = -(temp** decay_temp) /2 / sin_theta *2
        self.certification_offset = offset_scale * self.certification_offset
        self.no_certification_required = np.full((self.camera_height, self.camera_width), False)

        is_sky = np.full((self.camera_height, self.camera_width), False)
        is_sky[0:self.camera_height // 2, :] = True
        self.no_certification_required = np.logical_or(is_sky, self.no_certification_required)
    
    def resize_visualize(self, img, img_name, scale=0.5, binary_input=True, cert=True):
        step_text = f'Step: {self._step}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 30)
        font_scale = 1
        thickness =2
        
        new_width = int(img.shape[1] * scale)
        new_height = int(img.shape[0] * scale)
        
        if binary_input:
            img = np.repeat((1-img[:, :, np.newaxis].astype(np.uint8)) * 255, 3, axis=2)
            
        new_size = (new_width, new_height)
        
        # Resize the image to the new size
        if scale != 1:
            resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        else:
            resized_img = img
        
        if cert:
            color = (100, 200, 100)
        else:
            color = (100, 100, 200)
        cv2.putText(resized_img, step_text, position, font, font_scale, color, thickness)
        cv2.imshow(img_name, resized_img )
        # mapping steer

    def _calculate_steer(self, speed, imu_omega_z):
        # coefficient from CARLA get physics
        COM_length = 1.75
        wheel_length = 2.9
        if abs(imu_omega_z) < 10e-2 or speed < 0.1: # less sensitive to noisy data
            steer = 0
            return steer
        else:
            R = np.maximum(speed / abs(imu_omega_z), self.turning_radius)
        print(f'R: {R}, speed: {speed}, imu_omega_z: {imu_omega_z}')
        steer = math.atan(math.sqrt(wheel_length * wheel_length / (R * R - COM_length * COM_length))) * np.sign(imu_omega_z)
        np.clip(steer,-1,1)
        return steer
    
    def _distort(self, label, value=5, n_labels=24):
        """
        Fast version of the distort function using NumPy's vectorized operations.

        Parameters:
        - label: numpy array of shape [H, W], the segmentation label.
        - value: percentage of pixels to distort.
        - n_labels: the maximum label value (inclusive).

        Returns:
        - Distorted label array.
        """
        H, W = label.shape
        num_pixels = H * W
        num_distort = int((value / 100.0) * num_pixels)

        # Generate random locations to distort
        indices = np.random.choice(num_pixels, num_distort, replace=False)
        
        # Convert 1D indices to 2D coordinates
        ys, xs = np.unravel_index(indices, (H, W))

        # Vectorized label selection
        original_labels = label[ys, xs]

        # Generate new labels
        new_labels = np.random.randint(0, n_labels, size=num_distort)

        # Ensure new labels are different (add 1 and mod with n_labels+1 to avoid original label)
        new_labels = (original_labels + new_labels + 1) % (n_labels + 1)

        # Update the labels
        distorted_label = label.copy()
        distorted_label[ys, xs] = new_labels

        return distorted_label
        
class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative
