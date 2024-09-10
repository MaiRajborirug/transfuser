

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

import torchvision.transforms as T
from leaderboard.autoagents import autonomous_agent
import math
import yaml

from optical_flow import optical_flow
from segmentation import segmentation
from rgb_seg import UNet, Unetpad, group_segment


from alg1_pycuda_copy import Algorithm1 #
from alg2 import Algorithm2
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
# NOTE: adjust submission_agent
from submission_agent_adjust import HybridAgent
from optical_flow import OpticalFlowVisualizer
from collections import deque
import torchvision.transforms as transforms

from memory_profiler import profile

# NOTE: for depth estimation
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
        self.ppd = self.camera_width / self.fov # pixel:degree # assume the same for x and y
        # define ourselves # original size
        self.camera_size_x = 0.096
        self.camera_size_y = 0.048
        
        # NOTE: for control mapping
        self.delta_time = 0.05 # CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
        self.throttle_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1, n=20) # throttle PID controller
        self.c_speed_sqrt = 0.032 # constant for throttle_control
        self.c_acc = 0.025
        self.c_w_sq = 0.04
        self.c_w = 0.01
        self._step = -1
        self.turning_radius = 2.5 # This is the minimum turning radius of the vehicle when steer = 1
        self.agent_front2back_whl = 2.9 # This property are from agent_vehicle.get_physics_control()
        self.agent_backwhl2cm = 1.75 # This property are from agent_vehicle.get_physics_control()

        # NOTE: for certificate bound
        self._set_certificate_bound(x_deg_close=55, x_deg_far=35, y_deg=12, decay_temp=1.19, offset_scale=0.004)
        self.certify_threshold = 0.987
        
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
        
        # NOTE: for depth estimation monodepth2
        self.d_std = 0.6
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

        # NOTE: test when turn-off safety certificate
        # self.no_certification_required = np.full((self.camera_height, self.camera_width), True)
        # self.certification_offset = np.full((self.camera_height, self.camera_width), -1000000)

        # NOTE: define safety certificate solver
        self.alg1_solver = Algorithm1(self.focal_len, (self.camera_width, self.camera_height), self.X, self.Y, self.certification_offset)
        
        self.alg2_solver = Algorithm2()
        self.last_w_e = None  # angular velocity in z-axis

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
        control = super().run_step(input_data, timestamp)
        # nominal_control = (control.throttle, control.steer)
        # print("nominal a: {}, nominal w_e: {}".format(control.throttle, control.steer))
        delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
        self._step += 1
        
        print("transfuser raw throttle: {:.2%}, brake: {}, steer: {:.3%}".format(control.throttle, control.brake, control.steer))
        
        # NOTE: optical flow output
        self.bgr_ = input_data["rgb_front"][1][:, :, :3]  # rgb to rgb_front
        self.optical_flow_output = self.optical_flow.predict(self.bgr_, self.delta_time) 
        
        
        # NOTE: segmentation output -> replace with GT segmentation
        # rgb_pil = Image.fromarray(cv2.cvtColor(self.bgr_.astype('uint8'), cv2.COLOR_BGR2RGB))
        # rgb_pil = self.transform(rgb_pil)
        # rgb_pil = rgb_pil.unsqueeze(0)  # Add batch dimension # (3, 256, 256) -> (1, 3, 256, 256)
        # pred_mask = self.segment_model(rgb_pil.to(self.device))  # Pass the image to the model
        # pred_mask = torch.argmax(pred_mask, dim=1)[0].cpu().numpy()  # Assuming channel is at dim=1
        # pred_mask = cv2.resize(pred_mask, (960, 480), interpolation=cv2.INTER_NEAREST)
        
        # NOTE: gt segmentation
        semantic_front = input_data["semantics_front"][1][:, :, 2] # semantic to semantic_front
        pred_mask = self._distort(semantic_front, value=5, n_labels=24)
        
        # NOTE: CARLA segmentation, initialize gt segmentation
        is_road, is_nonanimated, is_terrain, is_animated, group_label = group_segment(pred_mask)
        self.resize_visualize(group_label, 'segment label', binary_input=False)
        # cv2.waitKey(10)
        
        # NOTE: monodepth2 estimation
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
        arr2 = cv2.resize(arr, (480, 240), interpolation=cv2.INTER_AREA)
        cv2.imshow("distance", arr2)
        disp_resized = torch.nn.functional.interpolate(disp,
                                                       (original_height, original_width), mode="bilinear", align_corners=False)
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        distance = self.disp_k / disp_resized_np
        
        
        # NOTE: for certificate and control mapping
        v_e = np.maximum(0,input_data["speed"][1]["speed"]) # speedometer in vehicle direction (m/s)
        self.w_e = input_data['imu'][1][5] # angular vel in z-axis (rad/s)
        a_e = input_data["imu"][1][0] # accelaration in vehicle direction (m/s^2)
        control_acc = a_e
        
        if self._step == 0:
            self.phi_e = 0 # angular acceleration in z-axis (rad/s^2)
            self.last_w_e = self.w_e 
        else:
            self.phi_e = (self.w_e - self.last_w_e) / self.delta_time
            self.last_w_e = self.w_e
        
        if v_e < 0.2 and control.brake > 0:
            delta = np.clip(math.sqrt(v_e) * self.c_speed_sqrt + a_e * self.c_acc + self.w_e**2 * self.c_w_sq + abs(self.w_e) * self.c_w, 0.0, 0.25)
            _ = self.throttle_controller.step(delta)
            cv2.waitKey(1)
            return control
        
        # convert optical flow to mu and nu
        mu = self.optical_flow_output[:, :, 0]
        nu = self.optical_flow_output[:, :, 1]
        mu = mu.astype(np.float32)
        nu = nu.astype(np.float32)
                
        # find angular acceleration
        if abs(control.steer) > 0.05: # desensitize the value
            turning_radius = (self.agent_backwhl2cm**2 + self.agent_front2back_whl**2*abs(1/math.tan(control.steer)))
        else:
            turning_radius = np.inf

        control_steer_normalized = control.steer * v_e/turning_radius # desire angular velocity
        control_steering_rate = (control_steer_normalized - self.w_e) / self.delta_time # current angular accelaration in z-axis

        steering_limit = v_e/self.turning_radius
        self.alg2_solver.phi_bounds = ((-steering_limit-self.w_e)/self.delta_time,(steering_limit-self.w_e)/self.delta_time)        
        self.nominal_pixel_is_certifieds, self._ = self.alg1_solver.run((mu,nu,v_e,self.w_e,control_acc,control_steering_rate, is_animated))
        self.nominal_pixel_is_certifieds = np.logical_or(is_road, self.nominal_pixel_is_certifieds) # remove road pixels
        self.nominal_pixel_is_certifieds = np.logical_or(self.nominal_pixel_is_certifieds, self.no_certification_required) # remove sky pixels        
        self.nominal_pixel_is_certifieds1 = self.nominal_pixel_is_certifieds.copy()
        self.resize_visualize(self.nominal_pixel_is_certifieds, 'white = certified pixels')

        
        if self.nominal_pixel_is_certifieds.sum() / self.nominal_pixel_is_certifieds.size > self.certify_threshold:
            delta = np.clip(math.sqrt(v_e) * self.c_speed_sqrt + a_e * self.c_acc + self.w_e**2 * self.c_w_sq + abs(self.w_e) * self.c_w, 0.0, 0.25)
            _ = self.throttle_controller.step(delta)
            cv2.waitKey(1)
            return control
        
        nominal_control = (control_acc, control_steering_rate)
        print("nominal control action: {:.3%}, {:.3%}".format(control_acc, control_steering_rate))
        
        def certify_control_action(control_action):
            a_e, phi_e = control_action
            # print("certifying:", a_e, phi_e)
            pixel_is_certifieds, _ = self.alg1_solver.run((mu, nu, v_e, self.w_e, a_e, phi_e, is_animated))


            # if (v_e > -0.001 and v_e < 0.001):
            #     print("certified due to being stationary")
            #     return True

            # the pixel is automatically certified if it is 
            # the road (in other words, not obstacle)
            pixel_is_certifieds = np.logical_or(is_road, pixel_is_certifieds)

            # we only want to check certification for those regions we care about
            pixel_is_certifieds = np.logical_or(pixel_is_certifieds, self.no_certification_required)

            # print("is certified: ", pixel_is_certifieds.sum() / pixel_is_certifieds.size > self.certify_threshold)
            # cv_img2 = np.repeat(pixel_is_certifieds[:, :, np.newaxis].astype(np.uint8) * 255, 3, axis=2)
            # cv2.putText(cv_img2, step_text, position, font, font_scale, color, thickness)
            # cv2.imshow("certified, post mask", cv_img2)
        
            
            # return pixel_is_certifieds.sum() / pixel_is_certifieds.size > self.certify_threshold
            return pixel_is_certifieds.sum() / pixel_is_certifieds.size
    
        res = self.alg2_solver.run(nominal_control, certify_control_action)

        if res is None:
            cv2.waitKey(1)
            return control
        else:
            # (a_control, phi_control) = res     
            (a_control, phi_control) = res
            # print("certified control action: ", a_control, phi_control)

            # NOTE: map (a, phi) back to carla control (throttle, steering, break)
            if abs(a_control) < 2 * abs(phi_control) * self.turning_radius: # prioritize turning arbitary comparison
                # print('sc case 1')
                # throttle mapping
                a_control = control_acc # before SC 2              
                delta = np.clip(math.sqrt(v_e) * self.c_speed_sqrt + a_control * self.c_acc + self.w_e**2 * self.c_w_sq + abs(self.w_e) * self.c_w, 0.0, 0.25)
                throttle = self.throttle_controller.step(delta)
                throttle = np.clip(throttle+0.15, 0.0, 0.75)
                # throttle = np.maximum(control.throttle /2, 0.25)
                brake = 0
                
                # steer mapping
                desire_w = self.w_e + self.delta_time * phi_control
                # print('change in phi', self.delta_time * phi_control)
                steer = self._calculate_steer(v_e, desire_w)
            
            elif a_control > -0.2: # prioritize throttle
                # print('sc case 2')
                # throttle mapping           
                delta = np.clip(math.sqrt(v_e) * self.c_speed_sqrt + a_control * self.c_acc + self.w_e**2 * self.c_w_sq + abs(self.w_e) * self.c_w, 0.0, 0.25)
                throttle = self.throttle_controller.step(delta)
                throttle = np.clip(throttle, 0.0, 0.75)
                brake = 0
                
                # steer mapping
                desire_w = self.w_e + self.delta_time * phi_control
                # print('change in phi', self.delta_time * phi_control)
                steer = self._calculate_steer(v_e, desire_w)

            else:
                # print('sc case 3')
                throttle = 0
                brake = 1
                steer = 0

            control.steer = steer
            control.throttle = np.minimum(throttle, control.throttle)
            control.brake = np.maximum(brake, control.brake)
            print('new control: throttle: {:.2%}, brake: {}, steer: {:.3%}'.format(control.throttle, control.brake, control.steer))
            
            self.nominal_pixel_is_certifieds, self._ = self.alg1_solver.run((mu,nu,v_e,self.w_e,a_control,phi_control, is_animated))
            # self.nominal_pixel_is_certifieds, self._ = self.alg1_solver.run((mu,nu,v_e,psi_e,interfuser_acc,interfuser_steering_rate))
            self.nominal_pixel_is_certifieds = np.logical_or(is_road, self.nominal_pixel_is_certifieds) # remove road pixels
            self.nominal_pixel_is_certifieds = np.logical_or(self.nominal_pixel_is_certifieds, self.no_certification_required) # remove sky pixels
            # self.resize_visualize(pixel_is_certifieds , 'corrected certified pixels', cert=False)
            self.resize_visualize(self.nominal_pixel_is_certifieds, 'corrected certified pixels', cert=False)
            self.nominal_pixel_is_certifieds2 = self.nominal_pixel_is_certifieds.copy()

            newim = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            newim[(self.nominal_pixel_is_certifieds1 == True) & (self.nominal_pixel_is_certifieds2 == False)] = [0, 255, 0]
            newim[(self.nominal_pixel_is_certifieds1 == True) & (self.nominal_pixel_is_certifieds2 == True)] = [0, 0, 0]
            newim[(self.nominal_pixel_is_certifieds1 == False) & (self.nominal_pixel_is_certifieds2 == False)] = [255, 255, 255]
            newim[(self.nominal_pixel_is_certifieds1 == False) & (self.nominal_pixel_is_certifieds2 == True)] = [0, 0, 255]
            # cv2.imshow('after certification', newim)
            # self.resize_visualize(newim, 'aftercertification')
            cv2.waitKey(1)
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
        step_text = f'Step: {self.step}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 30)
        font_scale = 1
        thickness =2
        
        new_width = int(img.shape[1] * scale)
        new_height = int(img.shape[0] * scale)
        
        if binary_input:
            img = np.repeat(img[:, :, np.newaxis].astype(np.uint8) * 255, 3, axis=2)
            
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
