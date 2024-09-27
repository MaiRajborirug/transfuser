import numpy as np
import pandas as pd
import imageio
import random
import os
import cv2

from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm

# Printing model summary is not as straightforward in PyTorch as it is in Keras.
# You may use additional libraries like torchsummary for this purpose.
from torchsummary import summary

import matplotlib.pyplot as plt
from patchify import patchify, unpatchify


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# ------------------
# Model Architecture
class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_probability=0, max_pooling=True):
        super(EncodingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.max_pooling = max_pooling
        if self.max_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        c = self.conv1(x)
        c = self.bn1(c)
        c = self.relu(c)
        c = self.conv2(c)
        c = self.bn2(c)
        c = self.relu(c)
        if self.max_pooling:
            x = self.pool(c)
        else:
            x = c
        return x, c

class DecodingBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DecodingBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(mid_channels + out_channels, mid_channels, kernel_size=3, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, filters, n_classes):
        super(UNet, self).__init__()
        self.enc1 = EncodingBlock(in_channels, filters, max_pooling=True)
        self.enc2 = EncodingBlock(filters, filters * 2, max_pooling=True)
        self.enc3 = EncodingBlock(filters * 2, filters * 4, max_pooling=True)
        self.enc4 = EncodingBlock(filters * 4, filters * 8, max_pooling=True)
        self.center = EncodingBlock(filters * 8, filters * 16, max_pooling=False)
        self.dec4 = DecodingBlock(filters * 16, filters * 8, filters * 8)
        self.dec3 = DecodingBlock(filters * 8, filters * 4, filters * 4)
        self.dec2 = DecodingBlock(filters * 4, filters * 2, filters * 2)
        self.dec1 = DecodingBlock(filters * 2, filters, filters)
        self.final_conv = nn.Conv2d(filters, n_classes, kernel_size=1)

    def forward(self, x):
        c1, skip1 = self.enc1(x)
        c2, skip2 = self.enc2(c1)
        c3, skip3 = self.enc3(c2)
        c4, skip4 = self.enc4(c3)
        center, _ = self.center(c4)
        d4 = self.dec4(center, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)
        out = self.final_conv(d1)
        return out

class Unetpad(nn.Module):
    """Unet model with padding to make image dimensions divisible by 128
    input array [H, W, 3]
    output array [H, W]"""
    def __init__(self, model='unet_model', model_state=None, patch_size=256, overlap=0):
        # model_state: path to model state dict
        super(Unetpad, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model == 'unet_model':
            self.model = UNet(in_channels=3, filters=32, n_classes=29)
            self.model.to(self.device)
        if model_state is not None:
            self.model.load_state_dict(torch.load(model_state))
            self.model.eval()
        self.patch_size = patch_size
        self.overlap = overlap
        self.step = self.patch_size - self.overlap 
    # def __init__(self, model='unet_model', model_state=None):
    #     # model_state: path to model state dict
    #     super(Unetpad, self).__init__()
    #     print('im here bois')
    #     self.model = UNet(in_channels=3, filters=32, n_classes=29)

    def forward(self, img_np):
        # Input as array [H, W, 3]
        self.h, self.w, _ = img_np.shape
        # patchify
        pad_height = (self.patch_size - self.h % self.patch_size) % self.patch_size
        pad_width = (self.patch_size - self.w % self.patch_size) % self.patch_size
        
        padded_img = np.pad(img_np, ((pad_height, 0), # pad at the top because it is less sensitive region
                            (pad_width // 2, pad_width - pad_width // 2), 
                            (0, 0)), 
                    mode='constant', constant_values=0)
        patch_size = (self.patch_size, self.patch_size, 3)
        padded_img_size = padded_img.shape
        patches = patchify(padded_img, patch_size, step=self.step) # (5, 10, 1, 128, 128, 3)
        # reverse_patches = unpatchify(patches, padded_img_size)
        # print('reverse', reverse_patches.shape)
        num_height, num_weight, _, _, _, _ = patches.shape # (4, 8, 1, 128, 128, 3) <class 'numpy.ndarray'>
        patches = patches.reshape(-1, *patch_size) # (50, 128, 128, 3)
        image_patches_tensor = torch.tensor(patches.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0 # (N, 3, 128, 128)
        
        with torch.no_grad():
            image_patches_tensor = image_patches_tensor.to(self.device).float()
            pred_mask = self.model(image_patches_tensor)
            pred_mask = torch.argmax(pred_mask, dim=1).cpu().numpy() # (N, 128, 128)
            pred_mask = pred_mask.reshape(num_height, num_weight, self.patch_size, self.patch_size) #(N_h, N_w, 128, 128)
            
            full_pred_mask = unpatchify(pred_mask, (self.h + pad_height, self.w + pad_width))
            # output = unpatchify(pred_mask, padded_img.shape)
            # print('output', output.shape)
            # return output
            full_pred_mask = full_pred_mask[pad_height:, pad_width // 2: pad_width // 2 + self.w]
            
        return full_pred_mask

    def _forward(self, img_np):
        # Input as array [H, W, 3]
        self.h, self.w, _ = img_np.shape
        # patchify
        pad_height = (self.patch_size - self.h % self.patch_size) % self.patch_size
        pad_width = (self.patch_size - self.w % self.patch_size) % self.patch_size
        
        padded_img = np.pad(img_np, ((pad_height, 0), # pad at the top because it is less sensitive region
                            (pad_width // 2, pad_width - pad_width // 2), 
                            (0, 0)), 
                    mode='constant', constant_values=0)
        patch_size = (self.patch_size, self.patch_size, 3)
        padded_img_size = padded_img.shape
        patches = patchify(padded_img, patch_size, step=self.step) # (5, 10, 1, 128, 128, 3)
        # reverse_patches = unpatchify(patches, padded_img_size)
        # print('reverse', reverse_patches.shape)
        num_height, num_weight, _, _, _, _ = patches.shape # (4, 8, 1, 128, 128, 3) <class 'numpy.ndarray'>
        patches = patches.reshape(-1, *patch_size) # (50, 128, 128, 3)
        image_patches_tensor = torch.tensor(patches.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0 # (N, 3, 128, 128)
        
        with torch.no_grad():
            image_patches_tensor = image_patches_tensor.to(self.device).float()
            
            pred_mask = image_patches_tensor[:, 0, :, :].cpu().numpy()
            
            # pred_mask = self.model(image_patches_tensor)
            # pred_mask = torch.argmax(pred_mask, dim=1).cpu().numpy() # (N, 128, 128)
            
            pred_mask = pred_mask.reshape(num_height, num_weight, self.patch_size, self.patch_size) #(N_h, N_w, 128, 128)
            full_pred_mask = unpatchify(pred_mask, (self.h + pad_height, self.w + pad_width))
            # output = unpatchify(pred_mask, padded_img.shape)
            # print('output', output.shape)
            # return output
            full_pred_mask = full_pred_mask[pad_height:, pad_width // 2: pad_width // 2 + self.w]
            
        return full_pred_mask
        
def group_segment(label):
    """
    input: label 2D segmetnation label
    outputs: 4 binary masks for road, non-animated, terrain, and animated objects
    """
    colormap_24 = {
        0: (0, 0, 0),        # unlabeled
        1: (70, 70, 70),     # building
        2: (100, 40, 40),    # fence
        3: (55, 90, 80),     # other
        4: (220, 20, 60),    # pedestrian
        5: (153, 153, 153),  # pole
        6: (157, 234, 50),   # road line
        7: (128, 64, 128),   # road
        8: (244, 35, 232),   # sidewalk
        9: (107, 142, 35),   # vegetation
        10: (0, 0, 142),     # vehicle
        11: (102, 102, 156), # wall
        12: (220, 220, 0),   # traffic sign
        13: (70, 130, 180),  # sky
        14: (81, 0, 81),     # ground
        15: (150, 100, 100), # bridge
        16: (230, 150, 140), # rail track
        17: (180, 165, 180), # guard rail
        18: (250, 170, 30),  # traffic light green
        19: (110, 190, 160), # static
        20: (170, 120, 50),  # dynamic
        21: (45, 60, 150),   # water
        22: (145, 170, 100), # terrain
        23: (255, 0, 0),     # traffic light yellow
        24: (0, 255, 0),     # traffic light red
    }
    
    # Define colors for each group for visualization
    group_colors = {
        'road': (128, 64, 128),
        'animated': (220, 20, 60),
        'terrain': (107, 142, 35),
        'non_animated': (70, 70, 70),
    }
    
    # Define class indices for each category
    # NOTE: for experiment use switch animate and non animated
    road_classes = [0, 6, 7, 13, 14, 15, 16, 21, 22,] # Date: sept 25 add '0' (class of WP)
    animated_classes = [4, 10]
    nonanimated_classes = [1, 2, 5,8, 9, 11, 12, 17, 18, 19]
    terrain_classes = [9]
    # nonanimated_classes = [4, 10]
    # animated_classes = [1, 2, 5,8, 9, 11, 12, 17, 18, 19]
    
    # Create binary masks for each category
    is_road = np.isin(label, list(road_classes)).astype(int)
    is_animated = np.isin(label, list(animated_classes)).astype(int)
    is_terrain = np.isin(label, list(terrain_classes)).astype(int)
    is_nonanimated = np.isin(label, list(nonanimated_classes)).astype(int)
    
    # Create a color group image initialized to all zeros (black)
    H, W = label.shape
    group_label = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Fill the group image with colors according to the category
    for category, color in group_colors.items():
        if category == 'road':
            group_label[is_road == 1] = color
        elif category == 'animated':
            group_label[is_animated == 1] = color
        elif category == 'terrain':
            group_label[is_terrain == 1] = color
        elif category == 'non_animated':
            group_label[is_nonanimated == 1] = color

    
    return is_road, is_nonanimated, is_terrain, is_animated, group_label
        
# NOTE: test model2
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# segment_state_path = '/media/haoming/970EVO/Amar/trained-model/carla-image-segmentation-model-datagen3-state-dict.pth'
# # segment_state_path = '/media/haoming/970EVO/Pharuj/git/DOS/testing_codes/carla-image-segmentation-model-resized-state-dict.pth'
# rgb_file = '/media/haoming/970EVO/Pharuj/transfuser_datagen/Town01_Scenario1/Town01_Scenario1_route0_03_20_13_28_33/rgb_front/0000.png'

# segment_model = UNet(in_channels=3, filters=32, n_classes=29)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# segment_model  = segment_model.to(device)
# # segment_state_path = '/media/haoming/970EVO/Pharuj/git/DOS/testing_codes/carla-image-segmentation-model-state-dict.pth'
# segment_model.load_state_dict(torch.load(segment_state_path))
# transform = transforms.Compose([
#             transforms.Resize((256, 256)),  # Resizes to 256x256. Change this line if aspect ratio needs to be preserved
#             transforms.ToTensor(),  # Converts to a tensor and scales to [0, 1]
#             # already convert from (H, W, 3) to (3, 256, 256)
#         ])

# rgb = np.array(Image.open(rgb_file))
# rgb_pil = Image.fromarray(rgb.astype('uint8'), 'RGB')
# rgb_pil = transform(rgb_pil)
# rgb_pil = rgb_pil.unsqueeze(0)  # Add batch dimension # (3, 256, 256) -> (1, 3, 256, 256)
# pred_mask = segment_model(rgb_pil.to(device))  # Pass the image to the model
# # print('rgb_shape', rgb.shape)
# pred_mask = torch.argmax(pred_mask, dim=1)[0].cpu().numpy()  # Assuming channel is at dim=1
# # interpolate mask
# pred_mask = cv2.resize(pred_mask, (960, 480), interpolation=cv2.INTER_NEAREST)
# im = Image.fromarray(pred_mask.astype(np.uint8))
# im.save('seg_resize.png')
# is_road, is_nonanimated, is_terrain, is_animated = group_segment(pred_mask)
# im = Image.fromarray((is_road*100).astype(np.uint8))
# im.save('seg_resize_is_road.png')


# segmentation2_model = Unetpad(model_state=segment_state_path, patch_size=128)
# pred_mask = segmentation2_model(rgb)
# print(type(pred_mask), pred_mask.shape)

# breakpoint()
# im = Image.fromarray(pred_mask)
# im = Image.fromarray(np.uint8(pred_mask*255), 'L')
# im.save('/media/haoming/970EVO/Pharuj/git/DOS/test_station/_test_patchify.png')

# is_road, is_nonanimated, is_terrain, is_animated, group_label = group_segment(pred_mask)
# # breakpoint()
# im = Image.fromarray(np.uint8(group_label))
# im.save('/media/haoming/970EVO/Pharuj/git/DOS/test_station/_test_patchify_label.png')