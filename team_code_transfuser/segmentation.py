import torch
import urllib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2 as cv
import os

import sys
sys.path.append('/home/haoming/git') # where folder DeepLab
from DeepLab.datasets import Cityscapes, cityscapes
from DeepLab import network

from memory_profiler import profile

class segmentation:
    def __init__(self):
        self.model = network.modeling.__dict__["deeplabv3plus_resnet101"](num_classes=19, output_stride=16)
        # pretrain weight
        self.model.load_state_dict(torch.load("/home/haoming/perception_based_control_alt/best_deeplabv3plus_resnet101_cityscapes_os16.pth")['model_state']  )
        # /home/haoming/git/DeepLab/best_deeplabv3plus_mobilenet_cityscapes_os16.pth
        self.model.eval()
        # for viz
        # create a color pallette, selecting a color for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.colors = torch.as_tensor([i for i in range(5)])[:, None] * self.palette
        self.colors = (self.colors % 255).numpy().astype("uint8")
        self.save_path = "/media/haoming/720A-A5AA/Transfuser_Data/seg_out"
        self.cnt = 0
        self.decode_fn = Cityscapes.decode_target

        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if torch.cuda.is_available():
            self.model.to('cuda')
                # 24 class in CARLA semantic segmentaion
        # color from https://carla.readthedocs.io/en/0.9.12/tuto_D_create_semantic_tags/
        self.colormap_24 =  {
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
            18: (250, 170, 30),  # traffic light
            19: (110, 190, 160), # static
            20: (170, 120, 50),  # dynamic
            21: (45, 60, 150),   # water
            22: (145, 170, 100), # terrain
            23: (255, 0, 0),     # MyNewTag
        }
        # Define a colormap for broader classes
        self.broader_colormap = {
            'road': (128, 64, 128),
            'nonanimated': (70, 70, 70),
            'terrain': (145, 170, 100),
            'animated': (220, 20, 60),
        }


    def unflatten(self, raw):
        # height, width, channels
        return raw.reshape((500, 1000, 4))
    
    def save_img(self, img):
        img.save(self.save_path + str(self.cnt) + ".jpg")
    
    #@profile
    def predict(self, frame):
        """
        frame: numpy ndarray with rgb colors.
        Dimension:
        (height, width, 3)

        returns: numpy ndarray with predicted class (uint) 
        Dimension:
        (height, width)
        """
        
        cv.imshow("input:", cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        if cv.waitKey(1) == 27:
            return

        self.cnt += 1
        self.frame = frame.astype(np.float32)
	
        # convert values from from 0-255 to 0-1
        self.frame = self.frame / 256.0
        # convert from [H W 3] shape to [3 H W] shape
        self.frame = np.swapaxes(self.frame, 0, 2)
        self.frame = np.swapaxes(self.frame, 1, 2)

        self.input_tensor = torch.from_numpy(self.frame)
        #input_tensor = self.preprocess(input_tensor)

        # the model expects a mini batch of shape [N, 3, H, W]
        # where N is number of image.
        # create a mini batch by add a dimension.
        self.input_batch = self.input_tensor.unsqueeze(0)
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            self.input_batch = self.input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            self.output = self.model(self.input_batch).max(1)[1].cpu().numpy()[0] # HW axis order
        
        # plot the semantic segmentation predictions into cityscapes colors for viz
        cv.imshow("segmentation:", (self.decode_fn(self.output) / 255))
        

        # train id of road = 0 
        # return np boolean True = is road
        return np.logical_or(np.equal(self.output, np.zeros(self.output.shape, dtype=np.int64)),np.equal(self.output, np.ones(self.output.shape, dtype=np.int64)))
    
    # new function
    def mask_seg(self, output):
        """
        (  'road'                 ,0 , 'ground'          , 0       ,(128, 64,128) ),
        (  'sidewalk'             ,1 , 'ground'          , 0       ,(244, 35,232) ),
        (  'building'             ,2 , 'construction'    , 1       ,( 70, 70, 70) ),
        (  'wall'                 ,3 , 'construction'    , 1       ,(102,102,156) ),
        (  'fence'                ,4 , 'construction'    , 1       ,(190,153,153) ),
        (  'pole'                 ,5 , 'object'          , 1       ,(153,153,153) ),
        (  'traffic light'        ,6 , 'object'          , 1       ,(250,170, 30) ),
        (  'traffic sign'         ,7 , 'object'          , 1       ,(220,220,  0) ),
        (  'vegetation'           ,8 , 'nature'          , 4       ,(107,142, 35) ),
        (  'terrain'              ,9 , 'nature'          , 4       ,(152,251,152) ),
        (  'sky'                  ,10 , 'sky'             , 5       ,( 70,130,180) ),
        (  'person'               ,11 , 'human'           , 6       ,(220, 20, 60) ),
        (  'rider'                ,12 , 'human'           , 6       ,(255,  0,  0) ),
        (  'car'                  ,13 , 'vehicle'         , 7       , (  0,  0,142) ),
        (  'truck'                ,14 , 'vehicle'         , 7       , (  0,  0, 70) ),
        (  'bus'                  ,15 , 'vehicle'         , 7       , (  0, 60,100) ),
        (  'train'                ,16 , 'vehicle'         , 7       ,(  0, 80,100) ),
        (  'motorcycle'           ,17 , 'vehicle'         , 7       ,(  0,  0,230) ),
        """
    
        # Initialize binary masks with zeros
        is_road = np.zeros(output.shape, dtype=int) # [batch, H, W]
        is_nonanimated = np.zeros(output.shape, dtype=int)
        is_terrain = np.zeros(output.shape, dtype=int)
        is_animated = np.zeros(output.shape, dtype=int)

        # Set elements to 1 based on your criteria
        is_road[np.isin(output, [0, 1, 10])] = 1
        is_nonanimated[np.isin(output, [2, 3, 4, 5, 6, 7, 8])] = 1
        is_terrain[np.isin(output, [9])] = 1
        is_animated[np.isin(output, [11, 12, 13, 14, 15, 16, 17, 18])] = 1

        return is_road, is_nonanimated, is_terrain, is_animated  # output as np.bool
    
    #@profile
    def predict_(self, frame):
        """
        frame: numpy ndarray with rgb colors.
        Dimension:
        (height, width, 3)

        returns: numpy ndarray with predicted class (uint) 
        Dimension:
        (height, width)
        """
        
        # cv.imshow("input:", cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        # if cv.waitKey(1) == 27: # without delay, can't see the image
        #     return
        cv.waitKey(1)
        
        self.cnt += 1
        self.frame = frame.astype(np.float32)
	
        # convert values from from 0-255 to 0-1
        self.frame = self.frame / 256.0
        # convert from [H W 3] shape to [3 H W] shape
        self.frame = np.swapaxes(self.frame, 0, 2)
        self.frame = np.swapaxes(self.frame, 1, 2)

        self.input_tensor = torch.from_numpy(self.frame)
        #input_tensor = self.preprocess(input_tensor)

        # the model expects a mini batch of shape [N, 3, H, W]
        # where N is number of image.
        # create a mini batch by add a dimension.
        self.input_batch = self.input_tensor.unsqueeze(0)
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            self.input_batch = self.input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            self.output = self.model(self.input_batch).max(1)[1].cpu().numpy()[0] # HW axis order
        
        # # plot the semantic segmentation predictions into cityscapes colors for viz
        # cv.imshow("segmentation:", (self.decode_fn(self.output) / 255))
        
        is_road, is_nonanimated, is_terrain, is_animated = self.mask_seg(self.output)
        # return np.logical_or(np.equal(self.output, np.zeros(self.output.shape, dtype=np.int64)),np.equal(self.output, np.ones(self.output.shape, dtype=np.int64)))
        return is_road, is_nonanimated, is_terrain, is_animated
    
    #@profile
    def predict_batch(self, input_batch):
        """
        frame: numpy ndarray with rgb colors.
        Dimension:
        (batch, 3, height, width)

        # returns: numpy ndarray with predicted class (uint) 
        # Dimension:
        # (height, width)
        """

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            self.input_batch = (input_batch/256.0).to('cuda')

        with torch.no_grad():
            self.output = self.model(self.input_batch).max(1)[1].cpu().numpy()
            # torch.Size([batch, 19, H, W]) -> torch.Size([batch, H, W]), value represents class index
            #.max(1)[1].cpu().numpy()[0] # HW axis order

        show_img = False
        if show_img:
            # show only the first frame in the batch
            frame = input_batch[0].cpu().numpy()
            frame = np.transpose(frame, (1, 2, 0)).astype(np.uint8) # assign element types
            cv.imshow("input:", cv.cvtColor(frame, cv.COLOR_RGB2BGR))
            if cv.waitKey(1) == 5: # without delay, can't see the image
                return
            
            # plot the semantic segmentation predictions into cityscapes colors for viz
            cv.imshow("segmentation:", (self.decode_fn(self.output[0]) / 255))
        
        # is_road, is_nonanimated, is_terrain, is_animated = self.mask_seg(self.output)
        # # return np.logical_or(np.equal(self.output, np.zeros(self.output.shape, dtype=np.int64)),np.equal(self.output, np.ones(self.output.shape, dtype=np.int64)))
        # return is_road, is_nonanimated, is_terrain, is_animated
        is_road, is_nonanimated, is_terrain, is_animated = self.mask_seg(self.output)
        return is_road, is_nonanimated, is_terrain, is_animated


    #@profile
    def predict_gt_batch(self, labels, visualize=False):
        # Check if input is a single image or a batch and handle accordingly
        if len(labels.shape) == 2:  # Single image
            labels = np.expand_dims(labels, axis=0)
        
        # Prepare output arrays
        is_road = np.isin(labels, [6, 7, 13, 14, 15, 16])
        is_nonanimated = np.isin(labels, [1, 2, 3, 5, 8, 9, 11, 17, 18, 19, 20, 21, 23])
        is_terrain = np.isin(labels, [22])
        is_animated = np.isin(labels, [4, 10])
        
        outputs = [is_road, is_nonanimated, is_terrain, is_animated]
        
        if visualize:
            for i, batch_i in enumerate(labels):    
                # Detailed visualization
                detailed_lab = np.zeros((*batch_i.shape, 3), dtype=np.uint8) # ensure type is uint8
                for label, color in colormap_24.items():
                    detailed_lab[batch_i == label] = color                    
                detailed_lab_bgr = cv.cvtColor(detailed_lab, cv.COLOR_RGB2BGR)
            
                # # Save the image
                # cv.imwrite(f'/home/pharujra/Downloads/seg_240111_2/Segmentation_{i}.png', detailed_lab_bgr)
                
                # Broader class visualization
                broader_lab = np.zeros((*batch_i.shape, 3), dtype=np.uint8)
                broader_lab[is_road[i]] = broader_colormap['road']
                broader_lab[is_nonanimated[i]] = broader_colormap['nonanimated']
                broader_lab[is_terrain[i]] = broader_colormap['terrain']
                broader_lab[is_animated[i]] = broader_colormap['animated']
                
                # Convert from RGB to BGR for OpenCV
                broader_lab_bgr = cv.cvtColor(broader_lab, cv.COLOR_RGB2BGR)
                
                # Save the image
                # cv.imwrite(f'/home/pharujra/Downloads/seg_240111_2/Segmentation_broader_{i}.png', broader_lab_bgr)

            # show only last frame in the batch
            cv.imshow("detail segmentation:", detailed_lab_bgr/ 255)
            # cv.imshow("broader segmentation:", broader_lab_bgr/255)
            # if cv.waitKey(1) == 27:
            #     return
            cv.waitKey(0)

        return outputs
    
    def vis_gt(self, frame, gt_frame):
        # Check if input is a single image or a batch and handle accordingly
        labels = gt_frame[:, :, 2]
        # Create an empty image for the visualization
        vis_image = np.zeros((gt_frame.shape[0], gt_frame.shape[1], 3), dtype=np.uint8)
        
        # Map each label to its corresponding color
        for label, color in self.colormap_24.items():
            # Set the color for pixels with the current label
            vis_image[labels == label] = color
        
        # Display the resulting image
        cv.imshow("input:", cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        cv.imshow('segmentation gt', vis_image/255)
        cv.waitKey(1)  # Wait for a key press to close the window
        