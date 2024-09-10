import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import torch
from torchvision import transforms

sys.path.append('/media/haoming/970EVO/Yaguang/depth_est/monodepth2')
import networks  
sys.path.append('/media/haoming/970EVO/Yaguang/depth_est/')
from monodepth2.utils import download_model_if_doesnt_exist


print('done importing')

model_name = "mono+stereo_1024x320"

download_model_if_doesnt_exist(model_name)
encoder_path = os.path.join("models", model_name, "encoder.pth")
depth_decoder_path = os.path.join("models", model_name, "depth.pth")

# LOADING PRETRAINED MODEL
encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
depth_decoder.load_state_dict(loaded_dict)

encoder.eval()
depth_decoder.eval()

# tune the value of k
k = 0.36
image_path = "/media/haoming/970EVO/Pharuj/transfuser_datagen/Town01_Scenario1/Town01_Scenario1_route0_03_20_13_28_33/rgb_front/0000.png"
# gt_path = f"test_data/depth/{iter:04d}.png"

input_image = Image.open(image_path).convert('RGB')
original_width, original_height = input_image.size

feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
input_image_resized = input_image.resize((feed_width, feed_height), Image.LANCZOS)

input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

with torch.no_grad():
    features = encoder(input_image_pytorch)
    outputs = depth_decoder(features)

disp = outputs[("disp", 0)]
# print("model shape: ", disp.shape)

arr = disp.squeeze().detach().cpu().numpy()
np.max(arr), np.min(arr)

disp_resized = torch.nn.functional.interpolate(disp,
    (original_height, original_width), mode="bilinear", align_corners=False)
# print("image shape: ", disp_resized.shape)

# Saving colormapped depth image
disp_resized_np = disp_resized.squeeze().cpu().numpy()
vmax = np.percentile(disp_resized_np, 95)
# print("disp_resized_np: ", disp_resized_np.shape)

# convert to distance
distance = k / disp_resized_np
distance = np.clip(distance, 0, 80)
# np.set_printoptions(threshold=np.inf)
print("distance: ", distance.shape, max(distance.flatten()), min(distance.flatten()))
# print("distance: ", distance)

dlog_upper = np.log(distance) + 0.6
dlog_lower = np.log(distance) - 0.6
d_upper = np.exp(dlog_upper)
d_lower = np.exp(dlog_lower)
print("d_upper:", d_upper)
print("d_lower:", d_lower)
np.save("/media/haoming/970EVO/Yaguang/examples/test_data/d_upper.npy", d_upper)
np.save("/media/haoming/970EVO/Yaguang/examples/test_data/d_lower.npy", d_lower)

# plot the result
plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.imshow(input_image)
plt.title("Input", fontsize=22)
plt.axis('off')

# plt.subplot(312)
# plt.imshow(np.log(d_gt))
# plt.title("log_GT", fontsize=22)
# plt.axis('off')
# plt.colorbar()

plt.subplot(212)
plt.imshow(np.log(distance))
plt.title("log_Distance", fontsize=22)
plt.axis('off')
plt.colorbar()
plt.show()

breakpoint()