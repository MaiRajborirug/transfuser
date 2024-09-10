import argparse
import json
import os
from tqdm import tqdm
import torch.nn.functional as F

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import GlobalConfig
from model import LidarCenterNet
from data import CARLA_Data, lidar_bev_cam_correspondences, CARLA_Data_new

import pathlib
import datetime
from torch.distributed.elastic.multiprocessing.errors import record
import random
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.multiprocessing as mp

from optical_flow import optical_flow
from segmentation import segmentation
# from team_code_transfuser.alg1_pycuda import Algorithm1
from alg1_pycuda_copy import Algorithm1 #
from alg2 import Algorithm2

from diskcache import Cache
# Records error and tracebacks in case of failure
@record
def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='transfuser', help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=42, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=0, help='Learning rate.') # e-6 now
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for one GPU. When training with multiple GPUs the effective batch size will be batch_size*num_gpus')
    parser.add_argument('--logdir', type=str, default='/home/haoming/git/transfuser/training', help='Directory to log data to.')
    # parser.add_argument('--load_file', type=str, default=None, help='ckpt to load.')
    parser.add_argument('--load_file', type=str, default='/home/haoming/git/transfuser/model_ckpt/models_2022/transfuser/model_seed1_39.pth', help='ckpt to load.')

    parser.add_argument('--start_epoch', type=int, default=41, help='Epoch to start with. Useful when continuing trainings via load_file.')
    parser.add_argument('--setting', type=str, default='all', help='What training setting to use. Options: '
                                                                   'all: Train on all towns no validation data. '
                                                                   '02_05_withheld: Do not train on Town 02 and Town 05. Use the data as validation data.')
    parser.add_argument('--root_dir', type=str, default='/media/haoming/970EVO/Pharuj/transfuser_data', help='Root directory of your training data')
    parser.add_argument('--schedule', type=int, default=1,
                        help='Whether to train with a learning rate schedule. 1 = True')
    parser.add_argument('--schedule_reduce_epoch_01', type=int, default=30,
                        help='Epoch at which to reduce the lr by a factor of 10 the first time. Only used with --schedule 1')
    parser.add_argument('--schedule_reduce_epoch_02', type=int, default=40,
                        help='Epoch at which to reduce the lr by a factor of 10 the second time. Only used with --schedule 1')
    parser.add_argument('--backbone', type=str, default='transFuser',
                        help='Which Fusion backbone to use. Options: transFuser, late_fusion, latentTF, geometric_fusion')
    parser.add_argument('--image_architecture', type=str, default='regnety_032',
                        help='Which architecture to use for the image branch. efficientnet_b0, resnet34, regnety_032 etc.')
    parser.add_argument('--lidar_architecture', type=str, default='regnety_032',
                        help='Which architecture to use for the lidar branch. Tested: efficientnet_b0, resnet34, regnety_032 etc.')
    parser.add_argument('--use_velocity', type=int, default=0,
                        help='Whether to use the velocity input. Currently only works with the TransFuser backbone. Expected values are 0:False, 1:True')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer layers used in the transfuser')
    parser.add_argument('--wp_only', type=int, default=0,
                        help='Valid values are 0, 1. 1 = using only the wp loss; 0= using all losses')
    parser.add_argument('--use_target_point_image', type=int, default=1,
                        help='Valid values are 0, 1. 1 = using target point in the LiDAR0; 0 = dont do it')
    parser.add_argument('--use_point_pillars', type=int, default=0,
                        help='Whether to use the point_pillar lidar encoder instead of voxelization. 0:False, 1:True')
    parser.add_argument('--parallel_training', type=int, default=0,
                        help='If this is true/1 you need to launch the train.py script with CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 --rdzv_id=123456780 --rdzv_backend=c10d train.py '
                             ' the code will be parallelized across GPUs. If set to false/0, you launch the script with python train.py and only 1 GPU will be used.')
    parser.add_argument('--val_every', type=int, default=5, help='At which epoch frequency to validate.') # 1
    parser.add_argument('--no_bev_loss', type=int, default=0, help='If set to true the BEV loss will not be trained. 0: Train noหrmally, 1: set training weight for BEV to 0')
    parser.add_argument('--sync_batch_norm', type=int, default=0, help='0: Compute batch norm for each GPU independently, 1: Synchronize Batch norms accross GPUs. Only use with --parallel_training 1')
    parser.add_argument('--zero_redundancy_optimizer', type=int, default=0, help='0: Normal AdamW Optimizer, 1: Use Zero Reduncdancy Optimizer to reduce memory footprint. Only use with --parallel_training 1')
    parser.add_argument('--use_disk_cache', type=int, default=0, help='0: Do not cache the dataset 1: Cache the dataset on the disk pointed to by the SCRATCH enironment variable. Useful if the dataset is stored on slow HDDs and can be temporarily stored on faster SSD storage.')
    parser.add_argument('--arg_file', type=str, default='/home/haoming/git/transfuser/model_ckpt/models_2022/args.txt', help='Path to a txt file containing the arguments. The arguments in the file will overwrite the arguments passed via the command line.')


    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)
    parallel = bool(args.parallel_training)

    if(bool(args.use_disk_cache) == True):
        if (parallel == True):
            # NOTE: This is specific to our cluster setup where the data is stored on slow storage.
            # During training we cache the dataset on the fast storage of the local compute nodes.
            # Adapt to your cluster setup as needed.
            # Important initialize the parallel threads from torch run to the same folder (so they can share the cache).
            tmp_folder = str(os.environ.get('SCRATCH'))
            print("Tmp folder for dataset cache: ", tmp_folder)
            tmp_folder = tmp_folder + "/dataset_cache"
            # We use a local diskcache to cache the dataset on the faster SSD drives on our cluster.
            shared_dict = Cache(directory=tmp_folder ,size_limit=int(768 * 1024 ** 3))
        else:
            shared_dict = Cache(size_limit=int(768 * 1024 ** 3))
    else:
        shared_dict = None

    # Use torchrun for starting because it has proper error handling. Local rank will be set automatically
    if(parallel == True): #Non distributed works better with my local debugger
        rank       = int(os.environ["RANK"]) #Rank accross all processes
        local_rank = int(os.environ["LOCAL_RANK"]) # Rank on Node
        world_size = int(os.environ['WORLD_SIZE']) # Number of processes
        print(f"RANK, LOCAL_RANK and WORLD_SIZE in environ: {rank}/{local_rank}/{world_size}")

        device = torch.device('cuda:{}'.format(local_rank))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank) # Hide devices that are not used by this process

        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank,
                                             timeout=datetime.timedelta(minutes=15))

        torch.distributed.barrier(device_ids=[local_rank])
    else:
        rank       = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda:{}'.format(local_rank))

    torch.cuda.set_device(device) # NOTE have to work on this accorddingly

    torch.backends.cudnn.benchmark = True # Wen want the highest performance

    # Configure config
    config = GlobalConfig(root_dir=args.root_dir, setting=args.setting)
    config.use_target_point_image = bool(args.use_target_point_image)
    config.n_layer = args.n_layer
    config.use_point_pillars = bool(args.use_point_pillars)
    config.backbone = args.backbone
    if(bool(args.no_bev_loss)):
        index_bev = config.detailed_losses.index("loss_bev")
        config.detailed_losses_weights[index_bev] = 0.0

    # NOTE overwrite config with arg.txt
    args_file = open(args.arg_file, 'r')
    args_ = json.load(args_file)
    args_file.close()

    if ('sync_batch_norm' in args_):
        config.sync_batch_norm = bool(args_['sync_batch_norm'])
    if ('use_point_pillars' in args_):
        config.use_point_pillars = args_['use_point_pillars']
    if ('n_layer' in args_):
        config.n_layer = args_['n_layer']
    if ('use_target_point_image' in args_):
        config.use_target_point_image = bool(args_['use_target_point_image'])
    if ('use_velocity' in args_):
        use_velocity = bool(args_['use_velocity'])
    else:
        use_velocity = True
    if ('image_architecture' in args_):
        image_architecture = args_['image_architecture']
    else:
        image_architecture = 'resnet34'

    if ('lidar_architecture' in args_):
        lidar_architecture = args_['lidar_architecture']
    else:
        lidar_architecture = 'resnet18'

    if ('backbone' in args_):
        backbone = args_['backbone']  # Options 'geometric_fusion', 'transFuser', 'late_fusion', 'latentTF'
    else:
        backbone = 'transFuser'  # Options 'geometric_fusion', 'transFuser', 'late_fusion', 'latentTF'

    # Create model and optimizers
    # model = LidarCenterNet(config, device, args.backbone, args.image_architecture, args.lidar_architecture, bool(args.use_velocity))
    model = LidarCenterNet(config, device, backbone, image_architecture, lidar_architecture, use_velocity)

    if (parallel == True):
        # Synchronizing the Batch Norms increases the Batch size with which they are compute by *num_gpus
        if(bool(args.sync_batch_norm) == True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)

    model.cuda(device=device)

    if ((bool(args.zero_redundancy_optimizer) == True) and (parallel == True)):

        optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=optim.AdamW, lr=args.lr) # Saves GPU memory during DDP training
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr) # For single GPU training


    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print ('Total trainable parameters: ', params)

    # NOTE: Data new
    train_set = CARLA_Data_new(root=config.train_data, config=config, shared_dict=shared_dict)
    val_set   = CARLA_Data_new(root=config.val_data,   config=config, shared_dict=shared_dict)

    g_cuda = torch.Generator(device='cpu')
    g_cuda.manual_seed(torch.initial_seed())

    if(parallel == True):
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, num_replicas=world_size, rank=rank)
        sampler_val   = torch.utils.data.distributed.DistributedSampler(val_set,   shuffle=True, num_replicas=world_size, rank=rank)
        dataloader_train = DataLoader(train_set, sampler=sampler_train, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=8, pin_memory=True)
        dataloader_val   = DataLoader(val_set,   sampler=sampler_val,   batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=8, pin_memory=True)
    else:
      dataloader_train = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=0, pin_memory=True)
      dataloader_val   = DataLoader(val_set,   shuffle=True, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=0, pin_memory=True)

    # Create logdir
    if ((not os.path.isdir(args.logdir)) and (rank == 0)):
        print('Created dir:', args.logdir, rank)
        os.makedirs(args.logdir, exist_ok=True)

    # We only need one process to log the losses
    if(rank == 0):
        writer = SummaryWriter(log_dir=args.logdir)
        # Log args
        with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        writer = None

    if (not (args.load_file is None)):
        # Load checkpoint
        print("=============load=================")
        print(args.load_file)
        state_dict = torch.load(args.load_file, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module'):
                new_k = k[7:]
            else:
                new_k = k
            new_state_dict[new_k] = v

        model.load_state_dict(new_state_dict, strict=False)  # load model have  ('_model.lidar_encoder._model.stem.conv.weight', torch.Size([32, 3, 3, 3]))
        del state_dict
        del new_state_dict
        model.cuda()
        # model.load_state_dict(torch.load(args.load_file, map_location=model.device))
        # optimizer.load_state_dict(torch.load(args.load_file.replace("model_", "optimizer_"), map_location=model.device))

    # NOTE: freeze layers
    # Define the main layer names to keep trainable
    trainable_main_layer_names = ["decoder", "join", "avgpool", "output"]

    # Loop through the named parameters and freeze layers not in the trainable_layer_names
    for name, param in model.named_parameters():
        should_train = any(main_layer_name == name.split('.')[0] for main_layer_name in trainable_main_layer_names)
        param.requires_grad = should_train

    # Verify that the desired layers are now trainable
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
            # print("Trainable Layer after freeze:", name)

    # check if trainable parameter changes
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters after freeze: ', params)


    trainer = Engine(model=model, optimizer=optimizer, dataloader_train=dataloader_train, dataloader_val=dataloader_val,
                     args=args, config=config, writer=writer, device=device, rank=rank, world_size=world_size,
                     parallel=parallel, cur_epoch=args.start_epoch)
    
    # alg1_solver = Algorithm1(trainer.focal_len, (trainer.camera_width, trainer.camera_height), trainer.X, trainer.Y, trainer.certification_offset) 

    # alg2_solver = Algorithm2()
    
    for epoch in range(trainer.cur_epoch, args.epochs):
        if(parallel == True):
            # Update the seed depending on the epoch so that the distributed sampler will use different shuffles across different epochs
            sampler_train.set_epoch(epoch)
        if ((epoch == args.schedule_reduce_epoch_01) or (epoch==args.schedule_reduce_epoch_02)) and (args.schedule == 1):
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = current_lr * 0.1
            print("Reduce learning rate by factor 10 to:", new_lr)
            for g in optimizer.param_groups:
                g['lr'] = new_lr
        trainer.train()

        if((args.setting != 'all') and (epoch % args.val_every == 0)):
            trainer.validate()

        if (parallel == True):
            if (bool(args.zero_redundancy_optimizer) == True):
                optimizer.consolidate_state_dict(0) # To save the whole optimizer we need to gather it on GPU 0.
            if (rank == 0):
                trainer.save()
        else:
            trainer.save()

class Engine(object):
    """
    Engine that runs training.
    """

    def __init__(self, model, optimizer, dataloader_train, dataloader_val, args, config, writer, device, rank=0, world_size=1, parallel=False, cur_epoch=0):
        self.cur_epoch = cur_epoch
        self.bestval_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10
        self.model = model
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.dataloader_val   = dataloader_val
        self.args = args
        self.config = config
        self.writer = writer
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.parallel = parallel
        self.vis_save_path = self.args.logdir + r'/visualizations'
        if(self.config.debug == True):
            pathlib.Path(self.vis_save_path).mkdir(parents=True, exist_ok=True)

        self.detailed_losses         = config.detailed_losses
        if self.args.wp_only:
            detailed_losses_weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            detailed_losses_weights = config.detailed_losses_weights
        self.detailed_weights = {key: detailed_losses_weights[idx] for idx, key in enumerate(self.detailed_losses)}

        # NOTE: add optical flow and segmentation
        self.fov = 60 # consider only front camera use arc tan
        self.fov_v = 32 # from computation
        self.camera_width = config.img_width # 320
        self.camera_height = config.img_resolution[0] # 160
        # c * tan(fov/2=60)=960/2 --> c = 160(3)^0.5
        # c * tan(new_fov/2)=320/2=160
        self.ppdh = self.camera_width / self.fov # pixel:degree # assume the same for x and y
        self.ppdv = self.camera_height / self.fov_v
        self.ppd = (self.ppdh + self.ppdv)/2
        # define ourselves # original size
        self.camera_size_x = 0.032
        self.camera_size_y = 0.016
        # x and y should be equal given no distortion
        self.meters_per_pixel_x = self.camera_size_x / self.camera_width
        self.meters_per_pixel_y = self.camera_size_y / self.camera_height
        self.focal_len = self.camera_size_x / 2 / np.tan(np.deg2rad(self.fov / 2))

        self.optical_flow = optical_flow(self.camera_height, self.camera_width, self.meters_per_pixel_x, self.meters_per_pixel_y)
        self.segmentation = segmentation()
        self.X, self.Y = self.create_XY()
        self.turning_radius = 3.7756/1.5

        # NOTE: add policy offset
        x_deg_close=12
        x_deg_far=12
        y_deg=11 # from horizontal
        const_temp = 1.15

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
        self.certification_offset = np.float32(self.certification_offset)

        self.no_certification_required = np.full((self.camera_height, self.camera_width), False)
        print('cert required:', self.certification_offset.shape, np.max(self.certification_offset), np.min(self.certification_offset))

        is_sky = np.full((self.camera_height, self.camera_width), False)
        is_sky[0:self.camera_height // 2, :] = True
        self.no_certification_required = np.logical_or(is_sky, self.no_certification_required)

        # algorithm 1 setup
        self.alg1_solver = Algorithm1(self.focal_len, (self.camera_width, self.camera_height), self.X, self.Y, self.certification_offset) 

        print(self.focal_len, self.camera_width, self.camera_height, self.X.shape, self.Y.shape, self.certification_offset.shape, type(self.certification_offset), type(self.X), type(self.Y))

        self.alg2_solver = Algorithm2()
        self.last_psi_e = None

        # self.certify_threshold = 0.998
        self.certify_threshold = 0.9

    def create_XY(self):

        x_mid = self.camera_width // 2
        y_mid = self.camera_height // 2
        xaxis = np.linspace(-x_mid, x_mid, self.camera_width)
        yaxis = np.linspace(-y_mid, y_mid, self.camera_height)

        X, Y = np.meshgrid(xaxis, yaxis)
        X = X * self.meters_per_pixel_x
        Y = Y * self.meters_per_pixel_y
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)

        # print("X row:", X[0, :])
        # print("Y col:", Y[:, 0])
        return X, Y
    
    def xor(self, a, b):
        # Convert inputs to booleans if they are integers
        if isinstance(a, int):
            a = bool(a)
        if isinstance(b, int):
            b = bool(b)
    
        # Perform XOR operation
        return a ^ b


    def load_data_compute_loss(self, data):
        # Move data to GPU
        rgb = data['rgb'].to(self.device, dtype=torch.float32)
        if self.config.multitask:
            depth = data['depth'].to(self.device, dtype=torch.float32)
            semantic = data['semantic'].squeeze(1).to(self.device, dtype=torch.long)
        else:
            depth = None
            semantic = None

        bev = data['bev'].to(self.device, dtype=torch.long)

        if (self.config.use_point_pillars == True):
            lidar = data['lidar_raw'].to(self.device, dtype=torch.float32)
            num_points = data['num_points'].to(self.device, dtype=torch.int32)
        else:
            lidar = data['lidar'].to(self.device, dtype=torch.float32)
            num_points = None

        label = data['label'].to(self.device, dtype=torch.float32)
        ego_waypoint = data['ego_waypoint'].to(self.device, dtype=torch.float32)

        target_point = data['target_point'].to(self.device, dtype=torch.float32)
        target_point_image = data['target_point_image'].to(self.device, dtype=torch.float32)

        ego_vel = data['speed'].to(self.device, dtype=torch.float32)

        # NOTE: adjust
        if ((self.args.backbone == 'transFuser') or (self.args.backbone == 'late_fusion') or (self.args.backbone == 'latentTF')):
            losses, steers, throttles, brakes = self.model.forward_lossncon(rgb, lidar, ego_waypoint=ego_waypoint, target_point=target_point, target_point_image=target_point_image, ego_vel=ego_vel.reshape(-1, 1), bev=bev, label=label, save_path=self.vis_save_path, depth=depth, semantic=semantic, num_points=num_points)

        # # NOTE: original
        # if ((self.args.backbone == 'transFuser') or (self.args.backbone == 'late_fusion') or (self.args.backbone == 'latentTF')):
        #     losses = self.model(rgb, lidar, ego_waypoint=ego_waypoint, target_point=target_point,
        #                    target_point_image=target_point_image,
        #                    ego_vel=ego_vel.reshape(-1, 1), bev=bev,
        #                    label=label, save_path=self.vis_save_path,
        #                    depth=depth, semantic=semantic, num_points=num_points)

        elif (self.args.backbone == 'geometric_fusion'):

            bev_points = data['bev_points'].long().to('cuda', dtype=torch.int64)
            cam_points = data['cam_points'].long().to('cuda', dtype=torch.int64)
            losses = self.model(rgb, lidar, ego_waypoint=ego_waypoint, target_point=target_point,
                           target_point_image=target_point_image,
                           ego_vel=ego_vel.reshape(-1, 1), bev=bev,
                           label=label, save_path=self.vis_save_path,
                           depth=depth, semantic=semantic, num_points=num_points,
                           bev_points=bev_points, cam_points=cam_points)
        else:
            raise ("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF")
        # return losses
        return losses, steers, throttles, brakes # NOTE adjusted

    def train(self):
        self.model.train()

        num_batches = 0
        loss_epoch = 0.0
        detailed_losses_epoch  = {key: 0.0 for key in self.detailed_losses}
        added_losses_epoch = {'throttle_sc':0.0, 'steer_sc': 0.0, 'brake_sc': 0.0}
        self.cur_epoch += 1

        # Train loop
        tot = len(self.dataloader_train)
        count = 0
        for data in tqdm(self.dataloader_train):
            # count += 1
            # if count > 2000:
            #     break

            # NOTE: add alg1_solver segtion
            rgb_sf = data['rgb_sf'].to(self.device, dtype=torch.float32)
            rgb = data['rgb'].to(self.device, dtype=torch.float32)
            # print(type(data['rgb']), data['rgb'].shape, torch.max(data['rgb']), torch.min(data['rgb'])) # shape [12, 3, 160, 704] max 255 min 0 torch.uint8
            rgb_w = rgb.shape[3] // 2
            rgb_seg = rgb[:,:,:,rgb_w-self.config.img_width//2:rgb_w+self.config.img_width//2]
            rgb_seg_sf = rgb_sf[:,:,:,rgb_w-self.config.img_width//2:rgb_w+self.config.img_width//2] # batch, rgb, h, w

            is_road, _, _, is_animated = self.segmentation.predict_batch(rgb_seg) # shape torch.Size([batch, 1, 160, 320])
            speed = data['speed'].numpy()
            steer =  data['steer'].numpy()
            psis = steer * speed / self.turning_radius # NOTE: computed
            
            # NOTE: original train code section
            self.optimizer.zero_grad(set_to_none=True)
            losses, steers_p, throttles_p, brake_p = self.load_data_compute_loss(data) # NOTE: get control action from the model instead of measurement (~ pred waypoints)
            
            throttle_sc, steer_sc, brake_sc = [], [], []
            # safety certificate
            for i in range(is_road.shape[0]): # ~batch size
                rgb_sf_i = rgb_seg_sf[i].cpu().numpy().transpose(1,2,0)
                rgb_i = rgb_seg[i].cpu().numpy().transpose(1,2,0)
                optical_flow_output = self.optical_flow.predict_sf(rgb_i, rgb_sf_i, self.config.carla_frame_rate)

                # convert optical flow to mu and nu
                mu = optical_flow_output[:, :, 0]
                nu = optical_flow_output[:, :, 1]
                mu = mu.astype(np.float32)
                nu = nu.astype(np.float32)
                v_e = speed[i].astype(np.float32)
                psi_e = psis[i].astype(np.float32)
                throttle_i = np.float32(throttles_p[i])
                steer_i = np.float32(steers_p[i])
                brake_i = np.float32(brake_p[i])
                delta_time = self.config.carla_frame_rate

                # convert CARLA control actions to acceleration and steering rate
                if throttle_i > 0:
                    # NOTE old function
                    # interfuser_acc = throttle_i*(-2.64038-0.00917088*v_e+1.53601/v_e+0.000785623*np.exp(0.5*v_e))+throttle_i*throttle_i*(6.85333-0.181589*v_e+3.67525/v_e-0.000760838*np.exp(0.5*v_e))
                    # interfuser_acc = np.maximum(interfuser_acc,-3)

                    # NOTE new function
                    c0, c1, c2, c3, c4, c5 =  -6.15624483, 179.82397538, 8.33112035, 0.8451532, 24.62378696, 1.39212557
                    interfuser_acc = c0 + c1*throttle_i**c3/(v_e+c2)+c4*np.exp(-c5*v_e)
                    interfuser_acc = np.maximum(interfuser_acc,0)
                else:
                    interfuser_acc = -13

                # control output
                interfuser_steer_normalized = steer_i * v_e/self.turning_radius
                interfuser_steering_rate = (interfuser_steer_normalized - psi_e) / delta_time # (now-next)delta time
                nominal_control = (interfuser_acc, interfuser_steering_rate)
                steering_limit = v_e/self.turning_radius
                self.alg2_solver.phi_bounds = ((-steering_limit-psi_e)/delta_time,(steering_limit-psi_e)/delta_time)

                a_e = np.float32(interfuser_acc)  # 
                phi_e = np.float32(interfuser_steering_rate)  # 

                self.nominal_pixel_is_certifieds, _ = self.alg1_solver.run((mu,nu,v_e,psi_e,a_e,phi_e, is_animated[i]))
                self.nominal_pixel_is_certifieds = np.logical_or(is_road, self.nominal_pixel_is_certifieds)
                self.nominal_pixel_is_certifieds = np.logical_or(self.nominal_pixel_is_certifieds, self.no_certification_required)

                if self.nominal_pixel_is_certifieds.sum() / self.nominal_pixel_is_certifieds.size > self.certify_threshold:
                    # collect lost
                    throttle_sc.append(throttle_i)
                    steer_sc.append(steer_i)
                    brake_sc.append(brake_i)
                else:
                    def certify_control_action(control_action):
                        a_e, phi_e = control_action
                        pixel_is_certifieds, _ = self.alg1_solver.run((mu, nu, v_e, psi_e, a_e, phi_e, is_animated[i]))
                        # if (v_e > -0.001 and v_e < 0.001):
                        #     print("certified due to being stationary")
                        #     return True

                        # the pixel is automatically certified if it is 
                        # the road (in other words, not obstacle)
                        pixel_is_certifieds = np.logical_or(is_road, pixel_is_certifieds)

                        # we only want to check certification for those regions we care about
                        pixel_is_certifieds = np.logical_or(pixel_is_certifieds, self.no_certification_required)

                        # cv.imshow("certified, post mask", np.repeat(pixel_is_certifieds[:, :, np.newaxis].astype(np.uint8) * 255, 3, axis=2))
                        return pixel_is_certifieds.sum() / pixel_is_certifieds.size
                    
                    res = self.alg2_solver.run(nominal_control, certify_control_action)
                    
                    if res is None:
                        throttle_sc.append(throttle_i)
                        steer_sc.append(steer_i)
                        brake_sc.append(brake_i)
                    else: # different in control (use throttle, steer) for sanity check
                        (a_control, phi_control) = res
                        steer_sc_i = psi_e + delta_time*phi_control*self.turning_radius/v_e

                        # steer = interfuser_out.steer
                        if a_control > 0:
                            coeff_a = 6.85333-0.181589*v_e+3.67525/v_e-0.000760838*np.exp(0.5*v_e)
                            coeff_b = -2.64038-0.00917088*v_e+1.53601/v_e+0.000785623*np.exp(0.5*v_e)
                            coeff_c = -a_control
                            sol1 = (-coeff_b-np.sqrt(coeff_b*coeff_b-4*coeff_a*coeff_c))/(2*coeff_a)
                            sol2 = (-coeff_b+np.sqrt(coeff_b*coeff_b-4*coeff_a*coeff_c))/(2*coeff_a)
                            if np.minimum(sol1,sol2) > 0:
                                throttle_sc_i = np.minimum(sol1,sol2)
                            else:
                                throttle_sc_i = np.maximum(sol1,sol2)
                            if throttle_sc_i < 0:
                                # print("How?")
                                # return interfuser_out
                                pass
                            brake_sc_i = 0
                        else:
                            throttle_sc_i = 0.0
                            brake_sc_i = 1

                        throttle_sc.append(throttle_sc_i)
                        steer_sc.append(steer_sc_i)
                        brake_sc.append(brake_sc_i)


            print(throttles_p, throttle_sc,  steers_p, steer_sc, brake_p,brake_sc)

            loss = torch.tensor(0.0).to(self.device, dtype=torch.float32)
            for key, value in losses.items():
                loss += self.detailed_weights[key] * value
                detailed_losses_epoch[key] += float(self.detailed_weights[key] * value.item())
    
            # NOTE: add loss in a, phi with weight hyper parameter
            # loss_throttle_sc = F.l1_loss(torch.tensor(throttle_sc), torch.tensor(throttles_p))
            loss_throttle_sc = torch.mean(torch.abs(torch.tensor(throttle_sc) - torch.tensor(throttles_p)))
            loss_steer_sc = torch.mean(torch.abs(torch.tensor(steer_sc) - torch.tensor(steers_p)))
            loss_brake_sc = torch.mean(torch.abs(torch.tensor(brake_sc, dtype=torch.float) - torch.tensor(brake_p, dtype=torch.float)))

            losses.update({
                'throttle_sc': loss_throttle_sc,
                'steer_sc': loss_steer_sc,
                'brake_sc': loss_brake_sc,
            })
            a, b, c = 2.0, 2.0, 2.0
            loss += a * loss_throttle_sc + b * loss_steer_sc + c * loss_brake_sc  # added loss
            added_losses_epoch['throttle_sc'] += float(a * loss_throttle_sc)
            added_losses_epoch['steer_sc'] += float(b * loss_steer_sc)
            added_losses_epoch['brake_sc'] += float(c * loss_brake_sc)
            # print(loss, losses) # NOTE show in terminal
            
            loss.backward()

            self.optimizer.step()
            num_batches += 1
            loss_epoch += float(loss.item())


        # self.log_losses(loss_epoch, detailed_losses_epoch, num_batches, '')
        self.log_losses(loss_epoch, detailed_losses_epoch, num_batches, '')


    @torch.inference_mode() # Faster version of torch_no_grad
    def validate(self):
        self.model.eval()

        num_batches = 0
        loss_epoch = 0.0
        detailed_val_losses_epoch  = {key: 0.0 for key in self.detailed_losses}

        # Evaluation loop loop
        for data in tqdm(self.dataloader_val):
            losses = self.load_data_compute_loss(data)

            loss = torch.tensor(0.0).to(self.device, dtype=torch.float32)

            for key, value in losses.items():
                loss += self.detailed_weights[key] * value
                detailed_val_losses_epoch[key] += float(self.detailed_weights[key] * value.item())

            num_batches += 1
            loss_epoch += float(loss.item())

        self.log_losses(loss_epoch, detailed_val_losses_epoch, num_batches, 'val_')

    # NOTE: adjust
    def log_losses(self, loss_epoch, detailed_losses_epoch, num_batches, prefix=''):
        # Average all the batches into one number
        loss_epoch = loss_epoch / num_batches
        for key, value in detailed_losses_epoch.items():
            detailed_losses_epoch[key] = value / num_batches

        # In parallel training aggregate all values onto the master node.

        gathered_detailed_losses = [None for _ in range(self.world_size)]
        gathered_loss = [None for _ in range(self.world_size)]

        if (self.parallel == True):
            torch.distributed.gather_object(obj=detailed_losses_epoch,
                                            object_gather_list=gathered_detailed_losses if self.rank == 0 else None, 
                                            dst=0)
            torch.distributed.gather_object(obj=loss_epoch, 
                                            object_gather_list=gathered_loss if self.rank == 0 else None,
                                            dst=0)
        else:
            gathered_detailed_losses[0] = detailed_losses_epoch
            gathered_loss[0] = loss_epoch
            
        if (self.rank == 0):
            # Log main loss
            aggregated_total_loss = sum(gathered_loss) / len(gathered_loss)
            self.writer.add_scalar(prefix + 'loss_total', aggregated_total_loss, self.cur_epoch)

            # Log detailed losses
            for key, value in detailed_losses_epoch.items():
                aggregated_value = 0.0
                for i in range(self.world_size):
                    aggregated_value += gathered_detailed_losses[i][key]

                aggregated_value = aggregated_value / self.world_size

                self.writer.add_scalar(prefix + key, aggregated_value, self.cur_epoch)

    def save(self):
        # NOTE saving the model with torch.save(model.module.state_dict(), PATH) if parallel processing is used would be cleaner, we keep it for backwards compatibility
        torch.save(self.model.state_dict(), os.path.join(self.args.logdir, 'model_%d.pth' % self.cur_epoch))
        torch.save(self.optimizer.state_dict(), os.path.join(self.args.logdir, 'optimizer_%d.pth' % self.cur_epoch))

# We need to seed the workers individually otherwise random processes in the dataloader return the same values across workers!
def seed_worker(worker_id):
    # Torch initial seed is properly set across the different workers, we need to pass it to numpy and random.
    worker_seed = (torch.initial_seed()) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    # The default method fork can run into deadlocks.
    # To use the dataloader with multiple workers forkserver or spawn should be used.
    mp.set_start_method('fork')
    print("Start method of multiprocessing:", mp.get_start_method())
    main()
