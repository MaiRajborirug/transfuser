# team_code_transfuser/config.py ------------
class GlobalConfig:
    def __init__(self, root_dir='', setting='all'):


# team_code_transfuser/train_copy.py -------
from model import LidarCenterNet
import torch.optim as optim
from tqdm import tqdm

def main():
    args = [
        backbone='transfuser',
        load_file='checkpoint.pth',
        logdir='directory to log data to',
    ]

    # initilize model
    # Create model and optimizers
    model = LidarCenterNet(config, device, args.backbone,
                           args.image_architecture, args.lidar_architecture,
                           bool(args.use_velocity))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # select some output of iterables based on output of a function (True / False)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters()) # select
    params = sum([np.prod(p.size()) for p in model_parameters])
    print ('Total trainable parameters: ', params)

    model.cuda(device='coda:0')

    dataloader_train = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=0, pin_memory=True)
    dataloader_val   = DataLoader(val_set,   shuffle=True, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=0, pin_memory=True)

    trainer = Engine(model=model, optimizer=optimizer,
                     dataloader_train=dataloader_train, dataloader_val=dataloader_val,
                     args=args, config=config, writer=writer, device=device, rank=rank, world_size=world_size,
                     parallel=parallel, cur_epoch=args.start_epoch)
    
    for epoch in range(trainer.cur_epoch, args.epochs):
        # adapptive lr
        trainer.train()
        trainer.save()
    
class Engine(object):
    "engine that run training"
    def __init__(self):
        # config, args, writer

    def load_data_compute_loss(self, data):
        rgb = data['rgb'].to(self.device, dtype=torch.float32)
        # do the same for: lidar, ego_waypoint, target_point, ego_vel
        losses = self.model(rgb, lidar, ego_waypoint, target_point, ego_vel)
        return losses

    def train(self):
        self.model.train()
        for data in tqdm(self.dataloader_train):
            losses = self.load_data_compute_loss(data)
            loss = 0 # scalar tensor
            for key, value in losses.item():
                loss += self.detailed_weights[key] * value # sum loss

            loss.backward()
            self.optimizer.step()
            # show training detail in self.log_loss
    
    def validate(self):
        self.model.eval()
        for data in tqdm(self.dataloader_train):
            losses = self.load_data_compute_loss(data)
            loss = 0 # scalar tensor
            for key, value in losses.item():
                loss += self.detailed_weights[key] * value # sum loss
            # show training detail in self.log_loss
            
    def save(self):
        # model.state_dict = learnable parameters
        torch.save(self.model.state_dict(), os.path.join(self.args.logdir, 'model_%d.pth' % self.cur_epoch))

        # optimizer.state_dict = hyperparameters used
        torch.save(self.optimizer.state_dict(), os.path.join(self.args.logdir, 'optimizer_%d.pth' % self.cur_epoch))

# freeze certain parameters?
# team_code_transfuser/model.py ------------
from transfuser import TransfuserBackbone, SegDecoder, DepthDecoder
class LidarCenterNetHead(nn.Module):
    def __init__(self, bb, img_arc lidar_arc):
        self._model = TransfuserBackbone(y=use_velocity).to(self.device)

    def forward(self, feats):
        pass 
    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
class LidarCenterNet(nn.Module):
    def __init__(self):
        self._model = TransfuserBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)

        self.head = LidarCenterNetHead() # prediction head
        self.join = nn.Sequential() # way point prediction
        self.decoder = nn.GRUCell
        self.avgpool = nn.AdaptiveAvgPool2d
        self.output = nn.Linear
    def forward_gru(self, z, target_point):
        z = self.decoder(z, target_point)
        dx = self.output(z)
        pred_wp = function(z, dx)
        return pred_wp, pred_brake, steer, throttle, brake
    
    def forward_ego(self):
         features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)

         pred_wp, _, _, _, _ = self.forward_gru(fused_features, target_point)


# team_code_transfuser/transfuser.py ------------
class TransfuserBackbone(nn.Module):
    def __init__(self):
        # GPT, ImageCNN, LidarEncoder
    def forward(self):
        return features, image_features_grid, fused_features

class SegDecoder(nn.Module):
    def __init__(self):
        nn.Sequential()

class DepthDecoder(nn.Module):
    def __init__(self):
        nn.Sequential()

class GPT(nn.Module):
    def __init__(self):
        nn.Sequential(*[Block()])
        nn.LayerNorm()
    def forward(self, img_tensor, lidar_tensor, velocity):
        # Batch * sqr_length, C H, W
        return img_tensor, lidar_tensor, velocity

# class Block <-- class SelfAttention 

# model that are not freezed
"""
  (join): Sequential(
    (0): Linear(in_features=512, out_features=256, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=256, out_features=128, bias=True)
    (3): ReLU(inplace=True)
    (4): Linear(in_features=128, out_features=64, bias=True)
    (5): ReLU(inplace=True)
  )
  (decoder): GRUCell(4, 64)
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (output): Linear(in_features=64, out_features=3, bias=True)
"""

RuntimeError: Error(s) in loading state_dict for LidarCenterNet:
	Missing key(s) in state_dict: "_model.image_encoder.features.stem.conv.weight", "_model.image_encoder.features.stem.bn.weight", "_model.image_encoder.features.stem.bn.bias", "_model.image_encoder.features.stem.bn.running_mean", "_model.image_encoder.features.stem.bn.running_var", "_model.image_encoder.features.s1.b1.conv1.conv.weight", "_model.image_encoder.features.s1.b1.conv1.bn.weight", "_model.image_encoder.features.s1.b1.conv1.bn.bias", "_model.image_encoder.features.s1.b1.conv1.bn.running_mean", "_model.image_encoder.features.s1.b1.conv1.bn.running_var", "_model.image_encoder.features.s1.b1.conv2.conv.weight", "_model.image_encoder.features.s1.b1.conv2.bn.weight", "_model.image_encoder.features.s1.b1.conv2.bn.bias", "_model.image_encoder.features.s1.b1.conv2.bn.running_mean", "_model.image_encoder.features.s1.b1.conv2.bn.running_var", "_model.image_encoder.features.s1.b1.se.fc1.weight", "_model.image_encoder.features.s1.b1.se.fc1.bias", "_model.image_encoder.features.s1.b1.se.fc2.weight", "_model.image_encoder.features.s1.b1.se.fc2.bias", "_model.image_encoder.features.s1.b1.conv3.conv.weight",

Unexpected key(s) in state_dict: "module._model.image_encoder.features.stem.conv.weight", "module._model.image_encoder.features.stem.bn.weight", "module._model.image_encoder.features.stem.bn.bias", "module._model.image_encoder.features.stem.bn.running_mean", "module._model.image_encoder.features.stem.bn.running_var", "module._model.image_encoder.features.stem.bn.num_batches_tracked", "module._model.image_encoder.features.s1.b1.conv1.conv.weight", "module._model.image_encoder.features.s1.b1.conv1.bn.weight", "module._model.image_encoder.features.s1.b1.conv1.bn.bias", "module._model.image_encoder.features.s1.b1.conv1.bn.running_mean", "module._model.image_encoder.features.s1.b1.conv1.bn.running_var", "module._model.image_encoder.features.s1.b1.conv1.bn.num_batches_tracked",
