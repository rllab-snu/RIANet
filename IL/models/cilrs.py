import math
from collections import deque

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class ModelConfig:
    """ base architecture configurations """
	# Data
    seq_len = 1 # input timesteps
    pred_len = 4 # future waypoints predicted

    ignore_sides = True # don't consider side cameras
    ignore_rear = True # don't consider rear cameras
    n_views = 1 # no. of camera views

    input_resolution = 256

    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    lr = 1e-4 # learning rate

    # Conv Encoder
    vert_anchors = 8
    horz_anchors = 8
    anchors = vert_anchors * horz_anchors

	# GPT Encoder
    n_embd = 512
    block_exp = 4
    n_layer = 8
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40 # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40 # buffer size

    max_throttle = 0.75 # upper limit on throttle signal value in dataset
    brake_speed = 0.1 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25 # maximum change in speed input to logitudinal controller

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class ImageCNN(nn.Module):
    """ Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, c_dim, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()

    def forward(self, inputs):
        c = 0
        for x in inputs:
            if self.normalize:
                x = normalize_imagenet(x)
            c += self.features(x)
        return c


def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x /= 255.0
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


class Controller(nn.Module):
    """ Decoder with velocity input, velocity prediction and conditional control outputs.
    Args:
        num_branch (int): number of conditional branches
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of each decoder branch
        input_velocity (bool): whether to add input velocity information to encoding
        predict_velocity (bool): whether to output a velocity branch prediction
    """

    def __init__(self, num_branch=6, dim=1, c_dim=512, hidden_size=256,
                 input_velocity=True, predict_velocity=True):
        super().__init__()
        self.num_branch = num_branch
        self.input_velocity = input_velocity
        self.predict_velocity = predict_velocity

        # Project input velocity measurement to feature size
        if input_velocity:
            self.vel_in = nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, c_dim),
            )

        # Project feature to velocity prediction
        if predict_velocity:
            self.vel_out = nn.Sequential(
                nn.Linear(c_dim, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, dim),
            )

        # Control branches
        fc_branch_list = []
        for i in range(num_branch):
            fc_branch_list.append(nn.Sequential(
                nn.Linear(c_dim, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, 3),
                nn.Sigmoid(),
            ))

        self.branches = nn.ModuleList(fc_branch_list)

    def forward(self, c, velocity, command):
        batch_size = c.size(0)
        encoding = c

        if self.input_velocity:
            encoding += self.vel_in(velocity.unsqueeze(1))

        control_pred = 0.
        for i, branch in enumerate(self.branches):
            # Choose control for branch of only active command
            # We check for (command - 1) since navigational command 0 is ignored
            control_pred += branch(encoding) * (i == (command - 1)).unsqueeze(1).expand(batch_size, 3)

        if self.predict_velocity:
            velocity_pred = self.vel_out(c)
            return control_pred, velocity_pred

        return control_pred

class Model(nn.Module):
    '''
    Transformer-based feature fusion followed by GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.args = args
        self.config = ModelConfig()
        self.pred_len = self.config.pred_len

        self.encoder = ImageCNN(512, normalize=True).to(self.device)
        self.controller = Controller(num_branch=6, dim=1, c_dim=512, hidden_size=256,
                                input_velocity=True, predict_velocity=True).to(self.device)

    def forward(self, data):
        encoding = [self.encoder(data['images'])]
        velocity, command = data['velocity'], data['command']
        ''' Predicts vehicle control.
        Args:
            c (tensor): latent conditioned code c
            velocity (tensor): speedometer input
            command (tensor): high-level navigational command
        '''
        c = sum(encoding)
        control_pred, velocity_pred = self.controller(c, velocity, command)
        steer = control_pred[:,0] * 2 - 1. # convert from [0,1] to [-1,1]
        throttle = control_pred[:,1] * self.config.max_throttle
        brake = control_pred[:,2]
        return steer, throttle, brake, velocity_pred
