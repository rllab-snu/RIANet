import math

import numpy as np
import torch
from torch import nn
from torchvision import models

from os.path import join

BatchNorm = nn.BatchNorm2d


class ModelConfig:
    """ base architecture configurations """
	# Data

    input_resolution = 256

    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    lr = 1e-4 # learning rate
    
    imagenet_normalize = True

    feature_size = 19

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

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

class Resnet_fusion(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc_list, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (list)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(Resnet_fusion, self).__init__()
        use_bias = (norm_layer == nn.InstanceNorm2d)
        
        self.input_nc_list = input_nc_list

        enc_model_list = []
        for input_nc in input_nc_list:
            enc_model = [nn.ReflectionPad2d(3),
                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                     norm_layer(ngf),
                     nn.ReLU(True)]


            n_downsampling = 2
            for i in range(n_downsampling):  # add downsampling layers
                mult = 2 ** i
                enc_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]

            mult = 2 ** n_downsampling
            for i in range(n_blocks):       # add ResNet blocks
                enc_model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            
            enc_model = nn.Sequential(*enc_model)
            enc_model_list.append(enc_model)
        self.enc_model_list = nn.ModuleList(enc_model_list)

        dec_model = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            dec_model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        dec_model += [nn.ReflectionPad2d(3)]
        dec_model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        dec_model += [nn.Tanh()]

        self.dec_model = nn.Sequential(*dec_model)

    def forward(self, batch_input_list):
        """Standard forward"""
        enc_result = []
        for i in range(len(self.input_nc_list)):
            enc_result.append(self.enc_model_list[i](batch_input_list[i]))
        enc_result = sum(enc_result)
        dec_result = self.dec_model(enc_result)
        
        return dec_result


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

    
class Model(nn.Module):
    '''
    Transformer-based feature fusion followed by GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.args = args
        self.config = ModelConfig()
        
        if args.perception_type == 'bev':
            self.base_channel_list = [3]
        elif args.perception_type in ['pred-sem', 'gt-sem']:
            channel = 4 if args.use_DA_sem else 12
            self.base_channel_list = [channel] if args.ignore_sides else [channel] * 3
        else:
            self.base_channel_list = [3] if args.ignore_sides else [3, 3, 3]
            
        self.use_lidar = args.use_lidar
        if self.use_lidar:
            self.base_channel_list.append(2)
            
        if args.use_sem_image:
            channel = 4 if args.use_DA_sem else 12
            self.base_channel_list2 = [channel] if args.ignore_sides else [channel] * 3
        else:
            self.base_channel_list2 = [19]

        self.encoder = Resnet_fusion(self.base_channel_list, 3)
        self.encoder = self.encoder.to(self.device)

        self.encoder2 = Resnet_fusion(self.base_channel_list2, 3)
        self.encoder2 = self.encoder2.to(self.device)
        
        self.feature_generator = models.resnet18()
        self.feature_generator.fc = nn.Linear(512, 512, bias=True)
        self.feature_generator = self.feature_generator.to(self.device)

    def forward(self, data):
        image_list, lidar_list = data['images'], data['lidars'] if self.use_lidar else []
        positive_input_list = data['positive_input'] if 'positive_input' in data else None
        negative_input_list = data['negative_input'] if 'negative_input' in data else None
        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
        '''
        if self.config.imagenet_normalize and self.args.perception_type not in ['pred-sem', 'gt-sem']:
            image_list = [normalize_imagenet(image_input) for image_input in image_list]
        input_ = image_list + lidar_list
        
        pred_feature = self.feature_generator(self.encoder(input_))
        positive_feature = self.feature_generator(self.encoder2(positive_input_list)) if positive_input_list is not None else None
        negative_feature = self.feature_generator(self.encoder2(negative_input_list)) if negative_input_list is not None else None

        return pred_feature, positive_feature, negative_feature
