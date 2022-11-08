import math
from collections import deque

import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import models

class ModelConfig:
    """ base architecture configurations """
	# Data
    seq_len = 1 # input timesteps
    pred_len = 4 # future waypoints predicted

    n_views = 1 # no. of camera views

    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    lr = 1e-4 # learning rate
    dropout = 0.1
    # Conv Encoder

    imagenet_normalize = True

    pred_target_point = True

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

    # GCN Encoder
    max_node_num = 96  # max graph node num
    f_hidden = 64
    f_out = 128

    max_obj_num = 12  # max graph node num

	# Transformer
    n_head = 4
    dim_feedforward = 128

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

class PositionEncoding(nn.Module):
    def __init__(self, n_filters=512, max_len=2000):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x, times):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = []
        for b in range(x.shape[0]):
            pe.append(self.pe.data[times[b].long()]) # (#x.size(-2), n_filters)
        pe_tensor = torch.stack(pe)
        x = x + pe_tensor
        return x

class Attblock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, normalize_before=False):
        super().__init__()
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, trg, src_mask):
        #q = k = self.with_pos_embed(src, pos)
        q = src.permute(1,0,2)
        k = trg.permute(1,0,2)
        if src_mask is not None:
            src_mask = ~src_mask.bool()
        src2, attention = self.attn(q, k, value=k, key_padding_mask=src_mask)
        src2 = src2.permute(1,0,2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attention

class GCN_layer(nn.Module):
    def __init__(self, in_feature_size, out_feature_size, bias=True):
        super().__init__()
        self.in_feature_size = in_feature_size
        self.out_feature_size = out_feature_size

        stdv = 1. / math.sqrt(self.out_feature_size)

        self.weight = Parameter(torch.empty([self.in_feature_size, self.out_feature_size]))
        self.weight.data.uniform_(-stdv, stdv)
        if bias:
            self.bias = Parameter(torch.empty([1, 1, self.out_feature_size]))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.bias = None

    def forward(self, adj, x, feature_mask, add_loop=True):
        # adjacency tensor (batch x node x node)
        # x : input feature (batch x node x f_size)
        # feature_mask : feature mask (batch x node)
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.int64, device=adj.device)
            adj[:, idx, idx] = 1

        x = torch.matmul(x, self.weight)  # dim : batch x node x f_in -> batch x node x f_out
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)  # batch x node x node
        x = torch.bmm(adj, x)  # batch x node x f_out

        if self.bias is not None:
            x = x + self.bias

        x = x * feature_mask.unsqueeze(-1)

        return x

class GNN(nn.Module):
    def __init__(self, f_hidden, f_out, dropout = 0.1, max_node_num=96):
        super().__init__()

        self.f_hidden = f_hidden
        self.f_out = f_out
        self.node_feature_layer = nn.Sequential(
            nn.Linear(6, self.f_hidden // 2), nn.LeakyReLU())
        self.edge_feature_layer = nn.Sequential(
            nn.Linear(2, self.f_hidden // 2), nn.LeakyReLU())

        self.max_node_num = max_node_num

        self.gcn1 = GCN_layer(self.f_hidden, self.f_hidden)
        self.gcn2 = GCN_layer(self.f_hidden, self.f_hidden)
        self.gcn3 = GCN_layer(self.f_hidden, self.f_out)

        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, node_feature_matrix, edge_feature_matrix, node_num):
        # adjacency : batch x node x node
        # node_feature_matrix : batch x node x f_size
        # edge_feature_matrix : batch x node x node x f_size
        # node_num : batch

        # node and edge level encoding
        node_feature_enc = self.node_feature_layer(node_feature_matrix)  # dim : batch x node x f_size
        edge_feature_enc = self.edge_feature_layer(edge_feature_matrix)  # dim : batch x node x node x f_size
        edge_feature_enc = edge_feature_enc * adj.unsqueeze(-1)
        edge_feature_enc = edge_feature_enc.sum(dim=1)  # dim : batch x node x f_size
        matrix_feature_enc = torch.cat([node_feature_enc, edge_feature_enc], dim=2)  # dim : batch x node x f_size

        feature_mask = 1 - torch.cumsum(F.one_hot(node_num, num_classes=self.max_node_num + 1), dim=1)[:, :-1]
        matrix_feature_enc = matrix_feature_enc * feature_mask.unsqueeze(-1)

        x1 = self.dropout(F.leaky_relu(self.gcn1(adj, matrix_feature_enc, feature_mask)))
        x2 = self.dropout(F.leaky_relu(self.gcn2(adj, x1, feature_mask)))
        x3 = self.dropout(F.leaky_relu(self.gcn3(adj, x2, feature_mask)))

        return x3

class Object_encoder(nn.Module):
    def __init__(self, c_size, f_size, max_obj_num=12):
        super().__init__()

        self.max_obj_num = max_obj_num
        self.c_size = c_size
        self.f_size = f_size

        self.linear1 = nn.Linear(c_size, f_size)
        self.linear2 = nn.Linear(4 + 4, f_size)

        self.linear3 = nn.Sequential(
            nn.Linear(f_size * 2, f_size),
            nn.LeakyReLU(),
            nn.Linear(f_size, f_size))

    def forward(self, img_hidden_features, img_class, img_mask, obj_num, img_mask_info):
        # img_hidden_features: b x 64 x 256 x 256
        # img_class: b x obj x 4
        # img_mask: b x obj x 256 x 256
        # obj_num: b
        # img_mask_info: b x obj x 4
        bz = img_hidden_features.shape[0]

        #conv_feature = self.conv1(img_patch) + self.img_pos_emb
        conv_feature = img_hidden_features.unsqueeze(1).expand(-1,self.max_obj_num,-1,-1,-1)   # b x obj x c x 256 x 256
        conv_feature = conv_feature * img_mask.unsqueeze(2)

        conv_feature_sum = conv_feature.sum(dim=4).sum(dim=3)   # b x obj x c
        conv_feature_mean = conv_feature_sum / (img_mask.sum(dim=3).sum(dim=2).unsqueeze(2) + 1e-6)   # b x obj x c
        conv_feature = self.linear1(conv_feature_mean)   # b x obj x f

        img_mask_info = img_mask_info / 64.0
        info_feature = torch.cat([img_class, img_mask_info], dim=2)  # b x obj x 8
        info_feature = self.linear2(info_feature)   # b x obj x f

        final_obj_feature = self.linear3(torch.cat([conv_feature, info_feature], dim=2))

        obj_feature_mask = 1 - torch.cumsum(F.one_hot(obj_num, num_classes=self.max_obj_num + 1), dim=1)[:, :-1]  # b x obj

        return final_obj_feature, obj_feature_mask

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.image_encoder = models.resnet34(pretrained=True)
        self.image_encoder.fc = nn.Linear(512, self.config.f_out)

        self.lidar_encoder = models.resnet18()
        self.lidar_encoder.fc = nn.Linear(512, self.config.f_out)
        _tmp = self.lidar_encoder.conv1
        self.lidar_encoder.conv1 = nn.Conv2d(2, out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)

        self.graph_encoder = GNN(self.config.f_hidden, self.config.f_out, max_node_num=self.config.max_node_num)

        self.velocity_encoder = nn.Sequential(
            nn.Linear(1, 64), nn.LeakyReLU(),
            nn.Linear(64, self.config.f_out))

        self.visual_pos_emb = nn.Parameter(torch.zeros(1, 3+2*self.config.max_obj_num, self.config.f_out))
        self.graph_pos_emb = nn.Parameter(torch.zeros(1, self.config.max_node_num, self.config.f_out))

        self.td_encoder = Object_encoder(64, self.config.f_out, self.config.max_obj_num)
        self.od_encoder = Object_encoder(64, self.config.f_out, self.config.max_obj_num)

        self.visual_query_transformer1 = Attblock(self.config.f_out,
                                      self.config.n_head,
                                      self.config.dim_feedforward,
                                      self.config.dropout)
        self.graph_query_transformer1 = Attblock(self.config.f_out,
                                      self.config.n_head,
                                      self.config.dim_feedforward,
                                      self.config.dropout)

        self.visual_query_transformer2 = Attblock(self.config.f_out,
                                      self.config.n_head,
                                      self.config.dim_feedforward,
                                      self.config.dropout)

        self.graph_query_transformer2 = Attblock(self.config.f_out,
                                      self.config.n_head,
                                      self.config.dim_feedforward,
                                      self.config.dropout)
                                      
        self.visual_query_transformer3 = Attblock(self.config.f_out,
                                      self.config.n_head,
                                      self.config.dim_feedforward,
                                      self.config.dropout)

        self.graph_query_transformer3 = Attblock(self.config.f_out,
                                      self.config.n_head,
                                      self.config.dim_feedforward,
                                      self.config.dropout)
                                      
        self.visual_query_transformer4 = Attblock(self.config.f_out,
                                      self.config.n_head,
                                      self.config.dim_feedforward,
                                      self.config.dropout)

        self.graph_query_transformer4 = Attblock(self.config.f_out,
                                      self.config.n_head,
                                      self.config.dim_feedforward,
                                      self.config.dropout)

        self.last_attn_transformer1 = Attblock(self.config.f_out,
                                      self.config.n_head,
                                      self.config.dim_feedforward,
                                      self.config.dropout)
        self.last_attn_transformer2 = Attblock(self.config.f_out,
                                      self.config.n_head,
                                      self.config.dim_feedforward,
                                      self.config.dropout)

    def forward(self, image_list, lidar_list, velocity, adj, node_feature_matrix, edge_feature_matrix, node_num,
                td_img_class, td_img_mask, td_num, td_img_mask_info, od_img_class, od_img_mask, od_num, od_img_mask_info):
        # encoding
        if self.config.imagenet_normalize:
            image_list = [normalize_imagenet(image_input) for image_input in image_list]

        bz, _, h, w = lidar_list[0].shape
        img_channel = image_list[0].shape[1]
        lidar_channel = lidar_list[0].shape[1]
        self.config.n_views = len(image_list) // self.config.seq_len

        image_tensor = torch.stack(image_list, dim=1).view(bz * self.config.n_views * self.config.seq_len, img_channel, h, w)
        lidar_tensor = torch.stack(lidar_list, dim=1).view(bz * self.config.seq_len, lidar_channel, h, w)

        #image_features = self.image_encoder(image_tensor)
        image_features = self.image_encoder.conv1(image_tensor)
        image_features = self.image_encoder.bn1(image_features)
        image_features = self.image_encoder.relu(image_features)
        image_features = self.image_encoder.maxpool(image_features)
        image_hidden_features = self.image_encoder.layer1(image_features)
        image_features = self.image_encoder.layer2(image_hidden_features)
        image_features = self.image_encoder.layer3(image_features)
        image_features = self.image_encoder.layer4(image_features)
        image_features = self.image_encoder.avgpool(image_features)
        image_features = torch.flatten(image_features, 1)
        image_features = self.image_encoder.fc(image_features)

        lidar_features = self.lidar_encoder(lidar_tensor)
        graph_features = self.graph_encoder(adj, node_feature_matrix, edge_feature_matrix, node_num)
        graph_features = graph_features + self.graph_pos_emb

        graph_feature_mask = 1 - torch.cumsum(F.one_hot(node_num, num_classes=self.config.max_node_num + 1), dim=1)[:, :-1]
        velocity_features = self.velocity_encoder(velocity.unsqueeze(-1))

        image_hidden_features = F.interpolate(image_hidden_features, scale_factor=4, mode='bilinear', align_corners=False) # upsample
        td_obj_features, td_mask = self.td_encoder(image_hidden_features, td_img_class, td_img_mask, td_num, td_img_mask_info)
        od_obj_features, od_mask = self.od_encoder(image_hidden_features, od_img_class, od_img_mask, od_num, od_img_mask_info)

        visual_features = torch.stack([image_features, lidar_features, velocity_features], dim=1)
        visual_features = torch.cat([visual_features, td_obj_features, od_obj_features], dim=1)  # batch x (3+obj+obj) x f
        visual_features += self.visual_pos_emb
        one_mask = torch.ones([bz,3]).to(device=visual_features.device, dtype=torch.int64)
        visual_feature_mask = torch.cat([one_mask, td_mask, od_mask], dim=1)

        visual_context, visual_attn = self.visual_query_transformer1(visual_features, graph_features, graph_feature_mask)  # batch x k x f
        graph_context, graph_attn = self.graph_query_transformer1(graph_features, visual_features, visual_feature_mask)  # batch x node x f

        visual_context, visual_attn = self.visual_query_transformer2(visual_context, graph_context, graph_feature_mask)  # batch x k x f
        graph_context, graph_attn = self.graph_query_transformer2(graph_context, visual_context, visual_feature_mask)  # batch x node x f
        
        visual_context, visual_attn = self.visual_query_transformer3(visual_context, graph_context, graph_feature_mask)  # batch x k x f
        graph_context, graph_attn = self.graph_query_transformer3(graph_context, visual_context, visual_feature_mask)  # batch x node x f
        
        visual_context, visual_attn = self.visual_query_transformer4(visual_context, graph_context, graph_feature_mask)  # batch x k x f
        graph_context, graph_attn = self.graph_query_transformer4(graph_context, visual_context, visual_feature_mask)  # batch x node x f

        visual_context_masked = visual_context * visual_feature_mask.unsqueeze(2)
        visual_mean = (visual_context_masked.sum(dim=1) / (visual_feature_mask.unsqueeze(2).sum(dim=1) + 1e-6)).unsqueeze(1)  # batch x 1 x f
        visual_context, _ = self.last_attn_transformer1(visual_mean, visual_context, visual_feature_mask)  # batch x 1 x f

        graph_context_masked = graph_context * graph_feature_mask.unsqueeze(2)
        graph_mean = (graph_context_masked.sum(dim=1) / (graph_feature_mask.sum(dim=1).unsqueeze(1) + 1e-6)).unsqueeze(1)  # batch x 1 x f
        graph_context, _ = self.last_attn_transformer2(graph_mean, graph_context, graph_feature_mask)  # batch x 1 x f

        visual_context = visual_context.squeeze(1)
        graph_context = graph_context.squeeze(1)
        fused_features = torch.cat([visual_context, graph_context], dim=1)   # batch x (2 * f)

        return fused_features

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.args = args
        self.device = device
        self.config = ModelConfig()
        self.pred_len = self.config.pred_len

        self.turn_controller = PIDController(K_P=self.config.turn_KP, K_I=self.config.turn_KI, K_D=self.config.turn_KD,
                                             n=self.config.turn_n)
        self.speed_controller = PIDController(K_P=self.config.speed_KP, K_I=self.config.speed_KI, K_D=self.config.speed_KD,
                                              n=self.config.speed_n)

        self.encoder = Encoder(self.config).to(self.device)

        self.join = nn.Sequential(
            nn.Linear(self.config.f_out * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        ).to(self.device)
        self.decoder = nn.GRUCell(input_size=2, hidden_size=64).to(self.device)
        self.output = nn.Linear(64, 2).to(self.device)

    def forward(self, data):
        image_list, lidar_list, velocity = data['images'], data['lidars'], data['velocity']
        adj, node_feature_matrix, edge_feature_matrix = data['adjacency_matrix'], data['node_feature_matrix'], data['edge_feature_matrix']
        node_num = data['node_num']
        target_point = data['target_point']

        # (b x obj x 4), (b x obj x 256 x 256), (b)
        td_img_class, td_img_mask, td_num = data['td_img_class'], data['td_img_mask'], data['td_num'].to(torch.int64)
        td_img_mask_info = data['td_img_mask_info']
        od_img_class, od_img_mask, od_num = data['od_img_class'], data['od_img_mask'], data['od_num'].to(torch.int64)
        od_img_mask_info = data['od_img_mask_info']

        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
        '''
        fused_features = self.encoder(image_list, lidar_list, velocity, \
                                      adj, node_feature_matrix, edge_feature_matrix, node_num,
                                      td_img_class, td_img_mask, td_num, td_img_mask_info,
                                      od_img_class, od_img_mask, od_num, od_img_mask_info)
        z = self.join(fused_features)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(self.device)

        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            # x_in = torch.cat([x, target_point], dim=1)
            x_in = x + target_point
            z = self.decoder(x_in, z)
            dx = self.output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)

        return pred_wp

    def control_pid(self, waypoints, velocity):
        '''
        Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): predicted waypoints
            velocity (tensor): speedometer input
        '''
        assert (waypoints.size(0) == 1)
        waypoints = waypoints[0].data.cpu().numpy()

        # flip y is (forward is negative in our waypoints)
        waypoints[:, 1] *= -1
        speed = velocity[0].data.cpu().numpy()

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        if (speed < 0.01):
            angle = np.array(0.0)  # When we don't move we don't want the angle error to accumulate in the integral
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata