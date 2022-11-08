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

    # GCN Encoder
    max_node_num = 96  # max graph node num
    f_hidden = 64
    f_out = 128

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

        self.graph_pos_encoder = nn.Sequential(
            nn.Linear(2, 64), nn.LeakyReLU(),
            nn.Linear(64, self.config.f_out))

        self.visual_pos_emb = nn.Parameter(
            torch.zeros(1, 2, self.config.f_out))

        self.velocity_encoder = nn.Sequential(
            nn.Linear(1, 64), nn.LeakyReLU(),
            nn.Linear(64, self.config.f_out))

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

    def forward(self, image_list, lidar_list, velocity, adj, node_feature_matrix, edge_feature_matrix, node_num):
        # encoding
        if self.config.imagenet_normalize:
            image_list = [normalize_imagenet(image_input) for image_input in image_list]

        bz, _, h, w = lidar_list[0].shape
        img_channel = image_list[0].shape[1]
        lidar_channel = lidar_list[0].shape[1]
        self.config.n_views = len(image_list) // self.config.seq_len

        image_tensor = torch.stack(image_list, dim=1).view(bz * self.config.n_views * self.config.seq_len, img_channel, h, w)
        lidar_tensor = torch.stack(lidar_list, dim=1).view(bz * self.config.seq_len, lidar_channel, h, w)

        image_features = self.image_encoder(image_tensor)
        lidar_features = self.lidar_encoder(lidar_tensor)
        graph_features = self.graph_encoder(adj, node_feature_matrix, edge_feature_matrix, node_num)
        pos_embedding = node_feature_matrix[:,:,:2]   # batch x node x 2
        pos_embedding = self.graph_pos_encoder(pos_embedding)  # batch x node x f
        graph_features = graph_features + pos_embedding
        feature_mask = 1 - torch.cumsum(F.one_hot(node_num, num_classes=self.config.max_node_num + 1), dim=1)[:, :-1]

        visual_features = torch.stack([image_features, lidar_features], dim=1)  # batch x k x f
        visual_features = visual_features + self.visual_pos_emb + self.velocity_encoder(velocity.unsqueeze(-1)).unsqueeze(1)

        visual_context, visual_attn = self.visual_query_transformer1(visual_features, graph_features, feature_mask)  # batch x k x f
        graph_context, graph_attn = self.graph_query_transformer1(graph_features, visual_features, None)  # batch x node x f

        visual_context, visual_attn = self.visual_query_transformer2(visual_context, graph_context, feature_mask)  # batch x k x f
        graph_context, graph_attn = self.graph_query_transformer2(graph_context, visual_context, None)  # batch x node x f

        visual_context, visual_attn = self.visual_query_transformer3(visual_context, graph_context, feature_mask)  # batch x k x f
        graph_context, graph_attn = self.graph_query_transformer3(graph_context, visual_context, None)  # batch x node x f

        visual_context, visual_attn = self.visual_query_transformer4(visual_context, graph_context, feature_mask)  # batch x k x f
        graph_context, graph_attn = self.graph_query_transformer4(graph_context, visual_context, None)  # batch x node x f

        visual_context = visual_context.mean(dim=1)
        graph_context = graph_context.mean(dim=1)
        fused_features = torch.cat([visual_context, graph_context], -1)   # batch x (2 * f)

        return fused_features

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.args = args
        self.device = device
        self.config = ModelConfig()

        self.encoder = Encoder(self.config).to(self.device)

        self.join = nn.Sequential(
            nn.Linear(self.config.f_out * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        ).to(self.device)
        self.speed_output = nn.Linear(64, 1).to(self.device)

    def forward(self, data):
        image_list, lidar_list, velocity = data['images'], data['lidars'], data['velocity']
        adj, node_feature_matrix, edge_feature_matrix = data['adjacency_matrix'], data['node_feature_matrix'], data['edge_feature_matrix']
        node_num = data['node_num'].to(torch.int64)

        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
        '''
        fused_features = self.encoder(image_list, lidar_list, velocity, \
                                      adj, node_feature_matrix, edge_feature_matrix, node_num)
        z = self.join(fused_features)

        output = self.speed_output(z)

        return output