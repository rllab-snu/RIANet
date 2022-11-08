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

    scale = 1  # image pre-processing
    crop = 256  # image pre-processing

    lr = 1e-4  # learning rate

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

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),  # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        B, T, C = x.size()

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer,
                 vert_anchors, horz_anchors, seq_len,
                 embd_pdrop, attn_pdrop, resid_pdrop, config):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(1, (self.config.n_views + 1) * seq_len * vert_anchors * horz_anchors, n_embd))

        # velocity embedding
        self.vel_emb = nn.Linear(1, n_embd)
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head,
                                            block_exp, attn_pdrop, resid_pdrop)
                                      for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, lidar_tensor, velocity):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """

        bz = lidar_tensor.shape[0] // self.seq_len
        h, w = lidar_tensor.shape[2:4]

        # forward the image model for token embeddings
        image_tensor = image_tensor.view(bz, self.config.n_views * self.seq_len, -1, h, w)
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, h, w)

        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat([image_tensor, lidar_tensor], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        token_embeddings = token_embeddings.view(bz, -1, self.n_embd)  # (B, an * T, C)

        # project velocity to n_embed
        velocity_embeddings = self.vel_emb(velocity.unsqueeze(1))  # (B, C)

        # add (learnable) positional embedding and velocity embedding for all tokens
        x = self.drop(self.pos_emb + token_embeddings + velocity_embeddings.unsqueeze(1))  # (B, an * T, C)
        # x = self.drop(token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)
        x = x.view(bz, (self.config.n_views + 1) * self.seq_len, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # same as token_embeddings

        image_tensor_out = x[:, :self.config.n_views * self.seq_len, :, :, :].contiguous().view(
            bz * self.config.n_views * self.seq_len, -1, h, w)
        lidar_tensor_out = x[:, self.config.n_views * self.seq_len:, :, :, :].contiguous().view(bz * self.seq_len, -1,
                                                                                                h, w)

        return image_tensor_out, lidar_tensor_out

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

        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))

        self.graph_encoder1 = GNN(self.config.f_hidden, self.config.f_out, max_node_num=self.config.max_node_num)

        self.graph_encoder2 = GCN_layer(self.config.f_out, self.config.f_out)
        self.graph_encoder3 = GCN_layer(self.config.f_out, self.config.f_out)
        self.graph_encoder4 = GCN_layer(self.config.f_out, self.config.f_out)
        self.graph_last_decoder = GCN_layer(self.config.f_out, 512)

        self.image_linear_encoding1 = nn.Linear(64*64, self.config.f_out)
        self.lidar_linear_encoding1 = nn.Linear(64*64, self.config.f_out)
        self.image_linear_decoding1 = nn.Linear(self.config.f_out, 64*64)
        self.lidar_linear_decoding1 = nn.Linear(self.config.f_out, 64*64)

        self.image_linear_encoding2 = nn.Linear(32*32, self.config.f_out)
        self.lidar_linear_encoding2 = nn.Linear(32*32, self.config.f_out)
        self.image_linear_decoding2 = nn.Linear(self.config.f_out, 32*32)
        self.lidar_linear_decoding2 = nn.Linear(self.config.f_out, 32*32)

        self.image_linear_encoding3 = nn.Linear(16*16, self.config.f_out)
        self.lidar_linear_encoding3 = nn.Linear(16*16, self.config.f_out)
        self.image_linear_decoding3 = nn.Linear(self.config.f_out, 16*16)
        self.lidar_linear_decoding3 = nn.Linear(self.config.f_out, 16*16)

        self.image_linear_encoding4 = nn.Linear(8*8, self.config.f_out)
        self.lidar_linear_encoding4 = nn.Linear(8*8, self.config.f_out)
        self.image_linear_decoding4 = nn.Linear(self.config.f_out, 8*8)
        self.lidar_linear_decoding4 = nn.Linear(self.config.f_out, 8*8)

        self.transformer1 = GPT(n_embd=64,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer2 = GPT(n_embd=128,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer3 = GPT(n_embd=256,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer4 = GPT(n_embd=512,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)

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

        feature_mask = 1 - torch.cumsum(F.one_hot(node_num, num_classes=self.config.max_node_num + 1), dim=1)[:, :-1]

        # start transformer
        image_features = self.image_encoder.conv1(image_tensor)
        image_features = self.image_encoder.bn1(image_features)
        image_features = self.image_encoder.relu(image_features)
        image_features = self.image_encoder.maxpool(image_features)
        lidar_features = self.lidar_encoder.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder.bn1(lidar_features)
        lidar_features = self.lidar_encoder.relu(lidar_features)
        lidar_features = self.lidar_encoder.maxpool(lidar_features)

        image_features = self.image_encoder.layer1(image_features)
        lidar_features = self.lidar_encoder.layer1(lidar_features)
        graph_features = self.graph_encoder1(adj, node_feature_matrix, edge_feature_matrix, node_num)

        # fusion at (B, 64, 64, 64)
        image_embd_layer1 = self.avgpool(image_features)
        lidar_embd_layer1 = self.avgpool(lidar_features)
        image_features_layer1, lidar_features_layer1 = self.transformer1(image_embd_layer1, lidar_embd_layer1, velocity)
        image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=8, mode='bilinear', align_corners=False)
        lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=8, mode='bilinear', align_corners=False)

        image_features_linear1 = torch.mean(self.image_linear_encoding1(torch.flatten(image_features_layer1, 2)), dim=1)  # batch x 1 x f
        lidar_features_linear1 = torch.mean(self.lidar_linear_encoding1(torch.flatten(lidar_features_layer1, 2)), dim=1)  # batch x 1 x f
        visual_features_linear1 = torch.stack([image_features_linear1, lidar_features_linear1], dim=1)
        visual_context1, visual_attn1 = self.visual_query_transformer1(visual_features_linear1, graph_features, feature_mask) # batch x 2 x f
        graph_context1, graph_attn1 = self.graph_query_transformer1(graph_features, visual_features_linear1, None)  # batch x node x f
        image_features = image_features + image_features_layer1 + self.image_linear_decoding1(visual_context1[:,:1,:]).view(-1,1,64,64)
        lidar_features = lidar_features + lidar_features_layer1 + self.lidar_linear_decoding1(visual_context1[:,1:,:]).view(-1,1,64,64)
        graph_features = graph_features + graph_context1

        image_features = self.image_encoder.layer2(image_features)
        lidar_features = self.lidar_encoder.layer2(lidar_features)
        graph_features = F.leaky_relu(self.graph_encoder2(adj, graph_features, feature_mask))
        # fusion at (B, 128, 32, 32)
        image_embd_layer2 = self.avgpool(image_features)
        lidar_embd_layer2 = self.avgpool(lidar_features)
        image_features_layer2, lidar_features_layer2 = self.transformer2(image_embd_layer2, lidar_embd_layer2, velocity)
        image_features_layer2 = F.interpolate(image_features_layer2, scale_factor=4, mode='bilinear', align_corners=False)
        lidar_features_layer2 = F.interpolate(lidar_features_layer2, scale_factor=4, mode='bilinear', align_corners=False)

        image_features_linear2 = torch.mean(self.image_linear_encoding2(torch.flatten(image_features_layer2, 2)), dim=1)  # batch x 1 x f
        lidar_features_linear2 = torch.mean(self.lidar_linear_encoding2(torch.flatten(lidar_features_layer2, 2)), dim=1)  # batch x 1 x f
        visual_features_linear2 = torch.stack([image_features_linear2, lidar_features_linear2], dim=1)
        visual_context2, visual_attn2 = self.visual_query_transformer2(visual_features_linear2, graph_features, feature_mask) # batch x 2 x f
        graph_context2, graph_attn2 = self.graph_query_transformer2(graph_features, visual_features_linear2, None)  # batch x node x f
        image_features = image_features + image_features_layer2 + self.image_linear_decoding2(visual_context2[:,:1,:]).view(-1,1,32,32)
        lidar_features = lidar_features + lidar_features_layer2 + self.lidar_linear_decoding2(visual_context2[:,1:,:]).view(-1,1,32,32)
        graph_features = graph_features + graph_context2

        image_features = self.image_encoder.layer3(image_features)
        lidar_features = self.lidar_encoder.layer3(lidar_features)
        graph_features = F.leaky_relu(self.graph_encoder3(adj, graph_features, feature_mask))
        # fusion at (B, 256, 16, 16)
        image_embd_layer3 = self.avgpool(image_features)
        lidar_embd_layer3 = self.avgpool(lidar_features)
        image_features_layer3, lidar_features_layer3 = self.transformer3(image_embd_layer3, lidar_embd_layer3, velocity)
        image_features_layer3 = F.interpolate(image_features_layer3, scale_factor=2, mode='bilinear', align_corners=False)
        lidar_features_layer3 = F.interpolate(lidar_features_layer3, scale_factor=2, mode='bilinear', align_corners=False)

        image_features_linear3 = torch.mean(self.image_linear_encoding3(torch.flatten(image_features_layer3, 2)), dim=1)  # batch x 1 x f
        lidar_features_linear3 = torch.mean(self.lidar_linear_encoding3(torch.flatten(lidar_features_layer3, 2)), dim=1)  # batch x 1 x f
        visual_features_linear3 = torch.stack([image_features_linear3, lidar_features_linear3], dim=1)
        visual_context3, visual_attn3 = self.visual_query_transformer3(visual_features_linear3, graph_features, feature_mask) # batch x 2 x f
        graph_context3, graph_attn3 = self.graph_query_transformer3(graph_features, visual_features_linear3, None)  # batch x node x f
        image_features = image_features + image_features_layer3 + self.image_linear_decoding3(visual_context3[:,:1,:]).view(-1,1,16,16)
        lidar_features = lidar_features + lidar_features_layer3 + self.lidar_linear_decoding3(visual_context3[:,1:,:]).view(-1,1,16,16)
        graph_features = graph_features + graph_context3

        image_features = self.image_encoder.layer4(image_features)
        lidar_features = self.lidar_encoder.layer4(lidar_features)
        graph_features = F.leaky_relu(self.graph_encoder4(adj, graph_features, feature_mask))
        # fusion at (B, 512, 8, 8)
        image_embd_layer4 = self.avgpool(image_features)
        lidar_embd_layer4 = self.avgpool(lidar_features)
        image_features_layer4, lidar_features_layer4 = self.transformer4(image_embd_layer4, lidar_embd_layer4, velocity)

        image_features_linear4 = torch.mean(self.image_linear_encoding4(torch.flatten(image_features_layer4, 2)), dim=1)  # batch x 1 x f
        lidar_features_linear4 = torch.mean(self.lidar_linear_encoding4(torch.flatten(lidar_features_layer4, 2)), dim=1)  # batch x 1 x f
        visual_features_linear4 = torch.stack([image_features_linear4, lidar_features_linear4], dim=1)
        visual_context4, visual_attn4 = self.visual_query_transformer4(visual_features_linear4, graph_features, feature_mask) # batch x 2 x f
        graph_context4, graph_attn4 = self.graph_query_transformer4(graph_features, visual_features_linear4, None)  # batch x node x f
        image_features = image_features + image_features_layer4 + self.image_linear_decoding4(visual_context4[:,:1,:]).view(-1,1,8,8)
        lidar_features = lidar_features + lidar_features_layer4 + self.lidar_linear_decoding4(visual_context4[:,1:,:]).view(-1,1,8,8)
        graph_features = graph_features + graph_context4

        image_features = self.image_encoder.avgpool(image_features)
        image_features = torch.flatten(image_features, 1)
        image_features = image_features.view(bz, self.config.n_views * self.config.seq_len, -1)
        lidar_features = self.lidar_encoder.avgpool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)
        lidar_features = lidar_features.view(bz, self.config.seq_len, -1)

        graph_features = torch.mean(self.graph_last_decoder(adj, graph_features, feature_mask), dim=1).unsqueeze(1)

        fused_features = torch.cat([image_features, lidar_features, graph_features], dim=1)
        fused_features = torch.sum(fused_features, dim=1)

        return fused_features

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.args = args
        self.device = device
        self.config = ModelConfig()

        self.encoder = Encoder(self.config).to(self.device)

        self.join = nn.Sequential(
            nn.Linear(512, 256),
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