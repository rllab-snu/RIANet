import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

import os
import sys

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, args=None):
        super().__init__()
        if base_kwargs is None:
            base_kwargs = {}
        elif args.env_name is not None:
            if args.net_model_type == 'Road_GNN':
                base = Road_GNN_Base
            elif args.net_model_type == 'MLP':
                base = MLPBase
            elif args.net_model_type == 'LSTM':
                base = LSTMBase
            else:
                base = CNNBase

        self.base = base(obs_shape, args, **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def predict(self, inputs, rnn_hxs=None, masks=None, deterministic=False):
        prediction = self.base.predict(inputs, rnn_hxs, masks)

        return prediction

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super().__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class CNNBase(NNBase):
    def __init__(self, obs_shape, args, recurrent=False, hidden_size=512):
        super().__init__(recurrent, hidden_size, hidden_size)

        self.args = args
        self.obs_shape = obs_shape
        self.img_size = [obs_shape[2], obs_shape[1]] #img_size  # dimension : height x width
        self.observation_length = self.args.observation_length
        self.act_dim = self.args.act_dim # 10 # 5
        self.input_channel = obs_shape[0]

        self.c_size = 16

        self.maxpool2d = nn.MaxPool2d(2)
        self.policy_conv = nn.Sequential(
            nn.Conv2d(self.input_channel, self.c_size, 3, stride=2, padding=1, padding_mode='replicate'), nn.LeakyReLU(),
            nn.Conv2d(self.c_size, self.c_size // 2, 3, stride=2, padding=1, padding_mode='replicate'), nn.LeakyReLU(),
            nn.Conv2d(self.c_size // 2, self.c_size // 4, 3, stride=2, padding=1, padding_mode='replicate'), nn.LeakyReLU(),
        )

        self.max_pool = nn.MaxPool2d(2)

        def div(a):
            return (a + 7) // 8

        img_len = (div(self.img_size[0])) * (div(self.img_size[1])) * (self.c_size // 4)
        self.critic_fc = nn.Sequential(
            nn.Linear(img_len, img_len // 4), nn.LeakyReLU(),
            nn.Linear(img_len // 4, 1),
        )
        self.actor_fc = nn.Sequential(
            nn.Linear(img_len, img_len // 4), nn.LeakyReLU(),
        )

        self._hidden_size = img_len // 4

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        # input dimension : batch x (fixed_channel + time_step * vehicle_num) x height x width
        # current_map = self.maxpool2d(inputs[:,:,:,:])
        inputs = inputs.squeeze()
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        batch_size, h, w = inputs.shape[0], inputs.shape[-2], inputs.shape[-1]

        conv_feature = self.policy_conv(inputs)
        conv_feature_input = conv_feature.view([batch_size, -1])

        critic_outputs = self.critic_fc(conv_feature_input).view([batch_size, 1])
        actor_outputs = self.actor_fc(conv_feature_input).view([batch_size, -1])

        return critic_outputs, actor_outputs, rnn_hxs

class LSTMBase(NNBase):
    def __init__(self, obs_shape, args, recurrent=False, hidden_size=512):
        super().__init__(recurrent, hidden_size, hidden_size)

        self.c_size = args.num_ov + 1
        self.f_size = obs_shape[-1]
        self.act_dim = args.act_dim
        self.obs_length = args.prev_steps
        self.num_ov = args.num_ov

        self.vehicle_encode = nn.Sequential(
            nn.Linear(self.f_size * (1+self.num_ov), self.c_size * self.c_size), nn.LeakyReLU(),
        )

        self.time_lstm = nn.LSTM(self.c_size * self.c_size, self.c_size * self.c_size, batch_first=True, num_layers=2)
        # input : batch x time x in_feature_size
        # output : batch x time x out_feature_size
        # hidden, cell : num_layers x batch x out_feature_size
        # output, (hidden, cell) = lstm(input)

        self.critic_fc = nn.Sequential(
            nn.Linear(self.c_size * self.c_size, self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size, self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size, 1),
        )
        self.actor_fc = nn.Sequential(
            nn.Linear(self.c_size * self.c_size, self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size, self.c_size), nn.LeakyReLU(),
        )

        self._hidden_size = self.c_size

    def forward(self, inputs, rnn_hxs=None, masks=None):
        # inputs : (batch x t x vehicles x features)
        inputs = inputs.squeeze()
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        #linear_features = output_features.view([inputs.size(0), -1])
        vehicle_features = self.vehicle_encode(inputs.view(inputs.shape[:2] + (-1,)))
        #vehicle_features = vehicle_features.view(vehicle_features.shape[:2] + (-1,))  # dim : batch x t x (vehicle x features)
        linear_features = self.time_lstm(vehicle_features)[0][:,-1]

        critic_outputs = self.critic_fc(linear_features)
        actor_outputs = self.actor_fc(linear_features)

        return critic_outputs, actor_outputs, rnn_hxs

class MLPBase(NNBase):
    def __init__(self, obs_shape, args, recurrent=False, hidden_size=512):
        super().__init__(recurrent, hidden_size, hidden_size)

        self.c_size = args.num_ov + 1
        self.f_size = obs_shape[-1]
        self.act_dim = args.act_dim
        self.obs_length = args.prev_steps
        self.num_ov = args.num_ov

        '''
        self.vehicle_encode = nn.Sequential(
            nn.Linear(self.f_size, self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size, self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size, self.c_size),
        )
        self.time_encode = nn.Sequential(
            nn.Linear(self.c_size * (1 + self.num_ov) * self.obs_length, self.c_size * self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size * self.c_size, self.c_size * self.c_size), nn.LeakyReLU(),
        )
        '''

        self.encode = nn.Sequential(
            nn.Linear(self.f_size * (1+self.num_ov) * self.obs_length, self.f_size * self.c_size * self.c_size), nn.LeakyReLU(),
            nn.Linear(self.f_size * self.c_size * self.c_size, self.c_size * self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size * self.c_size, self.c_size * self.c_size),
        )
        self.critic_fc = nn.Sequential(
            nn.Linear(self.c_size * self.c_size, self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size, self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size, 1),
        )
        self.actor_fc = nn.Sequential(
            nn.Linear(self.c_size * self.c_size, self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size, self.c_size), nn.LeakyReLU(),
        )

        self._hidden_size = self.c_size

    def forward(self, inputs, rnn_hxs=None, masks=None):
        # inputs : (batch x t x vehicles x features)
        inputs = inputs.squeeze()
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        #vehicle_features = self.vehicle_encode(inputs)
        #vehicle_features = vehicle_features.view(vehicle_features.shape[:1] + (-1,))
        #output = self.time_encode(vehicle_features)
        #linear_features = output

        linear_features = self.encode(inputs.view([inputs.size(0), -1]))

        critic_outputs = self.critic_fc(linear_features)
        actor_outputs = self.actor_fc(linear_features)

        return critic_outputs, actor_outputs, rnn_hxs

class Road_GNN_Base(NNBase):
    def __init__(self, obs_shape, args, recurrent=False, hidden_size=512):
        super().__init__(recurrent, hidden_size, hidden_size)

        self.node_size = obs_shape[1]
        #self.in_feature_size = obs_shape[2] - obs_shape[1]

        self.args = args
        self.c_size = 8 #args.num_ov + 1
        self.f_size = 8
        self.act_dim = args.act_dim
        self.obs_length = args.prev_steps
        self.num_ov = args.num_ov

        self.edge_feature_size = 2
        self.node_feature_size = 5
        self.edge_feature_layer = nn.Sequential(nn.Linear(self.edge_feature_size, self.f_size), nn.LeakyReLU())
        self.node_feature_layer = nn.Sequential(nn.Linear(self.node_feature_size, self.f_size), nn.LeakyReLU())
        self.ego_node_feature_layer = nn.Sequential(nn.Linear(self.node_feature_size, self.f_size), nn.LeakyReLU())

        self.gcn_module1 = Road_GNN_module(self.f_size, self.f_size, self.c_size)
        #self.gcn_module2 = Road_GNN_module(1, self.f_size, self.c_size)
        self.gcn_to_lin_fc = nn.Sequential(
            nn.Linear(self.node_size * self.f_size, self.c_size * self.f_size), nn.LeakyReLU(),
        )
        #self.graph_feature_weight = nn.Linear(self.obs_length, self.c_size)

        self.enc_lstm = nn.LSTM(self.c_size * self.f_size, self.c_size * self.c_size, batch_first=True, num_layers=2)
        # input : batch x time x in_feature_size
        # output : batch x time x out_feature_size
        # hidden, cell : num_layers x batch x out_feature_size
        # output, (hidden, cell) = lstm(input)

        # only for prediction
        #self.dec_lstm = nn.LSTM(self.c_size * self.f_size, self.c_size * self.c_size, batch_first=True, num_layers=2)
        #self.gcn_module3 = Road_GNN_module(self.f_size, self.f_size, 1+self.num_ov)
        #self.feature_decode_layer1 = nn.Sequential(nn.Linear(self.f_size, 1), nn.LeakyReLU())
        #self.feature_decode_layer2 = nn.Sequential(nn.Linear(1, 1))
        #self.feature_decode_layer3 = nn.Sequential(nn.Linear(self.node_size * self.f_size, self.node_size), nn.LeakyReLU(),
        #                                           nn.Linear(self.node_size, 1))

        self.leaky_relu = nn.LeakyReLU()

        self.critic_fc = nn.Sequential(
            nn.Linear(self.c_size * self.c_size, self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size, self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size, 1),
        )
        self.actor_fc = nn.Sequential(
            nn.Linear(self.c_size * self.c_size, self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size, self.c_size), nn.LeakyReLU(),
        )
        self._hidden_size = self.c_size

    def forward(self, inputs, rnn_hxs=None, masks=None):
        # inputs : (batch x (t x vehicle) x node x (node + 2 * node + features))
        adj, matrix_feature_enc = self.get_matrix_features(inputs)
        # adjacency : batch x node x node
        # matrix_feature_enc : batch x t x (1+vehicle) x node x f_size
        gnn_enc = self.get_gnn_encoding(adj, matrix_feature_enc)   # dim : batch x t x (node x f_size1)

        # LSTM encoding
        linear_features = self.gcn_to_lin_fc(gnn_enc)  # dim : batch x t x f_size2
        #linear_features2 = self.gcn_to_lin_fc2(ego_gnn_enc)  # dim : batch x t x f_size2
        lstm_out, cell = self.enc_lstm(linear_features)
        lstm_features = lstm_out[:,-1]  # dim : batch x f_size2

        # actor-critic network
        critic_outputs = self.critic_fc(lstm_features)
        actor_outputs = self.actor_fc(lstm_features)
        return critic_outputs, actor_outputs, rnn_hxs

    def predict(self, inputs, rnn_hxs=None, masks=None):
        # inputs : (batch x (t x vehicle) x node x (node + 2 * node + features))
        adj, matrix_feature_enc = self.get_matrix_features(inputs)
        # adjacency : batch x node x node
        # matrix_feature_enc : batch x t x (1+vehicle) x node x f_size
        gnn_enc = self.get_gnn_encoding(adj, matrix_feature_enc)  # dim : batch x t x (node x f_size1)

        # LSTM encoding
        linear_features = self.gcn_to_lin_fc(gnn_enc)  # dim : batch x t x f_size2
        # linear_features2 = self.gcn_to_lin_fc2(ego_gnn_enc)  # dim : batch x t x f_size2
        lstm_out, cell = self.enc_lstm(linear_features)
        lstm_features = lstm_out[:, -1]  # dim : batch x f_size2

        # decoding
        # USE a LSTM decoding for calculate P
        node_feature = inputs[:, :, :, 3 * self.node_size:]  # dim : batch x (t x vehicle) x node x features
        node_feature = node_feature.view(node_feature.size()[:1] + (self.obs_length, -1) + node_feature.size()[
                                                                                           -2:])  # dim : batch x t x vehicle x node x features
        p = node_feature[:, -1, :, :, -1:]  # dim : batch x vehicle x node x 1

        graph_feature = self.graph_feature_weight(gnn_enc.transpose(1, 2)).transpose(1,
                                                                                     2)  # dim : batch x c_size x (node x f_size1)
        graph_feature = graph_feature.view(
            [graph_feature.shape[0], self.c_size, self.node_size, -1])  # dim : batch x c_size x node x f_size

        output = []
        output2 = []
        for t in range(self.obs_length):
            p1 = self.gcn_module2(adj, p)  # dim : batch x c_size x node x f_size
            p2 = self.leaky_relu(p1 + graph_feature)
            p3 = self.gcn_module3(adj, p2)  # dim : batch x vehicle x node x f_size
            next_p = self.feature_decode_layer2(
                self.leaky_relu(p + self.feature_decode_layer1(p3)))  # dim : batch x vehicle x node x 1
            is_out_of_bound = self.feature_decode_layer3(p3.view(p3.shape[:2] + (-1,)))  # batch x vehicle x 1
            output.append(next_p.squeeze())
            output2.append(is_out_of_bound)
            p = next_p

        pred_outputs = torch.stack(output, dim=1)  # dim : batch x t x vehicle x node
        pred_outputs2 = torch.stack(output2, dim=1)  # dim : batch x t x vehicle x 1
        return pred_outputs, pred_outputs2

    def get_matrix_features(self, inputs):
        # inputs : (batch x (t x vehicle) x node x (node + 2 * node + features))
        inputs = inputs.squeeze()
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        # devide the inputs
        adj = inputs[:, 0, :, :self.node_size]  # -> not depends on time, vehicle      dim : batch x node x node

        edge_feature = inputs[:, 0, :, self.node_size:3 * self.node_size]  # -> not depends on time, vehicle
        edge_feature = edge_feature.view(edge_feature.size()[:-1] + (self.node_size, 2))   # dim : batch x node x node x 2
        edge_feature_enc = self.edge_feature_layer(edge_feature)  # dim : batch x node x node x f_size
        edge_feature_enc = edge_feature_enc * adj.unsqueeze(-1)
        edge_feature_enc = edge_feature_enc.sum(dim=1)  # dim : batch x node x f_size
        edge_feature_enc = edge_feature_enc.unsqueeze(1).repeat(1, self.obs_length, 1, 1).unsqueeze(2)  # dim : batch x t x 1 x node x f_size

        node_feature = inputs[:, :, :, 3 * self.node_size:]  # dim : batch x (t x vehicle) x node x features
        node_feature = node_feature.view(node_feature.size()[:1] + (self.obs_length, -1) + node_feature.size()[-2:]) # dim : batch x t x vehicle x node x features
        ego_node_feature_enc = self.ego_node_feature_layer(node_feature[:,:,:1,:,:])   # dim : batch x t x 1 x node x f_size
        node_feature_enc = self.node_feature_layer(node_feature[:,:,1:,:,:])   # dim : batch x t x num_ov x node x f_size
        node_feature_enc = torch.cat([ego_node_feature_enc, node_feature_enc], dim=2)   # dim : batch x t x vehicle x node x f_size

        matrix_feature_enc = torch.cat([edge_feature_enc, node_feature_enc],dim=2)  # dim : batch x t x (1+vehicle) x node x f_size

        return adj, matrix_feature_enc

    def get_gnn_encoding(self, adj, matrix_feature_enc):
        gnn_encoding = []
        for t in range(self.obs_length):
            input_features = matrix_feature_enc[:, t, :, :, :]  # dim : batch x (1 + vehicle) x node x f_size
            gcn_out = self.gcn_module1(adj, input_features) # dim : batch x (c_size) x node x f_size1
            gcn_out = gcn_out.sum(1)
            gcn_out = gcn_out.view(gcn_out.shape[:1] + (1, -1))   # dim : batch x 1 x (node x f_size1)
            gnn_encoding.append(gcn_out)
        gnn_encoding = torch.cat(gnn_encoding, dim=1)  # dim : batch x t x (node x f_size1)

        return gnn_encoding

    def sample(self, inputs):
        categorical, _, _ = self.forward(inputs)  # dim : (batch x n_gaussian), (batch x n_gaussian x output_dim)
        sampled_action = categorical.sample()

        return sampled_action

class Road_GNN_module(nn.Module):
    def __init__(self, in_feature_size, out_feature_size, c_size=4, stack_length=3, bias=True):
        super().__init__()
        self.in_feature_size = in_feature_size
        self.out_feature_size = out_feature_size
        self.c_size = 1
        self.stack_length = stack_length

        step = 0
        self.weight_linears = []
        #channel, stack
        for l in range(self.stack_length):
            weight_channel = []
            for i in range(self.c_size):
                wl = nn.Linear(self.in_feature_size, self.out_feature_size) if l==0 else nn.Linear(self.out_feature_size, self.out_feature_size)
                self.add_module('GCN_weight_'+str(step), wl)
                step += 1
                weight_channel.append(wl)
            self.weight_linears.append(weight_channel)

        stdv = 1. / math.sqrt(self.out_feature_size)
        if bias:
            self.bias = Parameter(torch.empty([self.c_size, 1, self.out_feature_size]))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.bias = None

        self.non_lin = nn.LeakyReLU()

    def forward(self, adj, x, add_loop=True):
        # adjacency tensor (batch x node x node)
        # x : input feature (batch x input_c_size x node x f_size)
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        x = x.unsqueeze(0) if x.dim() == 2 else x
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1

        #x = x.view([B * N, -1])
        for l in range(self.stack_length):
            x = [lin_layer(x).sum(1) for lin_layer in self.weight_linears[l]]    # lin_layer :batch x c_size x node x f_out, x : [batch x node x f_out] x c_out
            x = torch.stack(x, dim=1)    # dim : batch x c_out x node x f_out
            deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
            adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
            x = torch.stack([torch.matmul(adj, x[:,c]) for c in range(x.shape[1])], dim=1)  # dim : B x c_size x node x f_out

            if self.bias is not None:
                x = x + self.bias

            x = self.non_lin(x)

        return x

class GNN_MLP_Base(NNBase):
    def __init__(self, obs_shape, args, recurrent=False, hidden_size=512):
        super().__init__(recurrent, hidden_size, hidden_size)

        self.node_size = obs_shape[1]
        #self.in_feature_size = obs_shape[2] - obs_shape[1]

        self.args = args
        self.c_size = 8 #args.num_ov + 1
        self.f_size = 8
        self.act_dim = args.act_dim
        self.obs_length = args.prev_steps
        self.num_ov = args.num_ov

        self.edge_feature_size = 2
        self.node_feature_size = 5
        self.edge_feature_layer = nn.Sequential(nn.Linear(self.edge_feature_size, self.f_size), nn.LeakyReLU())
        self.node_feature_layer = nn.Sequential(nn.Linear(self.node_feature_size, self.f_size), nn.LeakyReLU())
        self.ego_node_feature_layer = nn.Sequential(nn.Linear(self.node_feature_size, self.f_size), nn.LeakyReLU())

        self.gcn_module1 = GNN_MLP_module(self.f_size, self.f_size, self.c_size)
        #self.gcn_module2 = GNN_MLP_module(1, self.f_size, self.c_size)
        self.gcn_to_lin_fc = nn.Sequential(
            nn.Linear(self.node_size * self.f_size, self.c_size * self.f_size), nn.LeakyReLU(),
        )
        #self.graph_feature_weight = nn.Linear(self.obs_length, self.c_size)

        self.enc_final = nn.Sequential(nn.Linear(self.obs_length * self.c_size * self.f_size, self.c_size * self.c_size), nn.LeakyReLU(),
                                      nn.Linear(self.c_size * self.c_size, self.c_size * self.c_size))

        #self.enc_lstm = nn.LSTM(self.c_size * self.f_size, self.c_size * self.c_size, batch_first=True, num_layers=2)
        # input : batch x time x in_feature_size
        # output : batch x time x out_feature_size
        # hidden, cell : num_layers x batch x out_feature_size
        # output, (hidden, cell) = lstm(input)

        # only for prediction
        #self.dec_lstm = nn.LSTM(self.c_size * self.f_size, self.c_size * self.c_size, batch_first=True, num_layers=2)
        #self.gcn_module3 = Road_GNN_module(self.f_size, self.f_size, 1+self.num_ov)
        #self.feature_decode_layer1 = nn.Sequential(nn.Linear(self.f_size, 1), nn.LeakyReLU())
        #self.feature_decode_layer2 = nn.Sequential(nn.Linear(1, 1))
        #self.feature_decode_layer3 = nn.Sequential(nn.Linear(self.node_size * self.f_size, self.node_size), nn.LeakyReLU(),
        #                                           nn.Linear(self.node_size, 1))

        self.leaky_relu = nn.LeakyReLU()

        self.critic_fc = nn.Sequential(
            nn.Linear(self.c_size * self.c_size, self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size, self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size, 1),
        )
        self.actor_fc = nn.Sequential(
            nn.Linear(self.c_size * self.c_size, self.c_size), nn.LeakyReLU(),
            nn.Linear(self.c_size, self.c_size), nn.LeakyReLU(),
        )
        self._hidden_size = self.c_size

    def forward(self, inputs, rnn_hxs=None, masks=None):
        # inputs : (batch x (t x vehicle) x node x (node + 2 * node + features))
        adj, matrix_feature_enc = self.get_matrix_features(inputs)
        # adjacency : batch x node x node
        # matrix_feature_enc : batch x t x (1+vehicle) x node x f_size
        gnn_enc = self.get_gnn_encoding(adj, matrix_feature_enc)  # dim : batch x t x (node x f_size1)

        # LSTM encoding
        linear_features = self.gcn_to_lin_fc(gnn_enc)  # dim : batch x t x f_size2
        # linear_features2 = self.gcn_to_lin_fc2(ego_gnn_enc)  # dim : batch x t x f_size2

        # lstm_out, cell = self.enc_lstm(linear_features)
        # lstm_features = lstm_out[:,-1]  # dim : batch x f_size2
        linear_features = linear_features.view([inputs.shape[0], -1])
        linear_features = self.enc_final(linear_features)

        # actor-critic network
        critic_outputs = self.critic_fc(linear_features)
        actor_outputs = self.actor_fc(linear_features)
        return critic_outputs, actor_outputs, rnn_hxs

    def predict(self, inputs, rnn_hxs=None, masks=None):
        # inputs : (batch x (t x vehicle) x node x (node + 2 * node + features))
        adj, matrix_feature_enc = self.get_matrix_features(inputs)
        # adjacency : batch x node x node
        # matrix_feature_enc : batch x t x (1+vehicle) x node x f_size
        gnn_enc = self.get_gnn_encoding(adj, matrix_feature_enc)  # dim : batch x t x (node x f_size1)

        # LSTM encoding
        linear_features = self.gcn_to_lin_fc(gnn_enc)  # dim : batch x t x f_size2
        # linear_features2 = self.gcn_to_lin_fc2(ego_gnn_enc)  # dim : batch x t x f_size2
        lstm_out, cell = self.enc_lstm(linear_features)
        lstm_features = lstm_out[:, -1]  # dim : batch x f_size2

        # decoding
        # USE a LSTM decoding for calculate P
        node_feature = inputs[:, :, :, 3 * self.node_size:]  # dim : batch x (t x vehicle) x node x features
        node_feature = node_feature.view(node_feature.size()[:1] + (self.obs_length, -1) + node_feature.size()[
                                                                                           -2:])  # dim : batch x t x vehicle x node x features
        p = node_feature[:, -1, :, :, -1:]  # dim : batch x vehicle x node x 1

        graph_feature = self.graph_feature_weight(gnn_enc.transpose(1, 2)).transpose(1,
                                                                                     2)  # dim : batch x c_size x (node x f_size1)
        graph_feature = graph_feature.view(
            [graph_feature.shape[0], self.c_size, self.node_size, -1])  # dim : batch x c_size x node x f_size

        output = []
        output2 = []
        for t in range(self.obs_length):
            p1 = self.gcn_module2(adj, p)  # dim : batch x c_size x node x f_size
            p2 = self.leaky_relu(p1 + graph_feature)
            p3 = self.gcn_module3(adj, p2)  # dim : batch x vehicle x node x f_size
            next_p = self.feature_decode_layer2(
                self.leaky_relu(p + self.feature_decode_layer1(p3)))  # dim : batch x vehicle x node x 1
            is_out_of_bound = self.feature_decode_layer3(p3.view(p3.shape[:2] + (-1,)))  # batch x vehicle x 1
            output.append(next_p.squeeze())
            output2.append(is_out_of_bound)
            p = next_p

        pred_outputs = torch.stack(output, dim=1)  # dim : batch x t x vehicle x node
        pred_outputs2 = torch.stack(output2, dim=1)  # dim : batch x t x vehicle x 1
        return pred_outputs, pred_outputs2

    def get_matrix_features(self, inputs):
        # inputs : (batch x (t x vehicle) x node x (node + 2 * node + features))
        inputs = inputs.squeeze()
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        # devide the inputs
        adj = inputs[:, 0, :, :self.node_size]  # -> not depends on time, vehicle      dim : batch x node x node

        edge_feature = inputs[:, 0, :, self.node_size:3 * self.node_size]  # -> not depends on time, vehicle
        edge_feature = edge_feature.view(edge_feature.size()[:-1] + (self.node_size, 2))   # dim : batch x node x node x 2
        edge_feature_enc = self.edge_feature_layer(edge_feature)  # dim : batch x node x node x f_size
        edge_feature_enc = edge_feature_enc * adj.unsqueeze(-1)
        edge_feature_enc = edge_feature_enc.sum(dim=1)  # dim : batch x node x f_size
        edge_feature_enc = edge_feature_enc.unsqueeze(1).repeat(1, self.obs_length, 1, 1).unsqueeze(2)  # dim : batch x t x 1 x node x f_size

        node_feature = inputs[:, :, :, 3 * self.node_size:]  # dim : batch x (t x vehicle) x node x features
        node_feature = node_feature.view(node_feature.size()[:1] + (self.obs_length, -1) + node_feature.size()[-2:]) # dim : batch x t x vehicle x node x features
        ego_node_feature_enc = self.ego_node_feature_layer(node_feature[:,:,:1,:,:])   # dim : batch x t x 1 x node x f_size
        node_feature_enc = self.node_feature_layer(node_feature[:,:,1:,:,:])   # dim : batch x t x num_ov x node x f_size
        node_feature_enc = torch.cat([ego_node_feature_enc, node_feature_enc], dim=2)   # dim : batch x t x vehicle x node x f_size

        matrix_feature_enc = torch.cat([edge_feature_enc, node_feature_enc],dim=2)  # dim : batch x t x (1+vehicle) x node x f_size

        return adj, matrix_feature_enc

    def get_gnn_encoding(self, adj, matrix_feature_enc):
        gnn_encoding = []
        for t in range(self.obs_length):
            input_features = matrix_feature_enc[:, t, :, :, :]  # dim : batch x (1 + vehicle) x node x f_size
            gcn_out = self.gcn_module1(adj, input_features) # dim : batch x (c_size) x node x f_size1
            gcn_out = gcn_out.sum(1)
            gcn_out = gcn_out.view(gcn_out.shape[:1] + (1, -1))   # dim : batch x 1 x (node x f_size1)
            gnn_encoding.append(gcn_out)
        gnn_encoding = torch.cat(gnn_encoding, dim=1)  # dim : batch x t x (node x f_size1)

        return gnn_encoding

    def sample(self, inputs):
        categorical, _, _ = self.forward(inputs)  # dim : (batch x n_gaussian), (batch x n_gaussian x output_dim)
        sampled_action = categorical.sample()

        return sampled_action

class GNN_MLP_module(nn.Module):
    def __init__(self, in_feature_size, out_feature_size, c_size=4, stack_length=3, bias=True):
        super().__init__()
        self.in_feature_size = in_feature_size
        self.out_feature_size = out_feature_size
        self.c_size = 1
        self.stack_length = stack_length

        step = 0
        self.weight_linears = []
        #channel, stack
        for l in range(self.stack_length):
            weight_channel = []
            for i in range(self.c_size):
                wl = nn.Linear(self.in_feature_size, self.out_feature_size) if l==0 else nn.Linear(self.out_feature_size, self.out_feature_size)
                self.add_module('GCN_weight_'+str(step), wl)
                step += 1
                weight_channel.append(wl)
            self.weight_linears.append(weight_channel)

        stdv = 1. / math.sqrt(self.out_feature_size)
        if bias:
            self.bias = Parameter(torch.empty([self.c_size, 1, self.out_feature_size]))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.bias = None

        self.non_lin = nn.LeakyReLU()

    def forward(self, adj, x, add_loop=True):
        # adjacency tensor (batch x node x node)
        # x : input feature (batch x input_c_size x node x f_size)
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        x = x.unsqueeze(0) if x.dim() == 2 else x
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1

        #x = x.view([B * N, -1])
        for l in range(self.stack_length):
            x = [lin_layer(x).sum(1) for lin_layer in self.weight_linears[l]]    # lin_layer :batch x c_size x node x f_out, x : [batch x node x f_out] x c_out
            x = torch.stack(x, dim=1)    # dim : batch x c_out x node x f_out
            deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
            adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
            x = torch.stack([torch.matmul(adj, x[:,c]) for c in range(x.shape[1])], dim=1)  # dim : B x c_size x node x f_out

            if self.bias is not None:
                x = x + self.bias

            x = self.non_lin(x)

        return x


class GNNBase(NNBase):
    def __init__(self, obs_shape, args, recurrent=False, hidden_size=512):
        super().__init__(recurrent, hidden_size, hidden_size)

        self.args = args
        self.node_size = obs_shape[1]
        #self.in_feature_size = obs_shape[2] - obs_shape[1]

        self.c_size = args.num_ov + 1
        self.f_size = 16
        self.act_dim = args.act_dim
        self.obs_length = 10 #args.prev_steps
        self.num_ov = args.num_ov

        self.edge_feature_size = 2
        self.node_feature_size = 5
        self.edge_feature_layer = nn.Sequential(nn.Linear(self.edge_feature_size, self.f_size), nn.LeakyReLU())
        self.node_feature_layer = nn.Sequential(nn.Linear(self.node_feature_size, self.f_size), nn.LeakyReLU())

        self.gcn_module1 = GNN_module(self.f_size, self.f_size, self.c_size)
        self.gcn_module2 = GNN_module(1, self.f_size, self.c_size, stack_length=1)
        self.gcn_module3 = GNN_module(self.f_size, self.f_size, 1+self.num_ov, stack_length=2)
        self.feature_decode_layer1 = nn.Sequential(nn.Linear(self.f_size, 1), nn.LeakyReLU())
        self.feature_decode_layer2 = nn.Linear(1, 1)

        # dim : batch x (1 + vehicle) x node x c_size
        self.critic_fc = nn.Sequential(
            nn.Linear((1+self.num_ov) * self.node_size * self.f_size, self.node_size * self.f_size), nn.LeakyReLU(),
            nn.Linear(self.node_size * self.f_size, self.f_size), nn.LeakyReLU(),
            nn.Linear(self.f_size, 1),
        )
        self.actor_fc = nn.Sequential(
            nn.Linear((1+self.num_ov) * self.node_size * self.f_size, self.node_size * self.f_size), nn.LeakyReLU(),
            nn.Linear(self.node_size * self.f_size, self.f_size), nn.LeakyReLU(),
        )

        self.leaky_relu = nn.LeakyReLU()
        self._hidden_size = self.f_size

    def forward(self, inputs, rnn_hxs=None, masks=None):
        # inputs : (batch x (t x vehicle) x node x (node + 2 * node + features))
        graph_features = self.encode(inputs)
        linear_features = graph_features.view(graph_features.size(0), -1)

        critic_outputs = self.critic_fc(linear_features)
        actor_outputs = self.actor_fc(linear_features)

        if self.args.use_pred_net:
            prediction = self.decode(inputs, graph_features)
            return critic_outputs, actor_outputs, prediction
        else:
            return critic_outputs, actor_outputs, rnn_hxs

    def encode(self, inputs):
        # inputs : (batch x (t x vehicle) x node x (node + 2 * node + features))
        inputs = inputs.squeeze()
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        # devide the inputs
        adj = inputs[:, 0, :, :self.node_size]   # -> not depends on time, vehicle
        edge_feature = inputs[:, 0, :, self.node_size:3*self.node_size]     # -> not depends on time, vehicle
        edge_feature = edge_feature.view(edge_feature.size()[:-1] + (self.node_size, 2))
        node_feature = inputs[:, :, :, 3 * self.node_size:]    # dim : batch x (t x vehicle) x node x features
        node_feature = node_feature.view(node_feature.size()[:1]+(self.obs_length,-1)+node_feature.size()[-2:])
        # adjacency tensor(batch x node x node)
        # edge feature tensor(batch x node x node x 2)
        # node feature : input feature (batch x t x vehicle x node x features)

        # encode features
        edge_feature_enc = self.edge_feature_layer(edge_feature)  # dim : batch x node x node x f_size
        edge_feature_enc = edge_feature_enc * adj.unsqueeze(-1)
        edge_feature_enc = edge_feature_enc.sum(dim=1)   # dim : batch x node x f_size
        node_feature_enc = self.node_feature_layer(node_feature)  # dim : batch x t x vehicle x node x f_size

        input_features = torch.cat([edge_feature_enc.unsqueeze(1), node_feature_enc[:, 0, :, :, :]],
                                   dim=1)  # dim : batch x (1 + vehicle) x node x f_size

        out = torch.zeros_like(node_feature_enc[:,0,:,:,:])
        for t in range(self.obs_length):
            input_features = torch.cat([edge_feature_enc.unsqueeze(1), node_feature_enc[:, t, :, :, :]],dim=1)  # dim : batch x (1 + vehicle) x node x f_size
            out += self.gcn_module1(adj, input_features)   # dim : batch x c_size x node x f_size

        return out  # dim : batch x c_size x node x f_size

    def decode(self, inputs, graph_feature):
        # inputs : (batch x (t x vehicle) x node x (node + 2 * node + features))
        inputs = inputs.squeeze()
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        # devide the inputs
        adj = inputs[:, 0, :, :self.node_size]   # -> not depends on time, vehicle
        node_feature = inputs[:, :, :, 3 * self.node_size:]
        node_feature = node_feature.view(node_feature.size()[:1]+(self.obs_length,-1)+node_feature.size()[-2:])
        p = node_feature[:,-1,:,:,:1]  # dim : batch x vehicle x node x 1
        # adjacency tensor(batch x node x node)
        # node feature : input feature (batch x t x vehicle x node x features)
        # graph feature : batch x c_size x node x f_size

        # decode features
        output = []
        for t in range(self.obs_length):
            p1 = self.gcn_module2(adj, p)   # dim : batch x c_size x node x f_size
            p2 = self.leaky_relu(p1 + graph_feature)
            p3 = self.gcn_module3(p2)   # dim : batch x vehicle x node x f_size
            next_p = self.feature_decode_layer2(self.leaky_relu(p + self.feature_decode_layer1(p3)))
            output.append(next_p.squeeze())
            p = next_p

        output = torch.stack(output, dim=1) # dim : batch x t x vehicle x node

        return output  # dim : batch x c_size x node x f_size

    def sample(self, inputs):
        categorical, _, _ = self.forward(inputs)  # dim : (batch x n_gaussian), (batch x n_gaussian x output_dim)
        sampled_action = categorical.sample()

        return sampled_action

class GNN_module(nn.Module):
    def __init__(self, in_feature_size, out_feature_size, c_size, stack_length, bias=True):
        super().__init__()
        self.in_feature_size = in_feature_size
        self.out_feature_size = out_feature_size
        self.c_size = c_size
        self.stack_length = stack_length
        self.weight_linears = [[nn.Linear(self.in_feature_size, self.out_feature_size) for i in range(self.c_size)] for l in range(self.stack_length)]

        stdv = 1. / math.sqrt(self.out_feature_size)
        if bias:
            self.bias = Parameter(torch.empty([self.c_size, 1, self.out_feature_size]))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.bias = None

        self.non_lin = nn.LeakyReLU()

    def forward(self, adj, x, add_loop=True):
        # adjacency tensor (batch x node x node)
        # x : input feature (batch x c_size x node x f_size)
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        x = x.unsqueeze(0) if x.dim() == 2 else x
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1

        #x = x.view([B * N, -1])
        for l in range(self.stack_length):
            x = [lin_layer(x).sum(1) for lin_layer in self.weight_linears[l]]    # lin_layer :batch x c_size x node x f_out, x : [batch x node x f_out] x c_out
            x = torch.stack(x, dim=1)    # dim : batch x c_out x node x f_out
            deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
            adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
            x = torch.stack([torch.matmul(adj, x[:,c]) for c in range(x.shape[1])], dim=1)  # dim : B x c_size x node x f_out

            if self.bias is not None:
                x = x + self.bias

            x = self.non_lin(x)

        return x