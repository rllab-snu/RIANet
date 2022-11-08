import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, WeightedRandomSampler

class DQN():
    def __init__(self,
                 q_net,
                 target_net,
                 gamma,
                 mini_batch_size,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.q_net = q_net
        self.target_net = target_net

        self.gamma = gamma
        self.mini_batch_size = mini_batch_size

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(q_net.parameters(), lr=lr, eps=eps)

        self.memory_charged = 0

    def update(self, rollouts):

        q_loss_epoch = 0

        num_steps, num_processes = rollouts.rewards.size()[0:2]
        memory_size = num_processes * num_steps

        self.memory_charged += 1

        memory_current_size = min(self.memory_charged, memory_size)

        if memory_current_size < self.mini_batch_size:
            return 0.0

        '''
        sampler = BatchSampler(
            SubsetRandomSampler(range(memory_current_size)),
            self.mini_batch_size,
            drop_last=True)
        '''

        indices = random.sample(range(memory_current_size), self.mini_batch_size)
        obs_batch = rollouts.obs[:-1].view(-1, *rollouts.obs.size()[2:])[indices]
        next_obs_batch = rollouts.obs[1:].view(-1, *rollouts.obs.size()[2:])[indices]

        actions_batch = rollouts.actions.view(-1, rollouts.actions.size(-1))[indices]
        rewards_batch = rollouts.rewards.view(-1, 1)[indices]
        masks_batch = rollouts.masks[1:].view(-1, 1)[indices]  # mask for terminal condition

        # Reshape to do in a single forward pass for all steps
        q_values = self.q_net(obs_batch).gather(1, actions_batch.long())
        next_q_values = self.target_net(next_obs_batch).max(1)[0].view([-1, 1]).detach()

        expected_q_values = masks_batch * next_q_values * self.gamma + rewards_batch

        loss = 0.5 * (expected_q_values - q_values).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(),
                                 self.max_grad_norm)
        self.optimizer.step()

        q_loss_epoch += loss.item()

        return q_loss_epoch