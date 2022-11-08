import argparse
import json
import os
import shutil
import importlib
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

from data_loader import CARLA_Data
from config import GlobalConfig

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='road_gac', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
parser.add_argument('--epochs', type=int, default=101, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--val-every', type=int, default=10, help='Validation frequency (epochs).')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--seed', type=int, default=1, help='Random train seed')
parser.add_argument('--logdir', type=str, default='./trained_models', help='Directory to log data to.')
parser.add_argument('--model-name', type=str, default='sat2graph', help='Directory to log data to.')
parser.add_argument('--data-dir', type=str, default='./data', help='Directory to dataset')

parser.add_argument("--use-bev-3", default=False, action='store_true', help="use bev-3 as input")
parser.add_argument("--use-bev", default=False, action='store_true', help="use bev as input")
parser.add_argument("--use-sides", default=False, action='store_true', help="use sides")
parser.add_argument("--use-rear", default=False, action='store_true', help="use rear")
parser.add_argument("--use-lidar", default=False, action='store_true', help="use lidar")

args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.id)
args.ignore_sides = not args.use_sides
args.ignore_rear = not args.use_rear

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

config = GlobalConfig(data_dir=args.data_dir)
writer = SummaryWriter(log_dir=args.logdir)

model_config = importlib.import_module('models.' + args.model_name).ModelConfig()
for c in dir(model_config):
    if not c.startswith('_'):
        setattr(config, c, getattr(model_config, c))

for c in dir(args):
    if not c.startswith('_'):
        setattr(config, c, getattr(args, c))

def bce_loss_with_logits(pred, target):
    pred = torch.sigmoid(pred)
    loss = -(target * torch.log(pred + 1e-6) + (1 - target) * torch.log(1 - pred + 1e-6))
    return loss

def graph_loss_with_logits(pred, target):
    field_idx = 1 + 3 * torch.arange(6)
    vertex_gt = target[:,0]
    edge_gt = target[:,field_idx]

    vertex_ce_loss = bce_loss_with_logits(pred[:,0], vertex_gt)
    edge_ce_loss = bce_loss_with_logits(pred[:,field_idx], edge_gt)
    l2_loss = torch.sqrt((pred[:,field_idx+1] - target[:,field_idx+1])**2 + \
              (pred[:,field_idx+2] - target[:,field_idx+2])**2)

    loss = vertex_ce_loss + vertex_gt * torch.sum(edge_gt * (edge_ce_loss + l2_loss), dim=1)
    return loss

class Trainer(object):
    """Trainer that runs training and inference.
    Args
        - cur_epoch (int): Current epoch.
        - print_every (int): How frequently (# batches) to print loss.
        - validate_every (int): How frequently (# epochs) to run validation.

    """

    def __init__(self, cur_epoch=0, cur_iter=0):
        self.cur_epoch = cur_epoch
        self.cur_iter = cur_iter
        self.bestval_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10

    def train(self):
        loss_epoch = 0.
        num_batches = 0
        model.train()

        # Train loop
        for data in tqdm(dataloader_train):

            # efficiently zero gradients
            for p in model.parameters():
                p.grad = None

            # create batch and move to GPU
            fronts_in = data['fronts']
            lefts_in = data['lefts']
            rights_in = data['rights']
            lidars_in = data['lidars']

            fronts = []
            lefts = []
            rights = []
            lidars = []
            fronts.append(fronts_in[0].to(args.device, dtype=torch.float32))
            if not config.ignore_sides:
                lefts.append(lefts_in[0].to(args.device, dtype=torch.float32))
                rights.append(rights_in[0].to(args.device, dtype=torch.float32))
            lidars.append(lidars_in[0].to(args.device, dtype=torch.float32))

            data_input = {}
            data_input['images'] = fronts + lefts + rights
            data_input['lidars'] = lidars

            # loss
            pred_graph_feature = model(data_input)
            target_graph_feature = data['graph_features'][0].to(args.device, dtype=torch.float32)

            loss = graph_loss_with_logits(pred_graph_feature, target_graph_feature).mean()
            loss.backward()
            loss_epoch += float(loss.item())

            num_batches += 1
            optimizer.step()

            writer.add_scalar('train_loss', loss.item(), self.cur_iter)
            self.cur_iter += 1

        loss_epoch = loss_epoch / num_batches
        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def validate(self):
        model.eval()

        with torch.no_grad():
            num_batches = 0
            wp_epoch = 0.

            # Validation loop
            for batch_num, data in enumerate(tqdm(dataloader_val), 0):
                # create batch and move to GPU
                fronts_in = data['fronts']
                lefts_in = data['lefts']
                rights_in = data['rights']
                lidars_in = data['lidars']

                fronts = []
                lefts = []
                rights = []
                lidars = []
                fronts.append(fronts_in[0].to(args.device, dtype=torch.float32))
                if not config.ignore_sides:
                    lefts.append(lefts_in[0].to(args.device, dtype=torch.float32))
                    rights.append(rights_in[0].to(args.device, dtype=torch.float32))
                lidars.append(lidars_in[0].to(args.device, dtype=torch.float32))

                data_input = {}
                data_input['images'] = fronts + lefts + rights
                data_input['lidars'] = lidars

                # loss
                pred_graph_feature = model(data_input)
                target_graph_feature = data['graph_features'][0].to(args.device, dtype=torch.float32)

                wp_epoch += float(graph_loss_with_logits(pred_graph_feature, target_graph_feature).mean())
                num_batches += 1

            wp_loss = wp_epoch / float(num_batches)
            tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Wp: {wp_loss:3.3f}')

            writer.add_scalar('val_loss', wp_loss, self.cur_epoch)

            self.val_loss.append(wp_loss)

    def save(self):

        save_best = False
        if self.val_loss[-1] <= self.bestval:
            self.bestval = self.val_loss[-1]
            self.bestval_epoch = self.cur_epoch
            save_best = True

        # Create a dictionary of all data to save
        log_table = {
            'epoch': self.cur_epoch,
            'iter': self.cur_iter,
            'bestval': self.bestval,
            'bestval_epoch': self.bestval_epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
        }

        # Save ckpt for every epoch
        torch.save(model.state_dict(), os.path.join(args.logdir, 'model_%d.pth' % self.cur_epoch))

        # Save the recent model/optimizer states
        torch.save(model.state_dict(), os.path.join(args.logdir, 'model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))

        # Log other data corresponding to the recent model
        with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))

        tqdm.write('====== Saved recent model ======>')

        if save_best:
            torch.save(model.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
            tqdm.write('====== Overwrote best model ======>')


# Data
train_set = CARLA_Data(root=config.train_data, config=config)
val_set = CARLA_Data(root=config.val_data, config=config)

dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Model
model = importlib.import_module('models.' + args.model_name).Model(args, args.device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
trainer = Trainer()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Total trainable parameters: ', params)

# Create logdir
if not os.path.isdir(args.logdir):
    os.makedirs(args.logdir)
    print('Created dir:', args.logdir)
elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
    print('Loading checkpoint from ' + args.logdir)
    with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
        log_table = json.load(f)

    # Load variables
    trainer.cur_epoch = log_table['epoch']
    if 'iter' in log_table: trainer.cur_iter = log_table['iter']
    trainer.bestval = log_table['bestval']
    trainer.train_loss = log_table['train_loss']
    trainer.val_loss = log_table['val_loss']

    # Load checkpoint
    model.load_state_dict(torch.load(os.path.join(args.logdir, 'model.pth')))
    optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'recent_optim.pth')))

# Log args
with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

shutil.copy('config.py', os.path.join(args.logdir, 'config.py'))

for epoch in range(trainer.cur_epoch, args.epochs):
    trainer.train()
    if epoch % args.val_every == 0:
        trainer.validate()
        trainer.save()
