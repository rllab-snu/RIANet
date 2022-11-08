import argparse
import json
import os
import importlib
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from data_baseline import CARLA_Data
from config import GlobalConfig

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='road_gac', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
parser.add_argument('--epochs', type=int, default=101, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--val-every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--seed', type=int, default=1, help='Random train seed')
parser.add_argument('--logdir', type=str, default='./trained_models', help='Directory to log data to.')
parser.add_argument('--model-name', type=str, default='Road_GAC', help='Directory to log data to.')
parser.add_argument('--data-dir', type=str, default='./data', help='Directory to dataset')
parser.add_argument('--use-2d-detection', type=int, default=0, help='0: without detection, 1: pre-detected, 2: gt detection')
parser.add_argument('--use-3d-detection', type=int, default=0, help='0: without detection, 1: pre-detected, 2: gt detection')

args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.id)

assert args.use_2d_detection == 0 or args.use_3d_detection == 0

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

config = GlobalConfig(data_dir=args.data_dir)
writer = SummaryWriter(log_dir=args.logdir)

class Trainer(object):
	"""Trainer that runs training and inference.
	Args
		- cur_epoch (int): Current epoch.
		- print_every (int): How frequently (# batches) to print loss.
		- validate_every (int): How frequently (# epochs) to run validation.
		
	"""

	def __init__(self,  cur_epoch=0, cur_iter=0):
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
			rears_in = data['rears']
			lidars_in = data['lidars']
			fronts = []
			lefts = []
			rights = []
			rears = []
			lidars = []
			for i in range(config.seq_len):
				fronts.append(fronts_in[i].to(args.device, dtype=torch.float32))
				if not config.ignore_sides:
					lefts.append(lefts_in[i].to(args.device, dtype=torch.float32))
					rights.append(rights_in[i].to(args.device, dtype=torch.float32))
				if not config.ignore_rear:
					rears.append(rears_in[i].to(args.device, dtype=torch.float32))
				lidars.append(lidars_in[i].to(args.device, dtype=torch.float32))

			target_point = torch.stack(data['target_point'], dim=1).to(args.device, dtype=torch.float32)

			data_input = {}
			data_input['images'] = fronts+lefts+rights+rears
			data_input['lidars'] = lidars
			data_input['target_point'] = target_point
			data_input['command'] = data['command'].to(args.device)
			data_input['velocity'] = data['velocity'].to(args.device, dtype=torch.float32)
			data_input['adjacency_matrix'] = data['adjacency_matrix'].to(args.device, dtype=torch.float32)
			data_input['node_feature_matrix'] = data['node_feature_matrix'].to(args.device, dtype=torch.float32)
			data_input['edge_feature_matrix'] = data['edge_feature_matrix'].to(args.device, dtype=torch.float32)
			data_input['node_num'] = data['node_num'].to(args.device, dtype=torch.int64)
			if args.use_2d_detection > 0:
				data_input['td_img_class'] = data['td_img_class'].to(args.device, dtype=torch.float32)
				data_input['td_img_mask'] = data['td_img_mask'].to(args.device, dtype=torch.float32)
				data_input['td_num'] = data['td_num'].to(args.device, dtype=torch.int64)
				data_input['td_img_mask_info'] = data['td_img_mask_info'].to(args.device, dtype=torch.float32)
				data_input['od_img_class'] = data['od_img_class'].to(args.device, dtype=torch.float32)
				data_input['od_img_mask'] = data['od_img_mask'].to(args.device, dtype=torch.float32)
				data_input['od_num'] = data['od_num'].to(args.device, dtype=torch.int64)
				data_input['od_img_mask_info'] = data['od_img_mask_info'].to(args.device, dtype=torch.float32)

			# target point
			if hasattr(model.config, "pred_target_point") and model.config.pred_target_point:
				pred_wp = model(data_input)
				gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i
								in range(1, len(data['waypoints']))]
				gt_waypoints = torch.stack(gt_waypoints, dim=1).to(args.device, dtype=torch.float32)
				loss = F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean()
				loss.backward()
				loss_epoch += float(loss.item())
			elif args.model_name.startswith("cilrs"):
				steer, throttle, brake, velocity = model(data_input)
				gt_velocity = data['velocity'].to(args.device, dtype=torch.float32)
				gt_steer = data['steer'].to(args.device, dtype=torch.float32)
				gt_throttle = data['throttle'].to(args.device, dtype=torch.float32)
				gt_brake = data['brake'].to(args.device, dtype=torch.float32)
				loss = 0.05 * F.l1_loss(velocity.squeeze(), gt_velocity).mean()
				loss += F.l1_loss(steer.squeeze(), gt_steer.squeeze()).mean()
				loss += F.l1_loss(throttle.squeeze(), gt_throttle.squeeze()).mean()
				loss += F.l1_loss(brake.squeeze(), gt_brake.squeeze()).mean()
				loss.backward()
				loss_epoch += float(loss.item())
			else:
				# target control
				target_control = data['target_speed'].to(args.device, dtype=torch.float32).unsqueeze(-1)
				pred_control = model(data_input)
				loss = F.l1_loss(pred_control, target_control, reduction='none').mean()
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
				rears_in = data['rears']
				lidars_in = data['lidars']
				fronts = []
				lefts = []
				rights = []
				rears = []
				lidars = []
				for i in range(config.seq_len):
					fronts.append(fronts_in[i].to(args.device, dtype=torch.float32))
					if not config.ignore_sides:
						lefts.append(lefts_in[i].to(args.device, dtype=torch.float32))
						rights.append(rights_in[i].to(args.device, dtype=torch.float32))
					if not config.ignore_rear:
						rears.append(rears_in[i].to(args.device, dtype=torch.float32))
					lidars.append(lidars_in[i].to(args.device, dtype=torch.float32))

				target_point = torch.stack(data['target_point'], dim=1).to(args.device, dtype=torch.float32)

				data_input = {}
				data_input['images'] = fronts + lefts + rights + rears
				data_input['lidars'] = lidars
				data_input['target_point'] = target_point
				data_input['command'] = data['command'].to(args.device)
				data_input['velocity'] = data['velocity'].to(args.device, dtype=torch.float32)
				data_input['adjacency_matrix'] = data['adjacency_matrix'].to(args.device, dtype=torch.float32)
				data_input['node_feature_matrix'] = data['node_feature_matrix'].to(args.device, dtype=torch.float32)
				data_input['edge_feature_matrix'] = data['edge_feature_matrix'].to(args.device, dtype=torch.float32)
				data_input['node_num'] = data['node_num'].to(args.device, dtype=torch.int64)
				if args.use_2d_detection > 0:
					data_input['td_img_class'] = data['td_img_class'].to(args.device, dtype=torch.float32)
					data_input['td_img_mask'] = data['td_img_mask'].to(args.device, dtype=torch.float32)
					data_input['td_num'] = data['td_num'].to(args.device, dtype=torch.int64)
					data_input['td_img_mask_info'] = data['td_img_mask_info'].to(args.device, dtype=torch.float32)
					data_input['od_img_class'] = data['od_img_class'].to(args.device, dtype=torch.float32)
					data_input['od_img_mask'] = data['od_img_mask'].to(args.device, dtype=torch.float32)
					data_input['od_num'] = data['od_num'].to(args.device, dtype=torch.int64)
					data_input['od_img_mask_info'] = data['od_img_mask_info'].to(args.device, dtype=torch.float32)

				if hasattr(model.config, "pred_target_point") and model.config.pred_target_point:
					pred_wp = model(data_input)
					gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i
									in range(1, len(data['waypoints']))]
					gt_waypoints = torch.stack(gt_waypoints, dim=1).to(args.device, dtype=torch.float32)
					wp_epoch += float(F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean())
				elif args.model_name.startswith("cilrs"):
					steer, throttle, brake, velocity = model(data_input)
					gt_velocity = data['velocity'].to(args.device, dtype=torch.float32)
					gt_steer = data['steer'].to(args.device, dtype=torch.float32)
					gt_throttle = data['throttle'].to(args.device, dtype=torch.float32)
					gt_brake = data['brake'].to(args.device, dtype=torch.float32)
					loss = 0.05 * F.l1_loss(velocity.squeeze(), gt_velocity).mean()
					loss += F.l1_loss(steer.squeeze(), gt_steer.squeeze()).mean()
					loss += F.l1_loss(throttle.squeeze(), gt_throttle.squeeze()).mean()
					loss += F.l1_loss(brake.squeeze(), gt_brake.squeeze()).mean()
					wp_epoch = float(loss)
				else:
					# target control
					target_control = data['target_speed'].to(args.device, dtype=torch.float32).unsqueeze(-1)
					pred_control = model(data_input)
					wp_epoch += float(F.l1_loss(pred_control, target_control, reduction='none').mean())

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
		torch.save(model.state_dict(), os.path.join(args.logdir, 'model_%d.pth'%self.cur_epoch))

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
train_set = CARLA_Data(root=config.train_data, config=config, args=args)
val_set = CARLA_Data(root=config.val_data, config=config, args=args)

dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Model
model = importlib.import_module('models.'+args.model_name).Model(args, args.device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
trainer = Trainer()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('Total trainable parameters: ', params)

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

for epoch in range(trainer.cur_epoch, args.epochs): 
	trainer.train()
	if epoch % args.val_every == 0: 
		trainer.validate()
		trainer.save()
