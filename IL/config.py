import os

class GlobalConfig:
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

            self.root_dir = self.data_dir
            self.train_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town06', 'Town07']
            self.val_towns = ['Town05']
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                self.train_data.append(os.path.join(self.root_dir, town + '_tiny'))
                self.train_data.append(os.path.join(self.root_dir, town + '_short'))
                if town != 'Town07':
                    self.train_data.append(os.path.join(self.root_dir, town + '_long'))
            for town in self.val_towns:
                self.val_data.append(os.path.join(self.root_dir, town + '_short'))

            # visualizing transformer attention maps
            self.viz_root = self.data_dir
            self.viz_towns = ['Town05_short']
            self.viz_data = []
            for town in self.viz_towns:
                self.viz_data.append(os.path.join(self.viz_root, town))
