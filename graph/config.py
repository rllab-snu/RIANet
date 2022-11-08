import os

class GlobalConfig:
    """ base architecture configurations """
	# Data
    x_max = 24
    y_max = 48
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
