import os
import random
import pickle
import torch
import numpy as np
from PIL import Image
from glob import glob
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F


class UWA3DDataset(Dataset):
    '''Missing Modality in Multi-modal Action Recognition Dataset'''

    def __init__(self, phase='train', img_size=224, num_frames=32,
                 modality=['rgb', 'depth'], temporal_augmentation=True,
                 uwa3d_dir='/dataset/uwa3d/',
                 **kwargs):
        '''
        phase: phase in ['train' , 'val']
        num_frames: number of input frames (i.e., sequence length) to transformer encoder
        modality: modalities to use (e.g., rgb, depth, skeleton, ir)
        uwa3d_dir: root directory for dataset
        '''
        super(UWA3DDataset).__init__()
        assert phase in ['train', 'val', 'test'], f'{phase} is not available. should be one of train/val.'
        self.phase = phase
        self.img_size = img_size
        self.num_frames = num_frames
        self.root = uwa3d_dir

        # non-overlapping videos
        EXCLUDE = [
            'a21_s02_v04_e01',
            'a06_s06_v04_e01',
            'a02_s06_v03_e01',
            'a07_s02_v02_e01',
            'a16_s06_v02_e01'
        ]

        if self.phase == 'train':
            IDS = [3, 4]
        else:
            IDS = [1, 2]

        self.modality = modality
        self.temporal_augmentation = temporal_augmentation and self.phase=='train'
        self.data_dir = defaultdict(list)

        if 'rgb' in self.modality:
            for dir1 in sorted(glob(os.path.join(self.root, 'UWA3DII_RGB', 'UWA3D-RGB', '*'))):  # a01_s01_v01_e00
                if int(dir1.split('/')[-1].split('_')[2][1:]) in IDS:
                    # exclude non-overlapping videos
                    if 'rgb' in self.modality and 'depth' in self.modality:
                        if dir1.split('/')[-1] in EXCLUDE:
                            continue
                    self.data_dir['rgb'].append(dir1)

        if 'depth' in self.modality:
            for dir1 in sorted(glob(os.path.join(self.root, 'UWA3DII_Depth_ROI', 'UWA3D-Depth', '*'))):  # a01_s01_v01_e00
                if int(dir1.split('/')[-1].split('_')[2][1:]) in IDS:
                    # exclude non-overlapping videos
                    if 'rgb' in self.modality and 'depth' in self.modality:
                        if dir1.split('/')[-1] in EXCLUDE:
                            continue
                    self.data_dir['depth'].append(dir1)

        if phase == 'train':
            self.augmenters = {
                'rgb': T.Compose([
                            T.Resize((self.img_size, self.img_size)),
                            T.ColorJitter(brightness=.5, hue=.3),
                            T.RandomHorizontalFlip(p=0.5),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]),
                'depth': T.Compose([
                            T.Resize((self.img_size, self.img_size)),
                            T.RandomHorizontalFlip(p=0.5),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]),
            }
        else:  # test
            self.augmenters = {
                'rgb': T.Compose([
                            T.Resize((self.img_size, self.img_size)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]),
                'depth': T.Compose([
                            T.Resize((self.img_size, self.img_size)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]),
            }

        # rgb_dirs = {rgb_dir.split('/')[-1][:-4] for rgb_dir in self.data_dir['rgb']}
        # depth_dirs = {depth_dir.split('/')[-1][:-4] for depth_dir in self.data_dir['depth']}
        # exclude_r = rgb_dirs - depth_dirs
        # exclude_d = depth_dirs - rgb_dirs
        # print('r-d', exclude_r)
        # print('d-r', exclude_d)

        # sanity check
        # print(f'{phase} dataset configurations')
        # rgb_len = len(self.data_dir['rgb'])
        # depth_len = len(self.data_dir['depth'])
        # print(rgb_len)
        # print(depth_len)
        # assert rgb_len == depth_len, f'rgb:{rgb_len}, depth:{depth_len}'

    def __len__(self):
        ''' number of total data '''
        return len(list(self.data_dir.values())[0])

    def __getitem__(self, idx):
        model_inputs = {}
        modality = self.modality

        video_len = len(os.listdir(self.data_dir[modality[0]][idx]))

        # sanity check
        # depth_len = len(os.listdir(self.data_dir[modality[1]][idx]))
        # assert video_len == depth_len, f'rgb:{video_len}, depth:{depth_len}'

        idxs = list(range(video_len))
        stepsize = video_len / self.num_frames

        if self.temporal_augmentation:
            sampled_idxs = idxs if video_len < self.num_frames else sorted(random.sample(idxs, self.num_frames))
        else:  # uniform sample
            sampled_idxs = idxs if video_len < self.num_frames else [idxs[round(stepsize*i)] for i in range(self.num_frames)]

            # if 'skeleton' in self.modality:  
            #     for dir2 in sorted(glob(os.path.join(dir1, '*_skeleton.txt'))):  # TODO: try '*_depth.png'
            #         self.data_dir['depth'].append(dir2)

        if 'rgb' in modality:
            video_rgb = []
            for i, path in enumerate(sorted(glob(os.path.join(self.data_dir['rgb'][idx], '*')))):  # 001,.jpg
                if i not in sampled_idxs:
                    continue
                try:
                    rgb = Image.open(path).convert('RGB')
                except:
                    print(f'unable to open RGB from {path}.')
                    continue
                else:
                    # rgb = F.crop(rgb, top=16 , left=48 , height=224, width=224)
                    rgb = F.crop(rgb, top=64 , left=72 , height=176, width=176)  # CxHxW
                    rgb = self.augmenters['rgb'](rgb)  # CxHxW
                    video_rgb.append(rgb)
            video_rgb = torch.stack(video_rgb)  # TxCxHxW
            model_inputs['rgb'] = video_rgb

        if 'depth' in modality:
            video_depth = []
            # TODO: try '*_depth.png'
            for i, path in enumerate(sorted(glob(os.path.join(self.data_dir['depth'][idx], '*')))):  # 001.png
                if i not in sampled_idxs:
                    continue
                try:
                    depth = Image.open(path).convert('RGB')

                    # gray map to color map using cm.jet
                    # this diminishes the number of pixel values
                    # depth = np.array(Image.open(path))
                    # depth = cm.jet((depth - depth.min())/ (depth.max() - depth.min()))[:,:,:3]
                    # depth = Image.fromarray((depth*255).astype(np.uint8))
                except:
                    print(f'unable to open Depth from {path}.')
                    continue
                else:
                    depth = self.augmenters['depth'](depth)  # CxHxW
                    video_depth.append(depth)
            video_depth = torch.stack(video_depth)  # TxCxHxW
            model_inputs['depth'] = video_depth

        # a01_s01_v01_e01
        target_class = int(self.data_dir[modality[0]][idx].split('/')[-1].split('_')[0][1:]) - 1
        
        return dict(model_inputs=model_inputs, target=target_class)