import os
import random
import torch
import numpy as np
from PIL import Image
from glob import glob
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms as T


class NWUCLADataset(Dataset):
    '''Missing Modality in Multi-modal Action Recognition Dataset'''

    def __init__(self, phase='train', img_size=224, num_frames=32,
                 modality=['rgb', 'depth'], temporal_augmentation=True,
                 nwucla_dir='/dataset/multiview_action/',
                 **kwargs):
        '''
        phase: phase in ['train' , 'val']
        num_frames: number of input frames (i.e., sequence length) to transformer encoder
        modality: modalities to use (e.g., rgb, depth, skeleton, ir)
        nwucla_dir: root directory for dataset
        '''
        super(NWUCLADataset).__init__()
        assert phase in ['train', 'val', 'test'], f'{phase} is not available. should be one of train/val.'
        self.phase = phase
        self.img_size = img_size
        self.num_frames = num_frames
        self.root = nwucla_dir

        if self.phase == 'train':
            IDS = [1, 2]
        else:
            IDS = [3]

        self.modality = modality
        self.temporal_augmentation = temporal_augmentation and self.phase=='train'
        self.data_dir = []

        for ID in IDS:
            for dir1 in sorted(glob(os.path.join(self.root, f'view_{ID}', '*'))):  # a01_s01_e00
                self.data_dir.append(dir1)

        if phase == 'train':
            self.augmenters = {
                'rgb': T.Compose([
                            T.CenterCrop(size=self.img_size),  # 640x480
                            T.RandomCrop(size=0.8*self.img_size),
                            T.Resize(size=self.img_size),
                            T.ColorJitter(brightness=.5, hue=.3),
                            T.RandomHorizontalFlip(p=0.5),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]),
                'depth': T.Compose([
                            T.CenterCrop(size=self.img_size*1.2),  # 320x240
                            T.RandomCrop(size=self.img_size),
                            T.RandomHorizontalFlip(p=0.5),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]),
            }
        else:  # test
            self.augmenters = {
                'rgb': T.Compose([
                            # T.Resize((self.img_size, self.img_size)),
                            T.CenterCrop(size=0.8*self.img_size),
                            T.Resize(size=self.img_size),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]),
                'depth': T.Compose([
                            # T.Resize((self.img_size, self.img_size)),
                            T.CenterCrop(size=self.img_size),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]),
            }

        # sanity check
        # print(f'{phase} dataset configurations')
        # print(len(self.data_dir))

    def __len__(self):
        ''' number of total data '''
        return len(self.data_dir)

    def __getitem__(self, idx):
        model_inputs = {}
        modality = self.modality

        video_len = len(glob(os.path.join(self.data_dir[idx], '*rgb.jpg')))  # rgb
        # sanity check
        # depth_len = len(glob(os.path.join(self.data_dir[idx], '*depth.png')))  # depth
        # skel_len = len(glob(os.path.join(self.data_dir[idx], '*skeleton.txt')))  # skeleton
        # assert video_len == depth_len == skel_len, f'rgb:{video_len}, depth:{depth_len}, skel:{skel_len}'
        
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
            for i, path in enumerate(sorted(glob(os.path.join(self.data_dir[idx], '*_rgb.jpg')))):  # frame_167_tc_90467319_rgb.jpg
                if i not in sampled_idxs:
                    continue
                try:
                    rgb = Image.open(path).convert('RGB')
                except:
                    print(f'unable to open RGB from {path}.')
                    continue
                else:
                    rgb = self.augmenters['rgb'](rgb)  # CxHxW
                    video_rgb.append(rgb)
            video_rgb = torch.stack(video_rgb)  # TxCxHxW
            model_inputs['rgb'] = video_rgb

        if 'depth' in modality:
            video_depth = []
            # TODO: try '*_depth.png'
            for i, path in enumerate(sorted(glob(os.path.join(self.data_dir[idx], '*_depth_vis.jpg')))):  # frame_167_tc_90467351_depth_vis.jpg
                if i not in sampled_idxs:
                    continue
                try:
                    # depth = Image.open(path).convert('RGB')

                    # gray map to color map using cm.jet
                    # this diminishes the number of pixel values
                    depth = np.array(Image.open(path))
                    depth = cm.jet((depth - depth.min())/ (depth.max() - depth.min()))[:,:,:3]
                    depth = Image.fromarray((depth*255).astype(np.uint8))
                except:
                    print(f'unable to open Depth from {path}.')
                    continue
                else:
                    depth = self.augmenters['depth'](depth)  # CxHxW
                    video_depth.append(depth)
            video_depth = torch.stack(video_depth)  # TxCxHxW
            model_inputs['depth'] = video_depth

        if 'skeleton' in modality:
            path = self.data_dir['skeleton'][idx]  # S001C001P004R002A001.skeleton.npy
            for i, path in enumerate(sorted(glob(os.path.join(self.data_dir[idx], '*_skeleton.txt')))):  # frame_167_tc_90467351_skeletons.txt
                if i not in sampled_idxs:
                    continue
                try:
                    skel = np.loadtxt(path, delimiter=',', skiprows=1)  # use_cols=[0,1,2]  # 20x4
                except:
                    print(f'unable to open Skeleton from {path}.')
                    continue
                else:
                    skel = torch.from_numpy(skel)  # 20x4
                    video_skel.append(skel)
            video_skel = torch.stack(video_skel)  # Tx20x4
            model_inputs['skeleton'] = video_skel

        # a01_s01_e01
        action_class = int(self.data_dir[idx].split('/')[-1].split('_')[0][1:])  
        ACTION_TO_LABEL = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 8:6, 9:7, 11:8, 12:9}
        target_class = ACTION_TO_LABEL[action_class]

        return dict(model_inputs=model_inputs, target=target_class)