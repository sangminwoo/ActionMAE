import os
import random
import torch
import numpy as np
from PIL import Image
from glob import glob
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms as T
from matplotlib import cm


class NTU60Dataset(Dataset):
    '''Missing Modality in Multi-modal Action Recognition Dataset'''

    def __init__(self, phase='train', img_size=224, num_frames=32,
                 modality=['rgb', 'depth'], evaluation='cross_subject',
                 temporal_augmentation=True,
                 rgb_dir='/dataset/rgb/',
                 depth_dir='/dataset/depth_color/',
                 ir_dir='/dataset/ir/',
                 skeleton_dir='/dataset/skeleton/',
                 **kwargs):
        '''
        phase: phase in ['train', 'val', 'test']
        img_size: input image size
        num_frames: number of input frames (i.e., sequence length) to transformer encoder
        modality: modalities to use (e.g., rgb, depth, ir, skeleton)
        evaluation: evaluation setting (e.g., cross-view, cross-subject)
        temporal_augmentation: whether to use temporal augmentation (time-ordered random sampling),
        '''
        super(NTU60Dataset).__init__()
        assert phase in ['train', 'val', 'test'], f'{phase} is not available. should be one of train/val.'
        self.phase = phase
        self.img_size = img_size
        self.num_frames = num_frames
        self.root = {
            'rgb': rgb_dir,
            'depth': depth_dir,
            'ir': ir_dir,
            'skeleton': skeleton_dir
        }

        self.eval = evaluation
        if self.eval == 'cross_view':
            TYPE = 1  # "C001" in S001C001P001R001A001_rgb
            if self.phase == 'train':  # cam2 & cam3
                IDS = [2, 3]
            else:  # test: cam1
                IDS = [1]
        elif self.eval == 'cross_subject':
            TYPE = 2  # "P001" in S001C001P001R001A001_rgb
            if self.phase == 'train':  # 20 subjects
                IDS = [1, 2, 4,  5,  8,  9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
            else:  # test: 20 subjects
                IDS = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

        self.modality = modality
        self.temporal_augmentation = temporal_augmentation and self.phase=='train'
        self.data_dir = defaultdict(list)
        
        # some of the captured samples in the "NTU RGB+D" dataset have missing or incomplete skeleton data.
        missing = open('missing_skeletons.txt').read().splitlines()

        if 'rgb' in self.modality:
            for dir1 in sorted(glob(os.path.join(self.root['rgb'], 'nturgb+d*'))):  # nturgb+d_rgb_s001
                if int(dir1.split('/')[-1][-3:]) > 17:  # int(001) > 17
                    continue
                for dir2 in sorted(glob(os.path.join(dir1, 'frames', '*'))):  # frames/S001C001P001R001A001_rgb
                    ''' S: setup / C: camera / P: performer / R: replication / A: action class '''
                    if int(dir2.split('/')[-1][4*TYPE:4*TYPE+4][1:]) not in IDS:
                        continue
                    if 'skeleton' in self.modality:
                        # some skeleton files are missing.
                        # thus, we skip missing files for pairing purpose.
                        if dir2.split('/')[-1][:20] in missing:  
                            continue
                    self.data_dir['rgb'].append(dir2)

        if 'depth' in self.modality:
            for dir1 in sorted(glob(os.path.join(self.root['depth'], 'nturgb+d*'))):  # nturgb+d_depth_masked_s001
                if int(dir1.split('/')[-1][-3:]) > 17:  # int(001) > 17
                    continue
                for dir2 in sorted(glob(os.path.join(dir1, 'nturgb+d_depth_masked', '*'))):  # nturgb+d_depth_masked/S001C001P001R001A001
                    ''' S: setup / C: camera / P: performer / R: replication / A: action class '''
                    if int(dir2.split('/')[-1][4*TYPE:4*TYPE+4][1:]) not in IDS:
                        continue
                    if 'skeleton' in self.modality:
                        # some skeleton files are missing.
                        # thus, we skip missing files for pairing purpose.
                        if dir2.split('/')[-1][:20] in missing:  
                            continue
                    self.data_dir['depth'].append(dir2)

        if 'ir' in self.modality:
            for dir1 in sorted(glob(os.path.join(self.root['ir'], 'nturgb+d*'))):  # nturgb+d_ir_s001
                if int(dir1.split('/')[-1][-3:]) > 17:  # int(001) > 17
                    continue
                for dir2 in sorted(glob(os.path.join(dir1, 'frames', '*'))):  # frames/S001C001P001R001A001_ir
                    ''' S: setup / C: camera / P: performer / R: replication / A: action class '''
                    if int(dir2.split('/')[-1][4*TYPE:4*TYPE+4][1:]) not in IDS:
                        continue
                    if 'skeleton' in self.modality:
                        # some skeleton files are missing.
                        # thus, we skip missing files for pairing purpose.
                        if dir2.split('/')[-1][:20] in missing:  
                            continue
                    self.data_dir['ir'].append(dir2)

        if 'skeleton' in self.modality:
            for dir1 in sorted(glob(os.path.join(self.root['skeleton'], 'raw_txt120', '*'))):  # S001C001P001R001A001.skeleton.npy
                if int(dir1.split('/')[-1][1:4]) > 17:  # int(001) > 17
                    continue
                ''' S: setup / C: camera / P: performer / R: replication / A: action class '''
                if int(dir1.split('/')[-1][4*TYPE:4*TYPE+4][1:]) not in IDS:
                    continue
                self.data_dir['skeleton'].append(dir1)

        # sanity check
        # print(f'{phase} dataset configurations')
        # for mode in modality:
        #     print(f'{mode}: {len(self.data_dir[mode])}')
        # rgb_dir = [d.split('/')[-1][:20] for d in self.data_dir['rgb']]
        # depth_dir = [d.split('/')[-1][:20] for d in self.data_dir['depth']]
        # skeleton_dir = [d.split('/')[-1][:20] for d in self.data_dir['skeleton']]
        # ir_dir = [d.split('/')[-1][:20] for d in self.data_dir['ir']]
        # assert rgb_dir == depth_dir == skeleton_dir == ir_dir

        if phase == 'train':
            self.augmenters = {
                'rgb': T.Compose([
                            T.CenterCrop(size=self.img_size),  # 480x270
                            T.RandomCrop(size=0.8*self.img_size),
                            T.Resize(size=self.img_size),
                            T.ColorJitter(brightness=.5, hue=.3),
                            T.RandomHorizontalFlip(p=0.5),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]),
                'depth': T.Compose([
                            T.CenterCrop(size=self.img_size*1.2),  # 512x424
                            T.RandomCrop(size=self.img_size),
                            T.RandomHorizontalFlip(p=0.5),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]),
                'ir': T.Compose([
                            T.CenterCrop(size=self.img_size*1.2),  # 512x424
                            T.RandomCrop(size=self.img_size),
                            T.RandomHorizontalFlip(p=0.5),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
            }
        else:  # test
            self.augmenters = {
                'rgb': T.Compose([
                            # T.Resize((self.img_size, self.img_size)),  # 480x270
                            T.CenterCrop(size=0.8*self.img_size),
                            T.Resize(size=self.img_size),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]),
                'depth': T.Compose([
                            # T.Resize((self.img_size, self.img_size)),  # 512x424
                            T.CenterCrop(size=self.img_size),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]),
                'ir': T.Compose([
                            T.CenterCrop(size=self.img_size),  # 512x424
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
            }

    def __len__(self):
        ''' number of total data '''
        return len(list(self.data_dir.values())[0])

    def __getitem__(self, idx):
        model_inputs = {}
        modality = list(self.data_dir.keys())  # rgb, depth, skeleton

        # sanity check
        # assert len(os.listdir(self.data_dir['rgb'][idx])) \
        #     == len(os.listdir(self.data_dir['depth'][idx])) \
        #     == len(os.listdir(self.data_dir['ir'][idx])) \
        #     == len(np.load(self.data_dir['skeleton'][idx], allow_pickle=True).item()['skel_body0'])

        if modality[0] == 'skeleton':
            video_len = np.load(self.data_dir[modality[0]][idx], allow_pickle=True).item()['skel_body0'].shape[0]
        else:
            video_len = len(os.listdir(self.data_dir[modality[0]][idx]))
        idxs = list(range(video_len))
        stepsize = video_len / self.num_frames

        if self.temporal_augmentation:
            sampled_idxs = idxs if video_len < self.num_frames else sorted(random.sample(idxs, self.num_frames))
        else:  # uniform sample
            sampled_idxs = idxs if video_len < self.num_frames else [idxs[round(stepsize*i)] for i in range(self.num_frames)]

        if 'rgb' in modality:
            video_rgb = []
            for i, path in enumerate(sorted(glob(os.path.join(self.data_dir['rgb'][idx], '*')))):  # output0001.png
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
            for i, path in enumerate(sorted(glob(os.path.join(self.data_dir['depth'][idx], '*')))):  # MDepth-00000001.png
                if i not in sampled_idxs:
                    continue
                try:
                    depth = Image.open(path)#.convert('RGB')

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

        if 'ir' in modality:
            video_ir = []
            for i, path in enumerate(sorted(glob(os.path.join(self.data_dir['ir'][idx], '*')))):  # output0001.png
                if i not in sampled_idxs:
                    continue
                try:
                    ir = Image.open(path)
                except:
                    print(f'unable to open IR from {path}.')
                    continue
                else:
                    ir = self.augmenters['ir'](ir)  # CxHxW
                    video_ir.append(ir)
            video_ir = torch.stack(video_ir)  # TxCxHxW
            model_inputs['ir'] = video_ir

        if 'skeleton' in modality:
            path = self.data_dir['skeleton'][idx]  # S001C001P004R002A001.skeleton.npy
            try:
                video_skel = np.load(path, allow_pickle=True).item()['skel_body0']
            except:
                print(f'unable to open Skeleton from {path}.')
                # TODO 
            else:
                video_skel = video_skel[sampled_idxs]
                video_skel = torch.from_numpy(video_skel)  # Tx25x3
                # video_skel = video_skel.transpose(1, 2)  # Tx3x25
                # x =  video_skel[:, :, :1] # Tx3x1
                # video_skel = video_skel.repeat(1, 1, 224*224//25)  # Tx3x50175
                # video_skel = torch.cat((video_skel, x), dim=2)  # Tx3x50176
                # video_skel = video_skel.reshape(-1, 3, 224, 224)  # Tx3x224x224
            model_inputs['skeleton'] = video_skel

        # S001C001P001R001A001_rgb
        target_class = int(self.data_dir[modality[0]][idx].split('/')[-1][17:20]) - 1

        return dict(model_inputs=model_inputs, target=target_class)