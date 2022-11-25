import torch
from .ntu60_dataset import NTU60Dataset
from .ntu120_dataset import NTU120Dataset
from .nwucla_dataset import NWUCLADataset
from .uwa3d_dataset import UWA3DDataset
from lib.utils.tensor_utils import pad_sequences_1d


def build_dataset(args, phase):
    if args.dataset == 'ntu_rgbd60':
        Dataset = NTU60Dataset
    elif args.dataset == 'ntu_rgbd120':
        Dataset = NTU120Dataset
    elif args.dataset == 'nw_ucla':
        Dataset = NWUCLADataset
    elif args.dataset == 'uwa3d':
        Dataset = UWA3DDataset
    else:
        raise NotImplementedError
        
    modality = args.modality.split('_')  # rgb, depth, skeleton

    return Dataset(
        phase=phase,
        img_size=args.img_size,
        num_frames=args.num_frames,
        modality=modality,
        evaluation=args.evaluation,
        temporal_augmentation=args.temporal_augmentation,
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        skeleton_dir=args.skeleton_dir,
        ir_dir=args.ir_dir,
        nwucla_dir=args.nwucla_dir,
        uwa3d_dir=args.uwa3d_dir
    )


def collate_fn(batch):
    batched_targets = [b['target'] for b in batch]
    input_keys = batch[0]['model_inputs'].keys()
    batched_inputs = dict()
    for k in input_keys:
        batched_inputs[k] = pad_sequences_1d(
            [b['model_inputs'][k] for b in batch],  # list[N * (TxCxHxW)]
            dtype=torch.float32,
        )  # NxTxCxHxW, NxT
    return batched_inputs, batched_targets


def prepare_batch_inputs(batched_inputs, device, non_blocking=False):
    model_inputs = {}

    if 'rgb' in batched_inputs:
        model_inputs['src_rgb'] = batched_inputs['rgb'][0].to(device, non_blocking=non_blocking)
        model_inputs['src_rgb_mask'] = batched_inputs['rgb'][1].to(device, non_blocking=non_blocking)
        
    if 'depth' in batched_inputs:
        model_inputs['src_depth'] = batched_inputs['depth'][0].to(device, non_blocking=non_blocking)
        model_inputs['src_depth_mask'] = batched_inputs['depth'][1].to(device, non_blocking=non_blocking)
    
    if 'ir' in batched_inputs:
        model_inputs['src_ir'] = batched_inputs['ir'][0].to(device, non_blocking=non_blocking)
        model_inputs['src_ir_mask'] = batched_inputs['ir'][1].to(device, non_blocking=non_blocking)

    if 'skeleton' in batched_inputs:
        model_inputs['src_skeleton'] = batched_inputs['skeleton'][0].to(device, non_blocking=non_blocking)
        model_inputs['src_skeleton_mask'] = batched_inputs['skeleton'][1].to(device, non_blocking=non_blocking)

    return model_inputs