import os
import argparse
from lib.utils.misc import dict_to_markdown


# core config
parser = argparse.ArgumentParser(description='Missing Modality in Multimodal Action Recognition')
parser.add_argument('--dataset', type=str, default='ntu_rgbd60',
                    choices=['ntu_rgbd60', 'ntu_rgbd120', 'nw_ucla', 'uwa3d'])
parser.add_argument('--num_frames', default=16, type=int,
                    help='number of input frames (i.e., sequence length) to transformer encoder.')
parser.add_argument('--img_size', default=224, type=int,
                    help='input image size.')
parser.add_argument('--modality', type=str, default='rgb_depth',
                    help='modality to use in training phase.')
parser.add_argument('--evaluation', type=str, default='cross_subject',
                    choices=['cross_subject', 'cross_setup', 'cross_view'],
                    help="NTU_RGB+D 60: cross_subject, cross_view; "
                         "NTU_RGB+D 120: cross_subject, cross_setup")
parser.add_argument('--temporal_augmentation', action='store_true',
                    help='whether to use temporal augmentation (time-ordered random sampling).')
parser.add_argument('--imagenet_pretrained', action='store_true',
                    help='whether to use imagenet initialization.')
parser.add_argument('--model', default='actionmae', type=str,
                    choices=['baseline', 'actionmae'],
                    help="Model type")
parser.add_argument('--model_size', default='small', type=str,
                    choices=['tiny', 'small', 'base', 'large', 'huge'],
                    help="Model size of ActionMAE")
parser.add_argument('--fusion', default='sum', type=str,
                    choices=['sum', 'concat', 'transformer'],
                    help="Fusion method")
parser.add_argument('--num_mem_token', default=1, type=int,
                    help="Number of memory tokens")
parser.add_argument('--set_loss_mask', default=1, type=int,
                    help="Mask coefficient in the total loss")
parser.add_argument('--set_loss_label', default=1, type=int,
                    help="Label coefficient in the total loss")
parser.add_argument('--use_gradient_modulation', action='store_true',
                    help='whether to use gradient modulation.')
parser.add_argument('--modulation_ratio', default=1.0, type=float,
                    help="modulation ratio")


# for NTU RGB+D dataset
parser.add_argument('--rgb_dir', type=str, default='/dataset/rgb/',
                    help='directory of rgb dataset')
parser.add_argument('--depth_dir', type=str, default='/dataset/depth_color/',
                    help='directory of depth dataset')
parser.add_argument('--ir_dir', type=str, default='/dataset/ir/',
                    help='directory of ir dataset')
parser.add_argument('--skeleton_dir', type=str, default='/dataset/skeleton/',
                    help='directory of skeleton dataset')


# for NW-UCLA dataset
parser.add_argument('--nwucla_dir', type=str, default='/dataset/multiview_action/',
                    help='dataset directory')


# for UWA3D dataset
parser.add_argument('--uwa3d_dir', type=str, default='/dataset/uwa3d/',
                    help='directory of rgb dataset')


# training config
parser.add_argument('--start_epoch', type=int, default=None,
                    help='if None, will be set automatically when using --resume_all')
parser.add_argument('--end_epoch', type=int, default=200,
                    help='number of epochs to run')
parser.add_argument('--early_stop_patience', type=int, default=-1,
                    help='number of epochs to early stop, use -1 to disable early stop')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay (default=0.0001)')


# loader config
parser.add_argument('--bs', type=int, default=16, # FIXME
                    help='batch size')
parser.add_argument('--eval_bs', type=int, default=16, # FIXME
                    help='batch size at inference, for query')
parser.add_argument('--num_workers', type=int, default=4,
                    help='num subprocesses used to load the data, 0: use main process')
parser.add_argument('--no_pin_memory', dest='pin_memory', action='store_false',
                    help='No use of pin_memory for data loading.'
                         'If pin_memory=True, the data loader will copy Tensors into CUDA pinned memory before returning them.')


# meta config
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1). if seed=0, seed is not fixed.')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='how many iters to wait before logging training status')
parser.add_argument('--val_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before validation')
parser.add_argument('--save_interval', type=int, default=200, metavar='N',
                    help='how many epochs to wait before saving a model')
parser.add_argument('--gpu_devices', type=str, default='0',
                    help='GPU ID')
parser.add_argument('--debug', action='store_true',
                    help='debug (fast) mode, break all loops, do not load all data into memory.')
parser.add_argument("--eval_untrained", action="store_true",
                    help="Evaluate on untrained model")
parser.add_argument('--log_dir', type=str, default='logs',
                    help='directory for saving logs')
parser.add_argument('--save', type=str, default='./save',
                    help='dir to save model')
parser.add_argument('--resume', action='store_true',
                    help='if --resume, only load model weights')
parser.add_argument('--resume_all', action='store_true',
                    help='if --resume_all, load optimizer/scheduler/epoch as well')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='checkpoint path to resume or evaluate')
parser.add_argument('--checkpoint_r', type=str, default=None,
                    help='rgb checkpoint')
parser.add_argument('--checkpoint_d', type=str, default=None,
                    help='depth checkpoint')
parser.add_argument('--checkpoint_i', type=str, default=None,
                    help='ir checkpoint')
parser.add_argument('--checkpoint_s', type=str, default=None,
                    help='skeleton checkpoint')
parser.add_argument('--use_neptune', action='store_true',
                    help='enable use of neptune for logging purpose')

# dist config
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
parser.add_argument('--sync_bn', action='store_true',
                    help='enabling apex sync BN.')
parser.add_argument('--channels-last', type=bool, default=False)
parser.add_argument('--opt-level', type=str, default='O1',
                    help='O0: "Pure FP32", '
                         'O1: "Official mixed precision recipe (recommended)", '
                         'O2: "Almost FP16", '
                         'O3: "Pure FP16"')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None,
                    help='only applicable for O2 and O3.')
parser.add_argument('--loss-scale', type=str, default=None,
                    help='if opt-level == O0 or O3: loss-scale=1.0; '
                         'if opt-level == O1 or O2: loss-scale="dynamic".')

args = parser.parse_args()

# Display settings
if args.local_rank == 0:
     print(dict_to_markdown(vars(args), max_str_len=120))