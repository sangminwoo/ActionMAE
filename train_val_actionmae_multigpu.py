import os
import time
import random
import pprint
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from lib.modeling.model_builder import build_model
from lib.dataset.dataset_builder import build_dataset, collate_fn, prepare_batch_inputs
from lib.dataset.action_classes import NTURGBD60_CLASSES, NTURGBD120_CLASSES, NWUCLA_CLASSES
from lib.utils.misc import cur_time, AverageMeter
from lib.utils.model_utils import count_parameters, accuracy, per_class_accuracy
from lib.utils.train_utils import to_python_float, reduce_tensor
from lib.utils.logger import setup_logger
from lib.configs import args

import warnings
warnings.filterwarnings('ignore')


def set_seed(seed, use_cuda=True):
    # fix seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_setup(logger):
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)

        # initialize the process group
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.seed:
        set_seed(args.seed)

    if args.debug: # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    model = build_model(args)

    if args.sync_bn:
        import apex
        logger.info("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model.to(device=device, memory_format=memory_format)

    ###########################################################################################
    # Disable learning rate decay for pretrained weights
    param_dicts = [{'params': [param for name, param in model.named_parameters() if param.requires_grad]}]
    
    # Enable learning rate decay of pretrained weights
    # pretrained_params = []
    # non_pretrained_params = []
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         if any(n in name for n in ['s_', 't_', 'predictor']):
    #             # param.requires_grad = False
    #             pretrained_params.append(param)
    #         else:
    #             non_pretrained_params.append(param)
    # param_dicts = [{'params':pretrained_params}, {'params':non_pretrained_params}]
    ###########################################################################################

    # Scale learning rate based on global batch size
    args.lr = args.lr*float(args.bs)/32
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.wd)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale)

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.checkpoint):
                if args.local_rank == 0:
                    logger.info(f"Loading checkpoint '{args.checkpoint}'")
                    
                checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda(args.gpu))
                model.load_state_dict(checkpoint['model'], strict=False)
                if args.resume_all:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    amp.load_state_dict(checkpoint['amp'])
                    args.start_epoch = checkpoint['epoch'] + 1
                if args.local_rank == 0:
                    logger.info(f'Loaded model saved at epoch {checkpoint["epoch"]} from checkpoint: {args.checkpoint}')
            else:
                if args.local_rank == 0:
                    logger.info(f"No checkpoint found at '{args.checkpoint}'")

        def resume_from_respective_modality():
            if os.path.isfile(args.checkpoint_r):
                if args.local_rank == 0:
                    logger.info(f"Loading checkpoint '{args.checkpoint_r}'")
                checkpoint_r = torch.load(args.checkpoint_r, map_location=lambda storage, loc: storage.cuda(args.gpu))
                rgb_ckpt = {k: v for k, v in checkpoint_r['model'].items() if 'module.t' not in k}
                model.load_state_dict(rgb_ckpt, strict=False)
                if args.local_rank == 0:
                    logger.info(f'Loaded model saved at epoch {checkpoint_r["epoch"]} from checkpoint: {args.checkpoint_r}')
            else:
                if args.local_rank == 0:
                    logger.info(f"No checkpoint found at '{args.checkpoint_r}'")

            if os.path.isfile(args.checkpoint_d):
                if args.local_rank == 0:
                    logger.info(f"Loading checkpoint '{args.checkpoint_d}'")
                checkpoint_d = torch.load(args.checkpoint_d, map_location=lambda storage, loc: storage.cuda(args.gpu))
                depth_ckpt = {k: v for k, v in checkpoint_d['model'].items() if 'module.t' not in k}
                model.load_state_dict(depth_ckpt, strict=False)
                if args.local_rank == 0:
                    logger.info(f'Loaded model saved at epoch {checkpoint_d["epoch"]} from checkpoint: {args.checkpoint_d}')
            else:
                if args.local_rank == 0:
                    logger.info(f"No checkpoint found at '{args.checkpoint_d}'")

            if os.path.isfile(args.checkpoint_i):
                if args.local_rank == 0:
                    logger.info(f"Loading checkpoint '{args.checkpoint_i}'")
                checkpoint_i = torch.load(args.checkpoint_i, map_location=lambda storage, loc: storage.cuda(args.gpu))
                ir_ckpt = {k: v for k, v in checkpoint_i['model'].items() if 'module.t' not in k}
                model.load_state_dict(ir_ckpt, strict=False)
                if args.local_rank == 0:
                    logger.info(f'Loaded model saved at epoch {checkpoint_i["epoch"]} from checkpoint: {args.checkpoint_i}')
            else:
                if args.local_rank == 0:
                    logger.info(f"No checkpoint found at '{args.checkpoint_i}'")

            if os.path.isfile(args.checkpoint_s):
                if args.local_rank == 0:
                    logger.info(f"Loading checkpoint '{args.checkpoint_s}'")
                checkpoint_s = torch.load(args.checkpoint_s, map_location=lambda storage, loc: storage.cuda(args.gpu))
                skeleton_ckpt = {k: v for k, v in checkpoint_s['model'].items() if 'module.t' not in k}
                model.load_state_dict(skeleton_ckpt, strict=False)
                if args.local_rank == 0:
                    logger.info(f'Loaded model saved at epoch {checkpoint_s["epoch"]} from checkpoint: {args.checkpoint_s}')
            else:
                if args.local_rank == 0:
                    logger.info(f"No checkpoint found at '{args.checkpoint_s}'")

        # Uncomment below to load pre-trained weights
        ###########################################################################################
        resume()
        ###########################################################################################
        
        # Uncomment below to load modality-specific pre-trained weights
        ###########################################################################################
        # resume_from_respective_modality()
        ###########################################################################################

    return model, optimizer, device


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    ###########################################################################################
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # optimizer.param_groups[0]['lr'] = lr*0.01  # decay learning rate of pretrained parameters 
    # optimizer.param_groups[1]['lr'] = lr 
    ###########################################################################################


def train_epoch(model, dataloader, optimizer, device, epoch_i):
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)
    acc_meters = defaultdict(AverageMeter)
    
    # switch to train mode
    model.train()

    tictoc = time.time()
    for idx, batch in enumerate(dataloader):
        
        adjust_learning_rate(optimizer, epoch_i, idx, len(dataloader))

        time_meters['dataloading_time'].update(time.time() - tictoc)
        tictoc = time.time()

        inputs, targets = batch
        model_inputs = prepare_batch_inputs(inputs, device, non_blocking=args.pin_memory)
        time_meters['prepare_inputs_time'].update(time.time() - tictoc)
        tictoc = time.time()

        # compute output
        targets = torch.tensor(targets).to(device)
        loss_mask_dict, loss_label_dict, preds_dict = model(targets, **model_inputs)
        time_meters['model_forward_time'].update(time.time() - tictoc)
        tictoc = time.time()

        # compute gradient and do SGD step
        optimizer.zero_grad()

        losses = sum(args.set_loss_mask * loss_mask for loss_mask in loss_mask_dict.values()) + \
                 sum(args.set_loss_label * loss_label for loss_label in loss_label_dict.values())

        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()

        optimizer.step()
        time_meters['model_backward_time'].update(time.time() - tictoc)
        tictoc = time.time()

        # Measure accuracy
        acc_dict = {
            k : accuracy(preds, targets, topk=(1, 5))
            for k, preds in preds_dict.items()
        }

        # Average loss and accuracy across processes for logging
        if args.distributed:
            loss_mask_dict = {
                f'loss_mask_{k}': reduce_tensor(loss_mask, args.world_size)
                for k, loss_mask in loss_mask_dict.items()
            }
            loss_label_dict = {
                f'loss_label_{k}': reduce_tensor(loss_label, args.world_size)
                for k, loss_label in loss_label_dict.items()
            }
            losses = reduce_tensor(losses.data, args.world_size)
            acc1_dict = {
                f'acc@1_{k}': reduce_tensor(acc_at_1, args.world_size)
                for k, (acc_at_1, _) in acc_dict.items()
            }
            acc5_dict = {
                f'acc@5_{k}': reduce_tensor(acc_at_5, args.world_size)
                for k, (_, acc_at_5) in acc_dict.items()
            }
        
        loss_dict = {}
        loss_dict.update(loss_mask_dict)
        loss_dict.update(loss_label_dict)
        loss_dict['loss_overall'] = float(losses)

        acc_dict = {}
        acc_dict.update(acc1_dict)
        acc_dict.update(acc5_dict)

        for k, v in loss_dict.items():
            if 'loss_label' in k:
                loss_meters[k].update(
                    val=float(v) * args.set_loss_label,
                    n=list(model_inputs.values())[0].size(0)
                )
            if 'loss_mask' in k:
                loss_meters[k].update(
                    val=float(v) * args.set_loss_mask,
                    n=list(model_inputs.values())[0].size(0)
                )
            if 'loss_overall' in k:
                loss_meters[k].update(
                    val=float(v),
                    n=list(model_inputs.values())[0].size(0)
                )

        for k, v in acc_dict.items():
            acc_meters[k].update(
                val=float(v),
                n=list(model_inputs.values())[0].size(0)
            )

        torch.cuda.synchronize()
        time_meters['logging_time'].update(time.time() - tictoc)
        tictoc = time.time()

        if args.local_rank == 0:
            if idx % args.log_interval == 0 or (idx+1) == len(dataloader):
                logger.info(
                    "Training Logs\n"
                    "[Epoch] [{epoch:03d}][{iter:03d}/{total:03d}]\n"
                    "[Time]\n{time_stats}\n"
                    "[Loss]\n{loss_str}\n"
                    "[Acc]\n{acc_str}\n".format(
                        time_str=time.strftime("%Y-%m-%d %H:%M:%S"),
                        epoch=epoch_i+1,
                        iter=idx,
                        total=len(dataloader),
                        time_stats="\n".join("\t> {} {:.4f} ({:.4f})".format(k, v.val, v.avg) for k, v in time_meters.items()),
                        loss_str="\n".join(["\t> {} {:.4f} ({:.4f})".format(k, v.val, v.avg) for k, v in loss_meters.items()]),
                        acc_str="\n".join(["\t> {} {:.4f} ({:.4f})".format(k, v.val, v.avg) for k, v in acc_meters.items()])
                    )
                )

        if args.debug:
            break

    return time_meters, loss_meters, acc_meters


@torch.no_grad()
def val_epoch(model, dataloader, optimizer, device, epoch_i):
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)
    acc_meters = defaultdict(AverageMeter)
    # per_cls_acc_meters = defaultdict(AverageMeter)

    # if args.dataset == 'ntu_rgbd60': num_classes = 60; ACTION_CLASSES=NTURGBD60_CLASSES
    # elif args.dataset == 'ntu_rgbd120': num_classes = 120; ACTION_CLASSES=NTURGBD120_CLASSES
    # elif args.dataset == 'nw_ucla': num_classes = 10; ACTION_CLASSES=NWUCLA_CLASSES
    # else: raise NotImplementedError

    # switch to evaluate mode
    model.eval()
    # criterion.eval()

    tictoc = time.time()
    for idx, batch in enumerate(dataloader):

        time_meters['dataloading_time'].update(time.time() - tictoc)
        tictoc = time.time()
        
        inputs, targets = batch
        model_inputs = prepare_batch_inputs(inputs, device, non_blocking=args.pin_memory)
        time_meters['prepare_inputs_time'].update(time.time() - tictoc)
        tictoc = time.time()

        # compute output
        targets = torch.tensor(targets).to(device)
        loss_mask_dict, loss_label_dict, preds_dict = model.forward(targets, eval_mode=True, **model_inputs)
        time_meters['model_forward_time'].update(time.time() - tictoc)
        tictoc = time.time()

        losses = sum(args.set_loss_mask * loss_mask for loss_mask in loss_mask_dict.values()) + \
                 sum(args.set_loss_label * loss_label for loss_label in loss_label_dict.values())

        # measure accuracy and record loss
        acc_dict = {
            k : accuracy(preds, targets, topk=(1, 5))
            for k, preds in preds_dict.items()
        }
        # per_cls_acc = per_class_accuracy(preds, targets, topk=(1, 5), num_classes=num_classes)
        # per_cls_acc = torch.tensor(per_cls_acc).to(preds.device)

        # Average loss and accuracy across processes for logging
        if args.distributed:
            loss_mask_dict = {
                f'loss_mask_{k}': reduce_tensor(loss_mask, args.world_size)
                for k, loss_mask in loss_mask_dict.items()
            }
            loss_label_dict = {
                f'loss_label_{k}': reduce_tensor(loss_label, args.world_size)
                for k, loss_label in loss_label_dict.items()
            }
            losses = reduce_tensor(losses.data, args.world_size)
            acc1_dict = {
                f'acc@1_{k}': reduce_tensor(acc_at_1, args.world_size)
                for k, (acc_at_1, _) in acc_dict.items()
            }
            acc5_dict = {
                f'acc@5_{k}': reduce_tensor(acc_at_5, args.world_size)
                for k, (_, acc_at_5) in acc_dict.items()
            }
            # per_cls_acc = reduce_tensor(per_cls_acc, args.world_size)

        loss_dict = {}
        loss_dict.update(loss_mask_dict)
        loss_dict.update(loss_label_dict)
        loss_dict['loss_overall'] = float(losses)

        acc_dict = {}
        acc_dict.update(acc1_dict)
        acc_dict.update(acc5_dict)

        for k, v in loss_dict.items():
            if 'loss_label' in k:
                loss_meters[k].update(
                    val=float(v) * args.set_loss_label,
                    n=list(model_inputs.values())[0].size(0)
                )
            if 'loss_mask' in k:
                loss_meters[k].update(
                    val=float(v) * args.set_loss_mask,
                    n=list(model_inputs.values())[0].size(0)
                )
            if 'loss_overall' in k:
                loss_meters[k].update(
                    val=float(v),
                    n=list(model_inputs.values())[0].size(0)
                )

        for k, v in acc_dict.items():
            acc_meters[k].update(
                val=float(v),
                n=list(model_inputs.values())[0].size(0)
            )

        torch.cuda.synchronize()
        time_meters['logging_time'].update(time.time() - tictoc)
        tictoc = time.time()
        
        if args.debug:
            break

    if args.local_rank == 0:
        logger.info(
            "Validation Logs\n"
            "[Epoch] [{epoch:03d}]\n"
            "[Time]\n{time_stats}\n"
            "[Loss]\n{loss_str}\n"
            "[Acc]\n{acc_str}\n".format(
            # "[Per Class Acc]\n{per_cls_acc}\n".format(
                time_str=time.strftime("%Y-%m-%d %H:%M:%S"),
                epoch=epoch_i+1,
                time_stats="\n".join("\t> {} {:.4f}".format(k, v.avg) for k, v in time_meters.items()),
                loss_str="\n".join(["\t> {} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]),
                acc_str="\n".join(["\t> {} {:.4f}".format(k, v.avg) for k, v in acc_meters.items()]),
                # per_cls_acc=" | ".join(["{} {:.4f}".format(k, v.avg) for k, v in per_cls_acc_meters.items()])
            )
        )

    return time_meters, loss_meters, acc_meters


def train_val(logger, run=None):
    model, optimizer, device = train_setup(logger)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)    
    # model = DDP(model, delay_allreduce=True)

    if args.local_rank == 0:
        logger.info(f'Model {model}')
        n_all, n_trainable, mem, mem_params, mem_bufs = count_parameters(model)
        if run:
            run[f"num_params"].log(n_all)
            run[f"num_trainable_params"].log(n_trainable) 
            run[f"mem"].log(n_all)
            run[f"mem_params"].log(n_trainable) 
            run[f"mem_bufs"].log(n_all)

    train_dataset = build_dataset(args, phase='train')
    val_dataset = build_dataset(args, phase='val')

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.bs,
        shuffle=(train_sampler is None), 
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        sampler=train_sampler,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        batch_size=args.eval_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        sampler=val_sampler,
        persistent_workers=True
    )

    # for early stop purpose
    best_acc = 0 # np.inf
    early_stop_count = 0
    
    # create checkpoint
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.start_epoch is None:
        start_epoch = -1 if args.eval_untrained else 0
    else:
        start_epoch = args.start_epoch

    for epoch_i in range(start_epoch, args.end_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch_i)

        args.phase = 'train'
        if start_epoch > -1:
            time_meters, loss_meters, acc_meters = train_epoch(model, train_loader, optimizer, device, epoch_i)
            
            # train log
            if args.local_rank == 0:
                if run:
                    run[f"Train/epoch"].log(epoch_i+1)

                    for k, v in loss_meters.items():
                        run[f"Train/{k}"].log(v.avg)

                    for k, v in acc_meters.items():
                        run[f"Train/{k}"].log(v.avg) 

        if (epoch_i + 1) % args.val_interval == 0:
            args.phase = 'val'
            with torch.no_grad():
                eval_time_meters, eval_loss_meters, eval_acc_meters = val_epoch(model, val_loader, optimizer, device, epoch_i)
                acc = sum(eval_acc_meters[k].avg for k in eval_acc_meters if 'acc@1' in k)

                # val log
                if args.local_rank == 0:
                    if run:
                        run[f"Val/epoch"].log(epoch_i+1)

                        for k, v in eval_loss_meters.items():
                            run[f"Val/{k}"].log(v.avg)

                        for k, v in eval_acc_meters.items():
                            run[f"Val/{k}"].log(v.avg)  

            # early stop
            if acc > best_acc:
                early_stop_count = 0
                best_acc = acc
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict(),
                    'epoch': epoch_i,
                    'args': args
                }
                torch.save(
                    checkpoint,
                    os.path.join(
                        args.save,
                        f'best_model_{args.dataset}_{args.num_frames}_{args.img_size}_' \
                        f'{args.modality}_{args.evaluation}_{args.model}_{args.model_size}_' \
                        f'{args.fusion}_{args.num_mem_token}_{args.set_loss_mask}_{args.set_loss_label}.ckpt'
                    )
                )
            else:
                early_stop_count += 1
                if args.local_rank == 0:
                    if args.early_stop_patience > 0 and early_stop_count > args.early_stop_patience:
                        logger.info(f'\n>>>>> Early Stop at Epoch {epoch_i+1} (best acc: {best_acc})\n')
                        break

        if (epoch_i + 1) % args.save_interval == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'epoch': epoch_i,
                'args': args
            }
            torch.save(
                checkpoint,
                os.path.join(
                    args.save,
                    f'{epoch_i:04d}_model_{args.dataset}_{args.num_frames}_{args.img_size}_' \
                    f'{args.modality}_{args.evaluation}_{args.model}_{args.model_size}_' \
                    f'{args.fusion}_{args.num_mem_token}_{args.set_loss_mask}_{args.set_loss_label}.ckpt'
                )
            )
        if args.debug:
            break


if __name__ == '__main__':
    run = None
    if args.use_neptune:
        import neptune.new as neptune

        # Neptune init
        if args.local_rank == 0:
            run = neptune.init(
                project='ANONYMOUS',
                api_token='ANONYMOUS',
            )

            # Neptune save args
            params = vars(args)
            run['parameters'] = params

    logger = None
    if args.local_rank == 0:
        logger = setup_logger('ActionMAE', args.log_dir, distributed_rank=0, filename=cur_time()+"_train.txt")
    
    train_val(logger, run=run)
    
    if args.local_rank == 0:
        if args.use_neptune:
            run.stop()