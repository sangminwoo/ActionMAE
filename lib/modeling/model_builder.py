from functools import partial

import torch.nn as nn

from lib.modeling.baseline_resnet_fusion import BaselineResNet
from lib.modeling.actionmae_v5 import ActionMAE

##################################
#        Baseline: ResNet        #
##################################
def baseline_tiny(**kwargs):  # params: 8,405,092
	model = BaselineResNet(
		s_encoder='resnet18',
		t_encoder_embed_dim=512, t_encoder_depth=2, t_encoder_num_heads=8,
		dim_feedforward=2048, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model


def baseline_small(**kwargs):  # params: 32,734,372
	model = BaselineResNet(
		s_encoder='resnet34',
		t_encoder_embed_dim=512, t_encoder_depth=2, t_encoder_num_heads=8,
		dim_feedforward=2048, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model


def baseline_base(**kwargs):  # params: 129,168,676
	model = BaselineResNet(
		s_encoder='resnet50',
		t_encoder_embed_dim=512, t_encoder_depth=2, t_encoder_num_heads=8,
		dim_feedforward=2048, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model


def baseline_large(**kwargs):  # params: 129,168,676
	model = BaselineResNet(
		s_encoder='resnet101',
		t_encoder_embed_dim=512, t_encoder_depth=2, t_encoder_num_heads=8,
		dim_feedforward=2048, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model


def baseline_huge(**kwargs):  # params: 129,168,676
	model = BaselineResNet(
		s_encoder='resnet152',
		t_encoder_embed_dim=512, t_encoder_depth=2, t_encoder_num_heads=8,
		dim_feedforward=2048, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model


##################################
#           ActionMAE            #
##################################
def actionmae_tiny(**kwargs):  # params: 15,334,008
	model = ActionMAE(
		s_encoder='resnet18',
		t_encoder_embed_dim=512, t_encoder_depth=2, t_encoder_num_heads=8,
		encoder_embed_dim=512, encoder_depth=2, encoder_num_heads=8,
		decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=8,
		mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model


def actionmae_small(**kwargs):  # params: 66,657,144
	model = ActionMAE(
		s_encoder='resnet34',
		t_encoder_embed_dim=512, t_encoder_depth=2, t_encoder_num_heads=8,
		encoder_embed_dim=512, encoder_depth=2, encoder_num_heads=8,
		decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=8,
		mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model


def actionmae_base(**kwargs):  # params: 130,308,216
	model = ActionMAE(
		s_encoder='resnet50',
		t_encoder_embed_dim=512, t_encoder_depth=2, t_encoder_num_heads=8,
		encoder_embed_dim=512, encoder_depth=2, encoder_num_heads=8,
		decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=8,
		mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model


def build_model(args):
	model = eval(f'{args.model}_{args.model_size}')

	if args.dataset == 'ntu_rgbd60': num_classes = 60
	elif args.dataset == 'ntu_rgbd120': num_classes = 120
	elif args.dataset == 'nw_ucla': num_classes = 10
	elif args.dataset == 'uwa3d': num_classes = 30
	else: raise NotImplementedError

	net = model(
		modality=args.modality, num_classes=num_classes, img_size=args.img_size,
		imagenet_pretrained=args.imagenet_pretrained, fusion=args.fusion,
		num_mem_token=args.num_mem_token
	)
	return net