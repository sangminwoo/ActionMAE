from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from lib.modeling.transformer import build_transformer_encoder
from lib.modeling.transformer_block import Transformer
from lib.modeling.pos_embed import PositionEmbeddingSine, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_multi
from lib.modeling.fusion_modules import SumFusion, ConcatFusion, TransformerFusion


class ConvBlock(nn.Module):
	def __init__(self, in_dim, out_dim, kernel_size=3,
				 batch_norm=True, dropout=0., relu=True):
		super(ConvBlock, self).__init__()
		self.conv = nn.Conv1d(in_dim, out_dim, kernel_size)
		self.batch_norm = batch_norm
		if batch_norm:
			self.BatchNorm = nn.BatchNorm1d(in_dim)
		self.relu = relu
		self.dropout = nn.Dropout(dropout) if dropout else None

	def forward(self, x):
		x = self.conv(x)
		if self.batch_norm:
			x = self.BatchNorm(x)
		if self.relu:
			x = F.relu(x, inplace=True)
		if self.dropout:
			x = self.dropout(x)

		return x


class BaselineResNet(nn.Module):
	""" ResNet baseline
	"""
	def __init__(self, modality='rgb_depth', num_classes=120,
				 s_encoder='resnet34', imagenet_pretrained=True,
				 t_encoder_dim=512, t_encoder_depth=2, t_encoder_num_heads=8,
				 dim_feedforward=2048, norm_layer=nn.LayerNorm, fusion='sum', **kwargs):
		super().__init__()
		self.modality = modality.split('_')
		self.fusion = fusion
		self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

		# --------------------------------------------------------------------------
		# spatial encoder specifics
		if 'rgb' in self.modality or 'depth' in self.modality or 'ir' in self.modality:
			self.s_encoder = nn.ModuleDict({
				mode: timm.create_model(s_encoder, pretrained=imagenet_pretrained)
				for mode in self.modality
			})
		if 'skeleton' in self.modality:
			self.skel_embed = nn.Linear(3, 512)
			num_layers = 10
			relu_args = [True] * num_layers
			relu_args[num_layers-1] = False
			dropout_args = [0.1] * num_layers
			for i in range(num_layers):
				if i % 2 == 1:
					dropout_args[i] = 0
			self.skel_encoder = nn.Sequential(*[
				ConvBlock(512, 512, kernel_size=3, batch_norm=True, dropout=dropout_args[i], relu=relu_args[i]) 
				for i in range(num_layers)
			])
		# --------------------------------------------------------------------------

		# --------------------------------------------------------------------------
		# temporal encoder specifics
		mlp_ratio = dim_feedforward // t_encoder_dim
		self.t_cls_token = nn.ParameterDict({
			mode: nn.Parameter(torch.zeros(1, 1, t_encoder_dim))
			for mode in self.modality
		})
		self.t_encoder_pos_embed = PositionEmbeddingSine(t_encoder_dim, normalize=False, cls_token=True)
		self.t_encoder = nn.ModuleDict({
			mode: Transformer(t_encoder_dim, t_encoder_num_heads, t_encoder_depth, mlp_ratio,
							  norm_layer, drop=0., attn_drop=0., drop_path=0.1, qkv_bias=True)
			for mode in self.modality
		})
		# self.t_encoder = nn.ModuleDict({
		# 	mode: build_transformer_encoder(t_encoder_dim, t_encoder_num_heads, t_encoder_depth, dim_feedforward,
		# 									dropout=0.1, activation="gelu", normalize_before=False)
		# 	for mode in self.modality
		# })
		self.t_encoder_norm = nn.ModuleDict({
			mode : norm_layer(t_encoder_dim)
			for mode in self.modality
		})
		# --------------------------------------------------------------------------

		# --------------------------------------------------------------------------
		# fusion encoder
		if self.fusion == 'sum':
			self.fusion_module = SumFusion(input_dim=t_encoder_dim, output_dim=num_classes, modality=self.modality)
		elif self.fusion == 'concat':
			self.fusion_module = ConcatFusion(input_dim=t_encoder_dim, output_dim=num_classes, modality=self.modality)
		elif self.fusion == 'transformer':
			self.fusion_module = TransformerFusion(embed_dim=t_encoder_dim, num_heads=8, depth=4, dim_feedforward=2048,
												   norm_layer=norm_layer, output_dim=num_classes, use_cls_tkn=True)
		else:
			raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
		# --------------------------------------------------------------------------

		self.initialize_weights()

	def initialize_weights(self):
		# initialization
		# timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
		for mode in self.modality:
			nn.init.normal_(self.t_cls_token[mode], std=.02)

		# initialize nn.Linear and nn.Layernorm
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			# we use xavier uniform following official JAX ViT:
			nn.init.xavier_uniform_(m.weight)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)

	def forward_spatial_encoder(self, x, mode):
		"""
		x: [N*T, C, H, W]
		mode: in {'rgb', 'depth', 'skeleton', 'ir'}
		"""
		if mode == 'skeleton':
			x = self.skel_embed(x)  # [N*T, 25, 3] -> [N*T, 25, 512]
			x = x.transpose(1, 2)  # [N*T, 512, 25]
			x = self.skel_encoder(x)  # [N*T, 5, 512]
			x = F.adaptive_avg_pool1d(x, 1)  # [N*T, 1, 512]
			x = x.squeeze()  # [N*T, 512]
		else:
			x = self.s_encoder[mode].forward_features(x)
			x = F.adaptive_avg_pool2d(x, 1)
			x = x.squeeze()

		return x

	def forward_temporal_encoder(self, x, mask, pos, mode):
		# append cls token
		cls_token = self.t_cls_token[mode]
		cls_token = cls_token.expand(x.shape[0], -1, -1)
		x = torch.cat((cls_token, x), dim=1)

		# apply Transformer blocks
		x = self.t_encoder[mode](x, src_key_padding_mask=mask, pos=pos)
		x = self.t_encoder_norm[mode](x)

		return x

	def forward(self, targets, **model_inputs):
		"""
		model_inputs: dict
			'src_rgb': [N, T, 3, H, W]
			'src_depth': [N, T, 3, H, W]
			'src_ir': [N, T, 3, H, W]
			'src_skeleton': [N, T, 25, 3]
		"""
		inputs = {}
		masks = {}
		if 'src_rgb' in model_inputs:
			N, T, C, H, W = model_inputs['src_rgb'].shape
			inputs['rgb'] = rearrange(model_inputs['src_rgb'], 'N T C H W -> (N T) C H W')
			masks['rgb'] = model_inputs['src_rgb_mask']

		if 'src_depth' in model_inputs:
			N, T, C, H, W = model_inputs['src_depth'].shape
			inputs['depth'] = rearrange(model_inputs['src_depth'], 'N T C H W -> (N T) C H W')
			masks['depth'] = model_inputs['src_depth_mask']

		if 'src_ir' in model_inputs:
			N, T, C, H, W = model_inputs['src_ir'].shape
			inputs['ir'] = rearrange(model_inputs['src_ir'], 'N T C H W -> (N T) C H W')
			masks['ir'] = model_inputs['src_ir_mask']

		if 'src_skeleton' in model_inputs:
			N, T, L, C = model_inputs['src_skeleton'].shape
			inputs['skeleton'] = rearrange(model_inputs['src_skeleton'], 'N T L C -> (N T) L C')  # joints = 25; dimension = 3
			masks['skeleton'] = model_inputs['src_skeleton_mask']
		
		x = {}
		for (mode, input_), (_, mask_) in zip(inputs.items(), masks.items()):
			# spatial encoding
			x_ = self.forward_spatial_encoder(input_, mode)  # [N*T, D]; cls_token
			x_ = x_.reshape(N, T, x_.shape[1])  # [N, T, D]

			# position embedding
			pos, mask_ = self.t_encoder_pos_embed(input_, mask_)
			mask_ = mask_.bool()

			# temporal encoding
			x_ = self.forward_temporal_encoder(x_, ~mask_, pos, mode)  # [N, T+1, D]
			x[mode] = x_[:, 0, :]  # cls_token: [N, D]

		# fusion & prediction
		preds = self.fusion_module(x)  # [N, #cls]
		loss_mask = torch.tensor(0., device=preds.device)  # NO mask loss
		loss_label = self.cross_entropy_loss(preds, targets)

		###########################################################################################
		# out_r, out_d, out_i, out_s are calculated to estimate
		# the performance of 'r', 'd', 'i' and 's' modality.
		feats = {}
		if self.fusion == 'sum':
			feats = {
				mode: (torch.mm(x_, torch.transpose(self.fusion_module.fc[mode].weight, 0, 1)) +
					   self.fusion_module.fc[mode].bias / len(x))
				for mode, x_ in x.items()
			}
		elif self.fusion == 'concat':
			weight_size = self.fusion_module.fc.weight.size(1)
			feats = {
				mode: (torch.mm(x_, torch.transpose(self.fusion_module.fc.weight[:,(weight_size//len(x))*i:(weight_size//len(x))*(i+1)], 0, 1)) +
					   self.fusion_module.fc.bias / len(x))
				for i, (mode, x_) in enumerate(x.items())
			}
		else:
			pass

		# loss_label_unimodal = {
		# 	mode: self.cross_entropy_loss(output, targets)
		# 	for mode, output in outputs
		# }
		###########################################################################################

		loss_mask_dict = {'base': loss_mask}
		loss_label_dict = {'base': loss_label}
		preds_dict = {'base': preds}

		return loss_mask_dict, loss_label_dict, preds_dict, feats
