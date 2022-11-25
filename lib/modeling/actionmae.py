from einops import rearrange
import random

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


class ActionMAE(nn.Module):
	""" Masked Autoencoder with ResNet backbone
	"""
	def __init__(self, modality='rgb_depth', num_classes=120,
				 s_encoder='resnet34', imagenet_pretrained=True,
				 t_encoder_dim=512, t_encoder_depth=2, t_encoder_num_heads=8,
				 encoder_dim=512, encoder_depth=2, encoder_num_heads=8,
				 decoder_dim=512, decoder_depth=2, decoder_num_heads=8,
				 dim_feedforward=2048, norm_layer=nn.LayerNorm, norm_pix_loss=True,
				 fusion='sum', num_mem_token=10, **kwargs):
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
		self.t_encoder_pos_embed = PositionEmbeddingSine(t_encoder_dim, normalize=False, cls_token=True, num_cls_token=1)
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
		# ActionMAE encoder specifics
		self.num_mem_token = num_mem_token
		self.actionmae_pos_embed = PositionEmbeddingSine(encoder_dim, normalize=False, cls_token=True, num_cls_token=num_mem_token)
		if self.num_mem_token > 0:
			self.mem_token = nn.Parameter(torch.zeros(1, num_mem_token, encoder_dim))
		self.encoder = Transformer(encoder_dim, encoder_num_heads, encoder_depth, mlp_ratio,
								   norm_layer, drop=0., attn_drop=0., drop_path=0., qkv_bias=True)
		self.encoder_norm = norm_layer(encoder_dim)
		# --------------------------------------------------------------------------

		# --------------------------------------------------------------------------
		# ActionMAE decoder specifics
		self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
		self.decoder = Transformer(decoder_dim, decoder_num_heads, decoder_depth, mlp_ratio,
								   norm_layer, drop=0., attn_drop=0., drop_path=0., qkv_bias=True)
		self.decoder_norm = norm_layer(decoder_dim)

		self.decoder_pred = nn.Linear(decoder_dim, encoder_dim, bias=True)  # decoder to modality token
		self.norm_pix_loss = norm_pix_loss
		# --------------------------------------------------------------------------
		
		# --------------------------------------------------------------------------
		# fusion encoder
		if self.fusion == 'sum':
			self.fusion_module = SumFusion(input_dim=t_encoder_dim, output_dim=num_classes, modality=self.modality)
		elif self.fusion == 'concat':
			self.fusion_module = ConcatFusion(input_dim=t_encoder_dim, output_dim=num_classes, modality=self.modality)
		elif self.fusion == 'transformer':
			self.fusion_module = TransformerFusion(embed_dim=t_encoder_dim, num_heads=8, depth=4, dim_feedforward=2048,
												   norm_layer=norm_layer, drop=0., attn_drop=0., drop_path=0., qkv_bias=True,
												   output_dim=num_classes, use_cls_tkn=True)
		else:
			raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
		# --------------------------------------------------------------------------

		self.initialize_weights()

	def initialize_weights(self):
		# initialization
		# timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
		for mode in self.modality:
			nn.init.normal_(self.t_cls_token[mode], std=.02)
		if self.num_mem_token > 0:
			nn.init.normal_(self.mem_token, std=.02)
		nn.init.normal_(self.mask_token, std=.02)

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
		mode: in {'rgb', 'depth', 'skeleton'}
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
		# embed tokens
		cls_token = self.t_cls_token[mode]
		cls_token = cls_token.expand(x.shape[0], -1, -1)
		x = torch.cat((cls_token, x), dim=1)

		# apply Transformer blocks
		x = self.t_encoder[mode](x, src_key_padding_mask=mask, pos=pos)
		x = self.t_encoder_norm[mode](x)

		return x

	def forward_encoder(self, x, mask, pos, keep=None):
		if self.num_mem_token > 0:
			mem_pos = pos[:, :self.num_mem_token, :]  # [N, 1, D]
			pos = pos[:, self.num_mem_token: :]  # [N, L, D]
		
		if keep == None:
			x, pos, mask_token, ids_restore = self.random_masking(x, pos)
		else:
			x, pos, mask_token, ids_restore = self.modality_masking(x, pos, keep)

		if self.num_mem_token > 0:
			pos = torch.cat([mem_pos, pos], dim=1)  # [N, L+1, D]

			# append memory token
			mem_token = self.mem_token
			mem_tokens = mem_token.expand(x.shape[0], -1, -1)
			x = torch.cat((mem_tokens, x), dim=1)

		# apply Transformer blocks
		x = self.encoder(x, src_key_padding_mask=mask, pos=pos)
		x = self.encoder_norm(x)

		return x, mask_token, ids_restore

	def forward_decoder(self, x, mask, pos, ids_restore):
		# append mask tokens to sequence
		mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + self.num_mem_token - x.shape[1], 1)
		x_ = torch.cat([x[:, self.num_mem_token:, :], mask_tokens], dim=1)  # no cls token
		x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
		x = torch.cat([x[:, :self.num_mem_token, :], x_], dim=1)  # append cls token

		# apply Transformer blocks
		x = self.decoder(x, src_key_padding_mask=mask, pos=pos)
		x = self.decoder_norm(x)

		# remove cls token
		x = x[:, self.num_mem_token:, :]

		# predictor projection
		recon_x = self.decoder_pred(x)

		return recon_x

	def random_masking(self, x, pos):
		"""
		Perform per-sample random masking by per-sample shuffling.
		Per-sample shuffling is done by argsort random noise.
		x: [N, L, D], sequence
		"""
		N, L, D = x.shape  # batch, length, dim
		len_keep = random.randint(1, L)  # random sample from [1, ... , L-1]

		# sort noise for each sample
		noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
		ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
		ids_restore = torch.argsort(ids_shuffle, dim=1)

		# keep the first subset
		ids_keep = ids_shuffle[:, :len_keep]  # [N, len_keep]
		x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
		pos_masked = torch.gather(pos, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

		# generate the binary mask: 0 is keep, 1 is remove
		mask_token = torch.ones([N, L], device=x.device)
		mask_token[:, :len_keep] = 0

		# unshuffle to get the binary mask
		mask_token = torch.gather(mask_token, dim=1, index=ids_restore)

		return x_masked, pos_masked, mask_token, ids_restore

	def modality_masking(self, x, pos, keep):
		N, L, D = x.shape  # batch, length, dim
		keep = torch.tensor([int(k_) for k_ in keep], device=x.device)  # index to keep; '01' -> [0, 1]

		ids_restore = torch.arange(L, device=x.device).unsqueeze(0).repeat(N, 1)  # [N, L]

		# keep the first subset
		ids_keep = keep.unsqueeze(0).repeat(N, 1)  # [N, len_keep]
		x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
		pos_masked = torch.gather(pos, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

		# generate the binary mask: 0 is keep, 1 is remove
		mask_token = torch.ones([N, L], device=x.device)
		mask_token[:, keep] = 0

		return x_masked, pos_masked, mask_token, ids_restore

	def forward_mask_loss(self, target, output, mask_token):
		"""
		target: [N, L, D]; L = #modality * #frames
		output: [N, L, D]; outputstructed outputs
		mask_token: [N, L], 0 is keep, 1 is remove.
		"""
		if self.norm_pix_loss:
			mean = target.mean(dim=-1, keepdim=True)
			var = target.var(dim=-1, keepdim=True)
			target = (target - mean) / (var + 1.e-6)**.5

		loss = (output - target) ** 2
		loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
		loss = (loss * mask_token).sum() / mask_token.sum()  # mean loss on removed patches
		return loss

	def forward_backbone(self, **model_inputs):
		"""
		inputs: dict
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

		x = torch.cat([x_[:, None, :] for mode, x_ in x.items()], dim=1)  # [N, M, D]

		return inputs, x

	def forward(self, targets, eval_mode=False, **model_inputs):
		inputs, x = self.forward_backbone(**model_inputs)
		hat_x = x 
		
		# position embedding
		N, L, D = x.shape
		mask = torch.ones((N, L), device=x.device)
		pos, mask = self.actionmae_pos_embed(x, mask)

		if not eval_mode:
			# actionmae encoder-decoder
			###########################################################################################
			# encoder
			x, mask_token, ids_restore = self.forward_encoder(x, mask, pos)

			# decoder
			loss_mask = torch.tensor(0., device=x.device)
			if x.shape[1] != L+self.num_mem_token:  # no mask; no need to decode
				recon_x = self.forward_decoder(x, mask, pos, ids_restore)  # [N, M, D]
				loss_mask = self.forward_mask_loss(hat_x, recon_x, mask_token)  # [N]
				x = recon_x

			# fusion & prediction
			x = {
				mode: x[:, i, :]
				for i, (mode, _) in enumerate(inputs.items())
			}
			preds = self.fusion_module(x)  # [N, #cls]
			loss_label = self.cross_entropy_loss(preds, targets)
			###########################################################################################

			loss_mask_dict = {'mae': loss_mask}
			loss_label_dict = {'mae': loss_label}
			preds_dict = {'mae': preds}

			return loss_mask_dict, loss_label_dict, preds_dict

		else: 
			loss_mask_dict = {}
			loss_label_dict = {}
			preds_dict = {}

			# for logging purpose
			from itertools import combinations
			modes = []
			combs = []
			modalities = [modality[0] for modality in self.modality]  # ['r', 'd', 'i']
			idxs = [str(idx) for idx in range(len(self.modality))]  # ['0', '1', '2']
			for i in range(1, len(modalities)+1):
				modes.extend(list(combinations(modalities, i)))
				combs.extend(list(combinations(idxs, i)))
			modes = [''.join(m) for m in modes]  # ['r', 'd', 'i', 'rd', 'ri', 'di', 'rdi']
			combs = [''.join(c) for c in combs]  # ['0', '1', '2', '01', '02', '12', '012']

			for k, idx in zip(modes, combs):
				x = hat_x
				# actionmae encoder-decoder
				###########################################################################################
				# encoder
				x, mask_token, ids_restore = self.forward_encoder(x, mask, pos, keep=idx)

				# decoder
				loss_mask = torch.tensor(0., device=x.device)
				if x.shape[1] != L+self.num_mem_token:  # no mask; no need to decode
					recon_x = self.forward_decoder(x, mask, pos, ids_restore)  # [N, M, D]
					loss_mask = self.forward_mask_loss(hat_x, recon_x, mask_token)  # [N]
					x = recon_x

				# fusion & prediction
				x = {
					mode: x[:, i, :]
					for i, (mode, _) in enumerate(inputs.items())
				}
				preds = self.fusion_module(x)  # [N, #cls]
				loss_label = self.cross_entropy_loss(preds, targets)
				###########################################################################################

				loss_mask_dict[k] = loss_mask
				loss_label_dict[k] = loss_label
				preds_dict[k] = preds

			return loss_mask_dict, loss_label_dict, preds_dict
