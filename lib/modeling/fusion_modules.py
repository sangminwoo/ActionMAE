import torch
import torch.nn as nn
from lib.modeling.transformer import build_transformer_encoder
from lib.modeling.transformer_block import Transformer
from lib.modeling.pos_embed import PositionEmbeddingSine
from lib.utils.tensor_utils import pad_sequences_1d


class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=60, modality=['rgb', 'depth']):
        super(SumFusion, self).__init__()
        self.fc = nn.ModuleDict({
            mode: nn.Linear(input_dim, output_dim)
            for mode in modality
        })

    def forward(self, x):
        assert isinstance(x, dict)
        outputs = sum([self.fc[mode](x_) for mode, x_ in x.items()])
        return outputs


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=60, modality=['rgb', 'depth']):
        super(ConcatFusion, self).__init__()
        self.fc = nn.Linear(input_dim*len(modality), output_dim)
        # self.fc = nn.Linear(1536, output_dim)

    def forward(self, x):
        assert isinstance(x, dict)
        x = torch.cat([x_ for mode, x_ in x.items()], dim=1)
        
        #########################################################
        # w_r = self.fc.weight[:,     : 512]        
        # w_d = self.fc.weight[:,  512:1024]        
        # w_i = self.fc.weight[:, 1024:1536]        

        # outputs = torch.mm(
        #     x, torch.transpose(torch.cat([w_r, w_d], dim=1), 0, 1)
        # ) + self.fc.bias * (1/3)
        #########################################################

        outputs = self.fc(x)
        return outputs


class TransformerFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, depth=4, dim_feedforward=2048, norm_layer=nn.LayerNorm, 
                 drop=0., attn_drop=0., drop_path=0., qkv_bias=True, output_dim=60, use_cls_tkn=True):
        super(TransformerFusion, self).__init__()
        if use_cls_tkn:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim
        self.pos_embed = PositionEmbeddingSine(embed_dim, normalize=True, cls_token=True)
        mlp_ratio = dim_feedforward // embed_dim
        self.transformer = Transformer(embed_dim, num_heads, depth, mlp_ratio, qkv_bias=True,
                                       drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        assert isinstance(x, dict)
        x = torch.cat([x_[:, None, :] for mode, x_ in x.items()], dim=1)  # [N, M, D]; M=#modalities
        N, L, D = x.shape
        mask = torch.ones((N, L), device=x.device)

        pos, mask = self.pos_embed(x, mask)
        mask = mask.bool()

        # append cls token
        cls_token = self.cls_token
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # apply Transformer blocks
        x = self.transformer(x, src_key_padding_mask=mask, pos=pos)
        x = x[:, 0, :]  # cls_token: [N, D]

        outputs = self.fc(x)

        return outputs


# class GatedFusion(nn.Module):
#     def __init__(self, input_dim=512, dim=512, output_dim=60):
#         super(GatedFusion, self).__init__()

#         self.fc = nn.ModuleDict({
#             mode: nn.Linear(input_dim, dim)
#             for mode in modality
#         })
#         self.fc_out = nn.Linear(dim, output_dim)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, y):
#         x = {
#             mode: self.fc[mode](x_)
#             for mode, x_ in x.items()
#         }

#         gate_x = {
#             self.sigmoid()

#         }

#         if self.x_gate:
#             gate = self.sigmoid(out_x)
#             output = self.fc_out(torch.mul(gate, out_y))
#         else:
#             gate = self.sigmoid(out_y)
#             output = self.fc_out(torch.mul(out_x, gate))

#         return outputs