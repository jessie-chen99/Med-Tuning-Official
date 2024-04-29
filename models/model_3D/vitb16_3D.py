import math
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import Block, adapt_input_conv
from timm.models.layers.trace_utils import _assert
from ..model_base.head import UperNet
from ..model_base.insert_blocks import MedAdapter_for_ViT
from configs.config import get_cfg_defaults
config = get_cfg_defaults()

input_img_size= config.TRAIN.input_size

_logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class ViTUperNet_3D_All(nn.Module):
    def __init__(self, transfer_type, args):
        super(ViTUperNet_3D_All, self).__init__()
        self.transfer_type = transfer_type
        self.args = args
        if self.transfer_type == "full_finetuning_3D":
            self.vit_3D = Baseline_ViT_3D(img_size=config.TRAIN.input_size, 
                                                patch_size=config.MODEL.VIT.PATCH_SIZE, 
                                                in_chans=config.DATA.INPUT_CHANNEL,
                                                num_classes=config.DATA.NUM_CLASS, 
                                                embed_dim=config.MODEL.VIT.EMBED_DIM, 
                                                depth = config.MODEL.VIT.DEPTH,
                                                num_heads=config.MODEL.VIT.NUM_HEADS,
                                                )
        
        elif self.transfer_type == "scratch":
            self.vit_3D = Baseline_ViT_3D(img_size=config.TRAIN.input_size, 
                                                patch_size=config.MODEL.VIT.PATCH_SIZE, 
                                                in_chans=config.DATA.INPUT_CHANNEL,
                                                num_classes=config.DATA.NUM_CLASS, 
                                                embed_dim=config.MODEL.VIT.EMBED_DIM, 
                                                depth = config.MODEL.VIT.DEPTH,
                                                num_heads=config.MODEL.VIT.NUM_HEADS,
                                                )
        
        elif self.transfer_type == "head":
            self.vit_3D = Baseline_ViT_3D(img_size=config.TRAIN.input_size, 
                                                patch_size=config.MODEL.VIT.PATCH_SIZE, 
                                                in_chans=config.DATA.INPUT_CHANNEL,
                                                num_classes=config.DATA.NUM_CLASS, 
                                                embed_dim=config.MODEL.VIT.EMBED_DIM, 
                                                depth = config.MODEL.VIT.DEPTH,
                                                num_heads=config.MODEL.VIT.NUM_HEADS,
                                                )
        
        elif self.transfer_type == "med_adapter":
            self.vit_3D = MedAdapter_ViT_3D(ratio=6,
                                                img_size=config.TRAIN.input_size, 
                                                patch_size=config.MODEL.VIT.PATCH_SIZE, 
                                                in_chans=config.DATA.INPUT_CHANNEL,
                                                num_classes=config.DATA.NUM_CLASS, 
                                                embed_dim=config.MODEL.VIT.EMBED_DIM, 
                                                depth = config.MODEL.VIT.DEPTH,
                                                num_heads=config.MODEL.VIT.NUM_HEADS,
                                                )
        else:
            raise ValueError("transfer type '{}' is not supported".format(self.transfer_type))

    def forward(self, x):
        logits = self.vit_3D(x)
        return logits

    def load_from(self, pretrain_ckpt_type):
        if pretrain_ckpt_type=="supervised":
            logging.info("pretrained_path:{}".format(config.MODEL.VIT.PRE_CKPT_SUP))
            _load_weights(self.vit_3D, config.MODEL.VIT.PRE_CKPT_SUP)
        

class PatchEmbed_3D(PatchEmbed):
    """ 3D Image to Patch Embedding"""
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3,
            embed_dim=768, norm_layer=None, flatten=True, bias=True
            ):
        super().__init__(
            img_size, patch_size, in_chans, 
            embed_dim, norm_layer, flatten, bias)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.transpose(1,2).contiguous().reshape(B*T,C,H,W)
        
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        return x


class Baseline_ViT_3D(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
        - https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    """
    def __init__(
            self,img_size=224,patch_size=16,in_chans=3,num_classes=1000,global_pool='token',embed_dim=768,depth=12,num_heads=12,mlp_ratio=4.,qkv_bias=True,
            init_values=None,class_token=False,no_embed_class=False,pre_norm=False,fc_norm=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.,
            weight_init='',embed_layer=PatchEmbed_3D,norm_layer=nn.LayerNorm, act_layer=nn.GELU,block_fn=Block):
        super().__init__()
        self.use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        self.norm_layer = partial(norm_layer, eps=1e-6)
        self.act_layer = act_layer
        self.patch_size=patch_size
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False
        self.depth = depth
        self.img_size=img_size

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.embed_len = self.num_patches if no_embed_class else self.num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.out_indices = [2, 5, 8, 11]
        self.layers = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=self.dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not self.use_fc_norm else nn.Identity()

        self.decoder=UperNet(num_classes=num_classes, feature_channels=[256, embed_dim, embed_dim, embed_dim])
        self.fpn = nn.Sequential(
                    nn.Conv2d(embed_dim, 256, kernel_size=1, stride=1),
                    nn.SyncBatchNorm(256),
                    nn.GELU(),
                    )

    def _pos_embed(self, x):
        if self.no_embed_class:
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            if self.cls_token is not None:
                x = torch.cat((
                        self.cls_token.expand(x.shape[0], -1, -1), 
                        x)
                    , dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        _, _, _, h, w = x.shape
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        downsamples = []
        for i, blk in enumerate(self.layers):
            x = blk(x)
            if i == len(self.layers) - 1:
                x = self.norm(x)
            if i in self.out_indices:
                B, _, C = x.shape    
                out = x.reshape(B, h // self.patch_size,
                                w // self.patch_size,
                                C).permute(0, 3, 1, 2).contiguous()
                if i == 2:
                    out = self.fpn(out)
                downsamples.append(out)
        return x, downsamples

    def reshape_2d_to_3d(self, x):
        B_,C,H,W = x.shape
        T = self.img_size
        x = x.reshape(int(B_/T),T,C,H,W).transpose(1, 2).contiguous()
        return x
    
    def forward(self, x):
        x, downsamples = self.forward_features(x)

        x = self.decoder(downsamples)

        x = self.reshape_2d_to_3d(x)

        return x


class MedAdapter_ViT_3D(Baseline_ViT_3D):
    def __init__(
            self,ratio=6,img_size=224,patch_size=16,in_chans=3,num_classes=1000,global_pool='token',embed_dim=768,depth=12,num_heads=12,mlp_ratio=4.,qkv_bias=True,
            init_values=None,class_token=False,no_embed_class=False,pre_norm=False,fc_norm=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.,
            weight_init='',embed_layer=PatchEmbed_3D,norm_layer=nn.LayerNorm,act_layer=nn.GELU,block_fn=Block
            ):
        super().__init__(
            img_size,patch_size,in_chans,num_classes,global_pool,embed_dim,depth,num_heads,mlp_ratio,qkv_bias,
            init_values,class_token,no_embed_class,pre_norm,fc_norm,drop_rate,attn_drop_rate,drop_path_rate,
            weight_init,embed_layer,norm_layer,act_layer,block_fn
            )
        
        self.ratio = ratio

        self.med_adapter = nn.Sequential(*[
            MedAdapter_for_ViT(
                in_chan=embed_dim, 
                out_chan=embed_dim, 
                ratio=self.ratio,
                inter_stage=True if i in self.out_indices else False,
            )
            for i in range(self.depth)])
        
        self.decoder=UperNet(num_classes=num_classes, feature_channels=[256, embed_dim, embed_dim, embed_dim])

    def forward_features(self, x):
        _, _, _, h, w = x.shape
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        
        downsamples = []
        
        B,L,C=x.shape
        block_attn = torch.zeros([int(B/input_img_size),int(C/self.ratio), input_img_size, int(math.sqrt(L)),int(math.sqrt(L))],dtype=torch.float,device='cuda')
        
        for i in range(self.depth):
            x = self.layers[i](x)

            x, block_attn = self.med_adapter[i](x, block_attn)

            if i == len(self.layers) - 1:
                x = self.norm(x)
            if i in self.out_indices:
                B, _, C = x.shape    
                out = x.reshape(B, h // self.patch_size,
                                w // self.patch_size,
                                C).permute(0, 3, 1, 2).contiguous()
                if i == 2:
                    out = self.fpn(out)
                downsamples.append(out)

        return x, downsamples




@torch.no_grad()
def _load_weights(model, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = np.ascontiguousarray(w.transpose([3, 2, 0, 1]))
            elif w.ndim == 3:
                w = np.ascontiguousarray(w.transpose([2, 0, 1]))
            elif w.ndim == 2:
                w = np.ascontiguousarray(w.transpose([1, 0]))
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.layers):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))

    for i, block in enumerate(model.layers.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))
    logging.info("Successfully load supervised pretrain_ckpt")

def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        _, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        _, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2).contiguous()
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).contiguous().reshape(1, gs_new[0] * gs_new[1], -1)
    return posemb_grid # posemb
