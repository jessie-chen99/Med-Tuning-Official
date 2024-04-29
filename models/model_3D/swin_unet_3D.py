import copy
import math
import warnings
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from ..model_base.swin_unet import SwinUnet, PatchEmbed, PatchMerging, BasicLayer
from ..model_base.insert_blocks import MedAdapter
from configs.config import get_cfg_defaults

config = get_cfg_defaults()

warnings.filterwarnings('ignore')

class SwinUnet_3D_All(nn.Module): 
    def __init__(self, transfer_type, args):
        super(SwinUnet_3D_All, self).__init__()
        self.transfer_type = transfer_type
        self.args = args
        if self.transfer_type == "full_finetuning_3D":
            self.swin_unet = Baseline_SwinUnet_3D(img_size= config.TRAIN.input_size,
                                in_chans= config.DATA.INPUT_CHANNEL,
                                num_classes= config.DATA.NUM_CLASS,
                                patch_size= config.MODEL.SWIN.PATCH_SIZE,
                                embed_dim = config.MODEL.SWIN.EMBED_DIM,
                                depths = config.MODEL.SWIN.DEPTHS,
                                depths_decoder= config.MODEL.SWIN.DEPTHS_DECODER,
                                num_heads= config.MODEL.SWIN.NUM_HEADS,
                                window_size= config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio= config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias= config.MODEL.SWIN.QKV_BIAS,
                                qk_scale= config.MODEL.SWIN.QK_SCALE,
                                drop_rate= config.MODEL.SWIN.DROP_RATE,
                                drop_path_rate= config.MODEL.SWIN.DROP_PATH_RATE,
                                ape= config.MODEL.SWIN.APE,
                                patch_norm= config.MODEL.SWIN.PATCH_NORM)
        
        elif self.transfer_type == "scratch":
            self.swin_unet = Baseline_SwinUnet_3D(img_size=config.TRAIN.input_size,
                                in_chans= config.DATA.INPUT_CHANNEL,
                                num_classes= config.DATA.NUM_CLASS,
                                patch_size= config.MODEL.SWIN.PATCH_SIZE,
                                embed_dim = config.MODEL.SWIN.EMBED_DIM,
                                depths = config.MODEL.SWIN.DEPTHS,
                                depths_decoder= config.MODEL.SWIN.DEPTHS_DECODER,
                                num_heads= config.MODEL.SWIN.NUM_HEADS,
                                window_size= config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio= config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias= config.MODEL.SWIN.QKV_BIAS,
                                qk_scale= config.MODEL.SWIN.QK_SCALE,
                                drop_rate= config.MODEL.SWIN.DROP_RATE,
                                drop_path_rate= config.MODEL.SWIN.DROP_PATH_RATE,
                                ape= config.MODEL.SWIN.APE,
                                patch_norm= config.MODEL.SWIN.PATCH_NORM)
        
        elif self.transfer_type == "head":
            self.swin_unet = Baseline_SwinUnet_3D(img_size=config.TRAIN.input_size,
                                in_chans= config.DATA.INPUT_CHANNEL,
                                num_classes= config.DATA.NUM_CLASS,
                                patch_size= config.MODEL.SWIN.PATCH_SIZE,
                                embed_dim = config.MODEL.SWIN.EMBED_DIM,
                                depths = config.MODEL.SWIN.DEPTHS,
                                depths_decoder= config.MODEL.SWIN.DEPTHS_DECODER,
                                num_heads= config.MODEL.SWIN.NUM_HEADS,
                                window_size= config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio= config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias= config.MODEL.SWIN.QKV_BIAS,
                                qk_scale= config.MODEL.SWIN.QK_SCALE,
                                drop_rate= config.MODEL.SWIN.DROP_RATE,
                                drop_path_rate= config.MODEL.SWIN.DROP_PATH_RATE,
                                ape= config.MODEL.SWIN.APE,
                                patch_norm= config.MODEL.SWIN.PATCH_NORM)

        elif self.transfer_type == "med_adapter":
            self.swin_unet = MedAdapter_SwinUnet_3D(ratio=6, 
                                img_size=config.TRAIN.input_size,
                                in_chans= config.DATA.INPUT_CHANNEL,
                                num_classes= config.DATA.NUM_CLASS,
                                patch_size= config.MODEL.SWIN.PATCH_SIZE,
                                embed_dim = config.MODEL.SWIN.EMBED_DIM,
                                depths = config.MODEL.SWIN.DEPTHS,
                                depths_decoder= config.MODEL.SWIN.DEPTHS_DECODER,
                                num_heads= config.MODEL.SWIN.NUM_HEADS,
                                window_size= config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio= config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias= config.MODEL.SWIN.QKV_BIAS,
                                qk_scale= config.MODEL.SWIN.QK_SCALE,
                                drop_rate= config.MODEL.SWIN.DROP_RATE,
                                drop_path_rate= config.MODEL.SWIN.DROP_PATH_RATE,
                                ape= config.MODEL.SWIN.APE,
                                patch_norm= config.MODEL.SWIN.PATCH_NORM)    

        else:
            raise ValueError("transfer type '{}' is not supported".format(self.transfer_type))
    
    def forward(self, x):
        logits = self.swin_unet(x)
        return logits

    def load_from(self, pretrain_ckpt_type=None):
        pretrained_path = config.MODEL.SWIN.PRE_CKPT
        if pretrained_path is not None:
            logging.info("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                return
            pretrained_dict = pretrained_dict['model']
            
            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)

            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})

            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        del full_dict[k]
            
            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            logging.info("Successfully load pretrained model of swin encoder")
        else:
            logging.info("none pretrain_ckpt to load")




class PatchEmbed_3D(PatchEmbed):
    """ 3D Image to Patch Embedding"""
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None
            ):
        super().__init__(
            img_size, patch_size, in_chans, embed_dim, norm_layer)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.transpose(1,2).contiguous().reshape(B*T,C,H,W)

        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous() 
        if self.norm is not None:
            x = self.norm(x)
        return x



class Baseline_SwinUnet_3D(SwinUnet):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__(img_size, patch_size, in_chans, num_classes,
                 embed_dim, depths, depths_decoder, num_heads,
                 window_size, mlp_ratio, qkv_bias, qk_scale,
                 drop_rate, attn_drop_rate, drop_path_rate,
                 norm_layer, ape, patch_norm,
                 use_checkpoint, final_upsample)

        self.img_size  = img_size

        self.patch_embed = PatchEmbed_3D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

    def reshape_2d_to_3d(self, x):
        B_,C,H,W = x.shape
        T = self.img_size
        x = x.reshape(int(B_/T),T,C,H,W).transpose(1, 2).contiguous()
        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x,x_downsample)
        x = self.up_x4(x)
        x = F.softmax(x, dim=1)

        x = self.reshape_2d_to_3d(x)
        return x



class BasicLayer_MedAdapter(BasicLayer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__(dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                 drop_path, norm_layer, downsample, use_checkpoint)

        self.med_adapter = nn.Sequential(*[
                    MedAdapter(in_chan=dim, 
                            out_chan=dim, 
                            ratio=6,
                            inter_stage=True if i==1 else False)  # 0:no-interstage, 1:inter stage
                    for i in range(depth)])  # usually depth==0 or 1     

    def forward(self, x, block_attn):
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            
            x, block_attn = self.med_adapter[i](x, block_attn)

        if self.downsample is not None:
            x = self.downsample(x)
        return x, block_attn


class MedAdapter_SwinUnet_3D(Baseline_SwinUnet_3D):
    def __init__(self, ratio=6, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__(img_size, patch_size, in_chans, num_classes,
                 embed_dim, depths, depths_decoder, num_heads,
                 window_size, mlp_ratio, qkv_bias, qk_scale,
                 drop_rate, attn_drop_rate, drop_path_rate,
                 norm_layer, ape, patch_norm,
                 use_checkpoint, final_upsample)

        self.ratio = ratio

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_MedAdapter(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer], 
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=self.dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        B,L,C=x.shape
        block_attn = torch.zeros([
            int(B/config.TRAIN.input_size),
            int(C/(2*self.ratio)),
            config.TRAIN.input_size,
            int(math.sqrt(L)*2),
            int(math.sqrt(L)*2)],
            dtype=torch.float,
            device='cuda')

        for layer in self.layers:
            x_downsample.append(x)
            x, block_attn = layer(x, block_attn)

        x = self.norm(x)
  
        return x, x_downsample
