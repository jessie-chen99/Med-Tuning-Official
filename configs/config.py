from yacs.config import CfgNode as CN

_C = CN()

# Datasets Parameters
_C.DATA = CN()
_C.DATA.INPUT_CHANNEL=4
_C.DATA.NUM_CLASS=4


_C.MODEL = CN()

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 2, 2]
_C.MODEL.SWIN.DEPTHS_DECODER = [2, 2, 2, 1]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 4
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.DROP_RATE = 0
_C.MODEL.SWIN.DROP_PATH_RATE = 0.1
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.PRE_CKPT='../../ckpt/swin_tiny_patch4_window7_224.pth'

#ViT parameters
_C.MODEL.VIT = CN()
_C.MODEL.VIT.PRE_CKPT=''
_C.MODEL.VIT.PATCH_SIZE=16
_C.MODEL.VIT.EMBED_DIM=768
_C.MODEL.VIT.DEPTH=12
_C.MODEL.VIT.NUM_HEADS=12
_C.MODEL.VIT.PRE_CKPT_SUP='../../ckpt/ViT-B_16.npz' 

# Test Parameters
_C.TEST = CN()
_C.TEST.input_C = 4
_C.TEST.input_H = 240
_C.TEST.input_W = 240
_C.TEST.input_D = 155

# Test Parameters
_C.TRAIN = CN()
_C.TRAIN.input_size = 128



def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  return _C.clone()

def get_model_config(args):
    config = get_cfg_defaults()
    config.freeze()
    if args.backbone_type not in 'ViTB16':
        raise ValueError("backbone_type '{}' is not supported".format(args.backbone_type))
    
    return config
