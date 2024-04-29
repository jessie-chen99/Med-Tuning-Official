import logging
import math
import torch
import torch.nn as nn
import torch.fft as fft
from configs.config import get_cfg_defaults

config = get_cfg_defaults()
input_img_size= config.TRAIN.input_size

class MedAdapter(nn.Module):
    def __init__(self, in_chan=None, out_chan=None, ratio=6, inter_stage=False):
        super().__init__()
        self.inter_stage = inter_stage
        self.hidden_chan=int(in_chan/ratio)

        self.down_projection = nn.Linear(in_chan, self.hidden_chan)
        self.activate1 = nn.GELU()
        self.dconv_3_1=nn.Conv3d(in_channels=self.hidden_chan, out_channels=self.hidden_chan, kernel_size=(1,3,3), padding=(0,1,1), groups=self.hidden_chan) 
        self.dconv_3_2=nn.Conv3d(in_channels=self.hidden_chan, out_channels=self.hidden_chan, kernel_size=(3,1,1), padding=(1,0,0), groups=self.hidden_chan) 
        self.bn_3 = nn.BatchNorm3d(self.hidden_chan)
        self.activate_3 = nn.ReLU(inplace=True)

        self.dconv_5_1=nn.Conv3d(in_channels=self.hidden_chan, out_channels=self.hidden_chan, kernel_size=(1,5,5), padding=(0,2,2), groups=self.hidden_chan) 
        self.dconv_5_2=nn.Conv3d(in_channels=self.hidden_chan, out_channels=self.hidden_chan, kernel_size=(5,1,1), padding=(2,0,0), groups=self.hidden_chan) 
        self.bn_5 = nn.BatchNorm3d(self.hidden_chan)
        self.activate_5 = nn.ReLU(inplace=True)

        self.conv=nn.Conv3d(in_channels=self.hidden_chan, out_channels=self.hidden_chan, kernel_size=(1,1,1), padding=0, groups=1)
        
        self.frequen_weight = torch.nn.Parameter(torch.ones([1, 1, input_img_size, 1, 1], requires_grad=True))
        self.frequen_bias = torch.nn.Parameter(torch.ones([1, 1, input_img_size, 1, 1], requires_grad=True))
        self.fft = fft.fftn
        self.ifft = fft.ifftn

        self.up_projection = nn.Linear(self.hidden_chan, out_chan)
        self.activate2 = nn.GELU()

        if inter_stage:
            self.attn_conv=nn.Conv3d(in_channels=int(self.hidden_chan/2), out_channels=self.hidden_chan, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
            self.attn_11conv=nn.Conv3d(in_channels=int(self.hidden_chan*2), out_channels=self.hidden_chan, kernel_size=1, stride=1, padding=0)    
            logging.info("this is inter-stage block")
        else:
            logging.info("this is not inter-stage block")

    def forward(self, x, atten_last_stage=None):
        shortcut_out=x

        x = self.down_projection(x)
        x = self.activate1(x)

        B_,L,C = x.shape  
        T=input_img_size
        B=int(B_/T)
        H=int(math.sqrt(L))
        W=int(math.sqrt(L))
        x = x.reshape(B,T,H,W,C).permute(0,4,1,2,3).contiguous()

        shortcut=x

        x1_1 = self.activate_3(self.bn_3(self.dconv_3_2(self.dconv_3_1(x))))
        x1_2 = self.activate_5(self.bn_5(self.dconv_5_2(self.dconv_5_1(x))))

        x2 = self.fft(x, dim=(2,3,4)) 
        x2 = x2*(self.frequen_weight.expand(B,x2.shape[1],T,H,W))+self.frequen_bias.expand(B,x2.shape[1],T,H,W) 
        x2 = torch.abs(self.ifft(x2, dim=(2,3,4)))

        x = self.conv(x1_1 + x1_2 + x2)

        if self.inter_stage:
            attn = self.attn_11conv(torch.cat((x, self.attn_conv(atten_last_stage)), dim=1))
        else:
            attn = x

        x = shortcut+attn

        x = x.permute(0,2,3,4,1).contiguous().reshape(B_,L,C)
        
        x = self.up_projection(x)
        x = self.activate2(x)

        x = shortcut_out + x

        if self.inter_stage:
            return x, attn
        else:
            return x, atten_last_stage


class MedAdapter_for_ViT(MedAdapter):
    def __init__(self, in_chan=None, out_chan=None, ratio=6, inter_stage=False):
        super().__init__(in_chan, out_chan, ratio, inter_stage)

        if inter_stage:
            self.attn_conv=nn.Conv3d(in_channels=self.hidden_chan, out_channels=self.hidden_chan, kernel_size=(1,1,1))

