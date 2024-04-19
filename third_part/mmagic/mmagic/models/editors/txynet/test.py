
from mmagic.models.editors.txynet.txy_avgpool2d import Local_Base
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from timm.models.layers import DropPath

from mmagic.registry import MODELS
from  mmagic.models.editors.txynet.module import  CoordAtt, TXYBlock, txy_conv3


class TXYMD(BaseModule):
  

    def __init__(self,
                 in_channels,
                 DW_Expand=2,
                 drop_out_rate=0.):
        super().__init__()

        # Part 1

        dw_channel = in_channels * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dw_channel,###
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True)
        self.conv2 = txy_conv3(dw_channel,2,'split_cat')
        
        self.se = CoordAtt(dw_channel)
        
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel ,
            out_channels=in_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True)

        # Part 2

        self.conv4 = TXYBlock(in_channels,3)
        
        # Dropout
        self.dropout1 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()


        # Feature weight ratio
        self.beta = nn.Parameter(
            torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(
            torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        
        

    def forward(self, inp):
        # part 1
        x = inp
        
        x = self.conv1(x)
        # print(x.shape)##1,128,64,64
        x = self.conv2(x)
        
        x = self.se(x)
       
        
        x = x * self.se(x)
        
        
        x = self.conv3(x)
       

        x = self.dropout1(x)
        y = inp + x * self.beta
        
        # part 2
       
        x = self.conv4(x)
        x = self.conv4(x)
      
        x = self.dropout2(x)
        out = y + x * self.gamma

        return out




if __name__ == '__main__':
    block = TXYMD(64).cuda()  # 实例化模型
    input = torch.rand(1, 64, 416, 416).cuda()  # 创建输入张量
    output = block(input)  # 执行前向传播
    # print(output.shape)  
