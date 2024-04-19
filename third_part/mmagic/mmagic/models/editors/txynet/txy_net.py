
from mmagic.models.editors.txynet.module import CoordAtt, TXYBlock
from mmagic.models.editors.txynet.test import TXYMD
from mmagic.models.editors.txynet.txy_avgpool2d import Local_Base
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from timm.models.layers import DropPath

from mmagic.registry import MODELS



@MODELS.register_module(name='TXYNet', force=True)
class TXYNet(BaseModule):
   def __init__(self, img_channels, mid_channels, enc_blk_nums, dec_blk_nums, middle_blk_num):
       super().__init__()
       self.intro = TXYBlock(img_channels, 3)
       self.ending = TXYBlock(img_channels, 3)
 
       self.encoders = nn.ModuleList()
       self.decoders = nn.ModuleList()
       self.middle_blks = nn.ModuleList()
       self.ups = nn.ModuleList()
       self.downs = nn.ModuleList()
 
       chan = img_channels
       for num in enc_blk_nums:
           self.encoders.append(nn.Sequential(*[TXYMD(chan) for _ in range(num)]))
           self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
           chan *= 2
 
       self.middle_blks = nn.Sequential(*[TXYMD(chan) for _ in range(middle_blk_num)])
 
       for num in dec_blk_nums:
           self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
           chan //= 2
           self.decoders.append(nn.Sequential(*[TXYMD(chan) for _ in range(num)]))
 
       # 计算下采样的最大步长，以确保输入图像尺寸可以被正确处理
       self.padder_size = 2 ** len(enc_blk_nums)
 
   def forward(self, inp):
       B, C, H, W = inp.shape
       inp = self.check_image_size(inp)
 
       x = self.intro(inp)
       encs = []
 
       for encoder, down in zip(self.encoders, self.downs):
           x = encoder(x)
           encs.append(x)
           x = down(x)
     
 
       x = self.middle_blks(x)
 
       for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
           x = up(x)
           x = x + enc_skip
           x = decoder(x)
       
       x = self.ending(x)
       x = x + inp
 
       return x[:, :, :H, :W]
 
   def check_image_size(self, x):
       _, _, h, w = x.size()
       mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
       mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
       x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
       return x
   
   
   
@MODELS.register_module(name='TXYNetLocal', force=True)
class TXYNetLocal(Local_Base, TXYNet):

    def __init__(self,
                 *args,
                 train_size=(1, 3, 416, 416),
                 fast_imp=False,
                 **kwargs):
        Local_Base.__init__(self)
        TXYNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(
                base_size=base_size, train_size=train_size, fast_imp=fast_imp)
 
# 示例用法
if __name__ == '__main__':
   img_channels = 3
   mid_channels = 32
   enc_blk_nums = [3, 3, 3]
   dec_blk_nums = [1, 1, 1]
   middle_blk_num = 1
 
   net = TXYNet(img_channels=img_channels, mid_channels=mid_channels, enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums, middle_blk_num=middle_blk_num)
 
   batch_size, channels, height, width = 1, img_channels, 416, 416
   inp = torch.randn(batch_size, channels, height, width)
 
   output = net(inp)
   print(output.shape)
 
    
    
    
    
        





