import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from typing import List
from timm.models.layers import DropPath
from torch import Tensor

####coordatt############
class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0
    
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = Hsigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
    
class CoordAtt(BaseModule):
    """_summary_

    Args:
        直接对 x_h 和 x_w 应用注意力机制,避免了将 x_h 和 x_w 合并成一个张量然后再分割。
        相反，对每个张量分别应用了卷积、批归一化和激活函数，
        然后再应用了注意力机制。这样，不需要担心分割尺寸不匹配的问题，
        同时仍然能够实现对高度和宽度信息的注意力机制
    """
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Apply convolutions separately instead of after concatenation
        x_h = self.conv1(x_h)
        x_h = self.bn1(x_h)
        x_h = self.act(x_h)

        x_w = self.conv1(x_w)
        x_w = self.bn1(x_w)
        x_w = self.act(x_w)

        # Permute x_w back to match dimensions
        x_w = x_w.permute(0, 1, 3, 2)

        # Apply attention mechanism separately
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Apply attention maps separately and then multiply
        out = identity * a_h * a_w

        return out

################################block1#####
class txy_conv3(BaseModule):
    def __init__(self, in_channels, n_div, forward):
        """
        初始化函数
        :param dim: 输入通道的维度
        :param n_div: 输入维度划分的份数，用于确定哪一部分通道会应用卷积
        :param forward: 指定前向传播的模式，'slicing' 或 'split_cat'
        """
        super().__init__()
        self.dim_conv3 = in_channels // n_div  # 应用卷积的通道数
        self.dim_untouched = in_channels - self.dim_conv3  # 保持不变的通道数
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)  # 部分应用的3x3卷积

        # 根据forward参数，选择前向传播的方式
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        """
        利用slicing方法的前向传播，主要用于推理
        :param x: 输入特征图
        :return: 输出特征图，部分通道被卷积处理
        """
        x = x.clone()  # 克隆输入以保持原输入不变，用于后续的残差连接
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        """
        利用split和cat方法的前向传播，可用于训练/推理
        :param x: 输入特征图
        :return: 输出特征图，部分通道被卷积处理，剩余通道保持不变
        """
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)  # 将输入特征图分为两部分
        x1 = self.partial_conv3(x1)  # 对第一部分应用卷积
        x = torch.cat((x1, x2), 1)  # 将处理后的第一部分和未处理的第二部分拼接
        return x
  
  
  
  
class TXYmodule(BaseModule):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
       # 重新计算chunk_dim，确保它是正整数
        chunk_dim = dim // n_levels
        remainder = dim % n_levels
        if remainder > 0:
           chunk_dim += 1

        # Spatial Weighting：针对每个尺度的特征，使用深度卷积进行空间加权
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # Feature Aggregation：用于聚合不同尺度处理过的特征
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation：使用GELU激活函数
        self.act = nn.GELU()
        

    def forward(self, x):
        h, w = x.size()[-2:]
        # 将输入特征在通道维度上分割成n_levels个尺度
        xc = x.chunk(self.n_levels, dim=1)
        
        out = []
        for i in range(len(xc)):
            if i > 0:
                # 计算每个尺度下采样后的大小
                p_size = (h // 2**i, w // 2**i)
                # 对特征进行自适应最大池化，降低分辨率
                s = F.adaptive_max_pool2d(xc[i], p_size)
                # 对降低分辨率的特征应用深度卷积
                s = self.mfr[i](s)
                # 使用最近邻插值将特征上采样到原始大小
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                # 第一尺度直接应用深度卷积，不进行下采样
                s = self.mfr[i](xc[i])
            out.append(s)
        
        # 将处理过的所有尺度的特征在通道维度上进行拼接
        out = torch.cat(out, dim=1)
        # 通过1x1卷积聚合拼接后的特征
        out = self.aggr(out)
        # 应用GELU激活函数并与原始输入相乘，实现特征调制
        out = self.act(out) * x
        return out


class TXYBlock(BaseModule):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0):
        super(TXYBlock, self).__init__()
        hidden_channel = int(in_channels * 2)
        self.txy = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channel, 3, 1, 1),
                nn.Conv2d(hidden_channel, in_channels, 1, 1, 0))
        self.moudel = TXYmodule(in_channels)
        
        

    def forward(self, x):
        x = self.moudel(x) + x
        x = self.txy(x) + x
        return x
        

if __name__ == '__main__':
    # block = txy_conv3(64,2,"split_cat").cuda()  # 实例化模型
    # input = torch.rand(1, 64, 64, 64).cuda()  # 创建输入张量
    # output = block(input)  # 执行前向传播
    # print(output.shape)
    
    
    # block = TXYBlock(64,3)  # 实例化Coordinate Attention模块
    # input = torch.rand(1, 64, 64, 64)  # 创建一个随机输入
    # output = block(input)  # 通过模块处理输入
    # print(output.shape)
    
    
    block = CoordAtt(3)  # 实例化Coordinate Attention模块
    input = torch.rand(1, 3, 416, 416)  # 创建一个随机输入
    output = block(input)  # 通过模块处理输入
    # print(output.shape)









    