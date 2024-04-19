import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from mmagic.registry import MODELS


@MODELS.register_module()
class TXYLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0, toY: bool = False):
        super(TXYLoss, self).__init__()
        self.loss_weight = loss_weight
        self.toY = toY
        self.psnr_loss = PSNRTXYLoss(loss_weight, toY)
        self.vgg = vgg19(pretrained=True).features[:16]
        self.vgg.eval()  # Set VGG to evaluation mode
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.edge_loss = EdgeTXYLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # PSNR Loss
        psnr_loss = self.psnr_loss(pred, target)
        
        # Content Loss (VGG)
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        content_loss = F.mse_loss(pred_features, target_features)
        
        # Edge Loss
        edge_loss = self.edge_loss(pred, target)
        
        # Combine losses
        combined_loss = psnr_loss + content_loss + edge_loss
        return combined_loss

class PSNRTXYLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0, toY: bool = False):
        super(PSNRTXYLoss, self).__init__()
        self.loss_weight = loss_weight
        self.toY = toY
        self.scale = 10 / torch.log(torch.tensor(10.0))
        if toY:
            self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.toY:
            pred = torch.sum(pred * self.coef, dim=1, keepdim=True)
            target = torch.sum(target * self.coef, dim=1, keepdim=True)
        mse = F.mse_loss(pred, target, reduction='mean')
        psnr = self.scale * torch.log(1.0 / mse)
        return self.loss_weight * psnr

class EdgeTXYLoss(nn.Module):
    def __init__(self):
        super(EdgeTXYLoss, self).__init__()
        # Sobel operator for x and y direction
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape((1, 1, 3, 3))
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape((1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Expand Sobel filters to match the input channels
        sobel_x = self.sobel_x.repeat(pred.size(1), 1, 1, 1).to(pred.device)
        sobel_y = self.sobel_y.repeat(pred.size(1), 1, 1, 1).to(pred.device)
        
        # Group convolution to apply Sobel operator on each channel independently
        edge_pred_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.size(1))
        edge_pred_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.size(1))
        edge_target_x = F.conv2d(target, sobel_x, padding=1, groups=target.size(1))
        edge_target_y = F.conv2d(target, sobel_y, padding=1, groups=target.size(1))
        
        # Calculate MSE loss for edges in both x and y directions
        edge_loss_x = F.mse_loss(edge_pred_x, edge_target_x)
        edge_loss_y = F.mse_loss(edge_pred_y, edge_target_y)
        
        # Average the losses from x and y directions
        edge_loss = (edge_loss_x + edge_loss_y) / 2
        return edge_loss