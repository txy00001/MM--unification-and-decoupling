import cv2
from mmseg.apis.inference import inference_model,init_model
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def split_image(image, target_size=640, overlap=64):
    """将大图分割为多个小图，每个小图大小为target_size，且有重叠部分"""
    h, w, _ = image.shape
    patches = []
    for y in range(0, h, target_size - overlap):
        for x in range(0, w, target_size - overlap):
            crop_end_y = min(y + target_size, h)
            crop_end_x = min(x + target_size, w)
            patch = image[y:crop_end_y, x:crop_end_x]
            patches.append((x, y, patch))
    return patches


def process_patches(patches, model):
    """批量处理图像块，返回预测掩膜"""
    pred_masks = []
    for x, y, patch in patches:
        result = inference_model(model, patch)
        pred_mask = result.pred_sem_seg.data[0].cpu().numpy()  # 获取预测掩膜
        pred_masks.append((x, y, pred_mask))
    return pred_masks


def merge_masks(pred_masks, original_shape, overlap=64):
    """合并预测掩膜到一个完整的掩膜图中"""
    pred_mask_full = np.zeros(original_shape[:2], dtype=np.uint8)  # 掩码单通道
    for x, y, pred_mask in pred_masks:
        mask_y_end = min(y + pred_mask.shape[0] - overlap, original_shape[0])
        mask_x_end = min(x + pred_mask.shape[1] - overlap, original_shape[1])
        pred_mask_full[y:mask_y_end, x:mask_x_end] = np.logical_or(pred_mask_full[y:mask_y_end, x:mask_x_end], pred_mask[0:mask_y_end-y, 0:mask_x_end-x])
    return pred_mask_full


class BubbleSegment:
    def __init__(self, config_path: str, weight_path: str):
        self.model = init_model(config_path, weight_path, device="cuda:0")
    
    @classmethod
    def from_dir(cls, config_dir: str):
        conf_parent = Path(config_dir)
        assert conf_parent.exists() and conf_parent.is_dir(), f"{config_dir} is not a directory"
        return cls(str(conf_parent / "config.py"), str(conf_parent / "weights.pth"))
    
    
    def segment(self, image: np.ndarray, crop_size:int = 640, overlap:int = 64):
        patches = split_image(image, crop_size, overlap)
        pred_masks = process_patches(patches, self.model)
        pred_mask_full = merge_masks(pred_masks, image.shape, overlap)
        return pred_mask_full
        
    def visualize(self, image: np.ndarray, result: np.ndarray, opacity: float = 0.3):
        palette = [
            ['background', [127, 127, 127]],
            ['bubble', [0, 0, 255]],
        ]
        palette_dict = {idx: each[1] for idx, each in enumerate(palette)}

        pred_mask_bgr = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for idx, color in palette_dict.items():
            pred_mask_bgr[result == idx] = color
        pred_viz = cv2.addWeighted(image, opacity, pred_mask_bgr, 1-opacity, 0)
        return pred_viz
    