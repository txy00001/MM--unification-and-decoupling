import cv2
from mmseg.apis.inference import inference_model, init_model
import numpy as np
from pathlib import Path

def split_image(image, target_size=640, overlap=64):
    h, w, _ = image.shape
    patches = []
    for y in range(0, h, target_size - overlap):
        for x in range(0, w, target_size - overlap):
            crop_end_y = min(y + target_size, h)
            crop_end_x = min(x + target_size, w)
            patch = image[y:crop_end_y, x:crop_end_x]

            # 检查patch是否需要填充
            if patch.shape[0] != target_size or patch.shape[1] != target_size:
                # 使用0进行填充以达到目标尺寸
                patch = np.pad(patch, ((0, target_size - patch.shape[0]), (0, target_size - patch.shape[1]), (0, 0)), mode='constant', constant_values=0)

            patches.append(patch)
    return patches

def process_patches(patches, model):
    pred_masks = []
    # 每次处理最多35张图片,(一次张数太多会out of memory)
    for i in range(0, len(patches), 35):
        batch = patches[i:i+35]
        results = inference_model(model, batch)
        for result in results:
            pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
            pred_masks.append(pred_mask)
    return pred_masks

def merge_masks(pred_masks, original_shape, target_size=640, overlap=64):
    h, w = original_shape[:2]
    pred_mask_full = np.zeros((h, w), dtype=np.uint8)
    idx = 0
    for y in range(0, h, target_size - overlap):
        for x in range(0, w, target_size - overlap):
            crop_end_y = min(y + target_size, h)
            crop_end_x = min(x + target_size, w)
            pred_mask_full[y:crop_end_y, x:crop_end_x] = np.logical_or(pred_mask_full[y:crop_end_y, x:crop_end_x], pred_masks[idx][0:crop_end_y-y, 0:crop_end_x-x])
            idx += 1
    return pred_mask_full

class BubbleSegment_batch:
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
        pred_mask_full = merge_masks(pred_masks, image.shape, crop_size, overlap)
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
