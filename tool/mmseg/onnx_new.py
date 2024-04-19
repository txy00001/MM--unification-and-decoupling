import cv2
import numpy as np
import onnxruntime
from pathlib import Path

def split_image(image, target_size=640, overlap=64):
    h, w, _ = image.shape
    patches = []
    for y in range(0, h, target_size - overlap):
        for x in range(0, w, target_size - overlap):
            crop_end_y = min(y + target_size, h)
            crop_end_x = min(x + target_size, w)
            patch = image[y:crop_end_y, x:crop_end_x]

            if patch.shape[0] != target_size or patch.shape[1] != target_size:
                patch = np.pad(patch, ((0, target_size - patch.shape[0]), (0, target_size - patch.shape[1]), (0, 0)), mode='constant', constant_values=0)

            patches.append(patch)
    return patches

def process_patches(patches, ort_session):
    pred_masks = []
    for patch in patches:
        img_tensor = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        img_tensor = (img_tensor - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
        img_tensor = img_tensor.astype('float32')
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(img_tensor, axis=0)

        ort_input = {'input': input_tensor}
        ort_output = ort_session.run(None, ort_input)[0]
        pred_mask = ort_output[0][0]
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
            sub_mask = pred_masks[idx][0:crop_end_y-y, 0:crop_end_x-x]
            enhanced_mask = (sub_mask > 0.5)  # Threshold to handle small bubbles
            pred_mask_full[y:crop_end_y, x:crop_end_x] = np.logical_or(pred_mask_full[y:crop_end_y, x:crop_end_x], enhanced_mask)
            idx += 1
    return pred_mask_full

class BubbleSegmentONNX:
    def __init__(self, onnx_path: str):
        self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def segment(self, image: np.ndarray, crop_size:int = 640, overlap:int = 64):
        patches = split_image(image, crop_size, overlap)
        pred_masks = process_patches(patches, self.ort_session)
        pred_mask_full = merge_masks(pred_masks, image.shape, crop_size, overlap)
        return pred_mask_full

    def visualize(self, image: np.ndarray, result: np.ndarray, opacity: float = 0.3):
        palette = [['background', [127, 127, 127]], ['bubble', [0, 0, 255]]]
        palette_dict = {idx: each[1] for idx, each in enumerate(palette)}

        pred_mask_bgr = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for idx, color in palette_dict.items():
            pred_mask_bgr[result == idx] = color

        pred_viz = cv2.addWeighted(image, opacity, pred_mask_bgr, 1 - opacity, 0)
        return pred_viz