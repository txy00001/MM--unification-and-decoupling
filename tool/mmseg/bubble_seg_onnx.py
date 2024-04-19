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
                patch = np.pad(patch, ((0, target_size - patch.shape[0]), (0, target_size - patch.shape[1]), (0, 0)),
                               mode='constant', constant_values=0)

            patches.append(patch)
    return patches

def process_patches(patches, ort_session, batch_size=1):
    pred_masks = []
    num_batches = (len(patches) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(patches))
        batch_patches = patches[batch_start:batch_end]

        # Ensure all batches have the same size by padding with zeros if necessary
        if len(batch_patches) < batch_size:
            padding = [np.zeros_like(batch_patches[0]) for _ in range(batch_size - len(batch_patches))]
            batch_patches.extend(padding)

        batch_input_tensors = [
            ((cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype('float32') - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375]))
            for patch in batch_patches
        ]
        batch_input_tensors = np.array([np.transpose(tensor, (2, 0, 1)) for tensor in batch_input_tensors], dtype=np.float32)

        # Create input dictionary for ONNX runtime
        ort_input = {'input': batch_input_tensors}
        ort_outputs = ort_session.run(None, ort_input)

        # Collect masks from ONNX output, ignoring padded results
        for ort_output in ort_outputs[0][:len(batch_patches)]:
            pred_mask = ort_output[0]
            pred_masks.append(pred_mask)

    return pred_masks



class BubbleSegmentONNX:
    def __init__(self, onnx_path: str, batch_size=4, target_size=640, overlap=64):
        self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.batch_size = batch_size
        self.target_size = target_size
        self.overlap = overlap

    def segment(self, image: np.ndarray):
        patches = split_image(image, self.target_size, self.overlap)
        pred_masks = process_patches(patches, self.ort_session, self.batch_size)
        pred_mask_full = self.merge_masks(pred_masks, image.shape)
        return pred_mask_full

    def merge_masks(self, pred_masks, original_shape):
        h, w = original_shape[:2]
        pred_mask_full = np.zeros((h, w), dtype=np.uint8)
        idx = 0
        for y in range(0, h, self.target_size - self.overlap):
            for x in range(0, w, self.target_size - self.overlap):
                crop_end_y = min(y + self.target_size, h)
                crop_end_x = min(x + self.target_size, w)
                sub_mask = pred_masks[idx][0:crop_end_y-y, 0:crop_end_x-x]
                enhanced_mask = (sub_mask > 0.5)  # Threshold to handle small bubbles
                pred_mask_full[y:crop_end_y, x:crop_end_x] = np.logical_or(pred_mask_full[y:crop_end_y, x:crop_end_x], enhanced_mask)
                idx += 1
        return pred_mask_full

    def visualize(self, image: np.ndarray, result: np.ndarray, opacity: float = 0.3):
        palette = [['background', [127, 127, 127]], ['bubble', [0, 0, 255]]]
        palette_dict = {idx: each[1] for idx, each in enumerate(palette)}

        pred_mask_bgr = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for idx, color in palette_dict.items():
            pred_mask_bgr[result == idx] = color

        pred_viz = cv2.addWeighted(image, opacity, pred_mask_bgr, 1 - opacity, 0)
        return pred_viz

# Example of using the BubbleSegmentONNX class
if __name__ == "__main__":
    onnx_model_path = 'path_to_your_model.onnx'
    image_path = 'path_to_your_image.jpg'
    image = cv2.imread(image_path)

    segmenter = BubbleSegmentONNX(onnx_model_path, batch_size=4, target_size=640, overlap=64)
    segmented_mask = segmenter.segment(image)
    visualized_result = segmenter.visualize(image, segmented_mask)

    cv2.imshow("Segmented Image", visualized_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
