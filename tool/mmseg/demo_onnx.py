import cv2
import time
from bubble_seg_onnx import BubbleSegmentONNX
if __name__ == "__main__":
    onnx_path = "/home/txy/code/TXY_code/onnx/mmseg/kent-640-sim.onnx"
    img_path = "pic/mmseg/894.png"
    img = cv2.imread(img_path)
   
    segmenter = BubbleSegmentONNX(onnx_path, batch_size=1, target_size=640, overlap=64)
    
    start_time = time.time()
    segmented_mask = segmenter.segment(img)
    end_time = time.time()
    
    time = (end_time - start_time)
    print(f"Inference time: {time} seconds")
    
    
   
    visualized_result = segmenter.visualize(img, segmented_mask)
    cv2.imwrite("output/mmseg-pred/批量-new——静态.png", visualized_result)
   