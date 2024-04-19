

import cv2

from bubble_seg_batch import BubbleSegment_batch
if __name__ == "__main__":
    seg = BubbleSegment_batch.from_dir("tool/mmseg/demo_test")
    img_path = "pic/mmseg/894.png"
    img = cv2.imread(img_path)
    res = seg.segment(img, crop_size=640, overlap=64)
    vis = seg.visualize(img, res)
    cv2.imwrite("output/mmseg-pred/demo_result-batch-30.png", vis)