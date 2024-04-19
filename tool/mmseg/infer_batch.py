import os
import numpy as np
import cv2
from tqdm import tqdm

from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv

import matplotlib.pyplot as plt

config_file = 'configs/config_mmseg/qx_seg_config_2.py'

# 模型 checkpoint 权重文件
checkpoint_file = 'work_dirs/bubble_2/best_mIoU_iter_29500.pth'

# device = 'cpu'
device = 'cuda:0'

model = init_model(config_file, checkpoint_file, device=device)


palette = [
    ['background', [127,127,127]],
    ['bubble', [0,0,200]],
    
]
palette_dict = {}
for idx, each in enumerate(palette):
    palette_dict[idx] = each[1]
    
    
    
if not os.path.exists('output/mmseg-pred'):
    os.mkdir('output/mmseg-pred')

##输入图片数据集
PATH_IMAGE = '/home/txy/code/TXY_code/pic/mmseg'

os.chdir(PATH_IMAGE)

opacity = 0.5


def process_single_img(img_path, save=False):
    
    img_bgr = cv2.imread(img_path)

    # 语义分割预测
    result = inference_model(model, img_bgr)
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

    # 将预测的整数ID，映射为对应类别的颜色
    pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
    for idx in palette_dict.keys():
        pred_mask_bgr[np.where(pred_mask==idx)] = palette_dict[idx]
    pred_mask_bgr = pred_mask_bgr.astype('uint8')

    # 将语义分割预测图和原图叠加显示
    pred_viz = cv2.addWeighted(img_bgr, opacity, pred_mask_bgr, 1-opacity, 0)
    
    # 保存图像至 output/mmseg-pred 目录
    if save:
        save_path = os.path.join('output/mmseg-pred',img_path.split('/')[-1])
        cv2.imwrite(save_path, pred_viz)
        
        
for each in tqdm(os.listdir()):
    process_single_img(each, save=True)



#测试
os.chdir('output/mmseg-pred')

n = 4

fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(16, 10))

for i, file_name in enumerate(os.listdir()[:n**2]):
    
    img_bgr = cv2.imread(file_name)
    
    # 可视化
    axes[i//n, i%n].imshow(img_bgr[:,:,::-1])
    axes[i//n, i%n].axis('off') # 关闭坐标轴显示
fig.suptitle('Semantic Segmentation Predictions', fontsize=30)
# plt.tight_layout()
plt.savefig('../all.png')
    
    