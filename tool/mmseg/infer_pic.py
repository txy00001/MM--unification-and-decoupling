import numpy as np
import matplotlib.pyplot as plt

from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import cv2
config_file = 'configs/config_mmseg/qx_seg_config_2.py'

# 模型 checkpoint 权重文件
checkpoint_file = 'work_dirs/bubble_2/best_mIoU_iter_29500.pth'

# device = 'cpu'
device = 'cuda:0'

model = init_model(config_file, checkpoint_file, device=device)

img_path = '/home/txy/code/TXY_code/test_succ.jpg'

img_bgr = cv2.imread(img_path)



result = inference_model(model, img_bgr)

pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

# plt.figure(figsize=(10, 8))
# plt.imshow(img_bgr[:,:,::-1])
# plt.imshow(pred_mask, alpha=0.55) # alpha 高亮区域透明度，越小越接近原图
# plt.axis('off')
# plt.savefig('output/mmseg/1-1.png')




###与原图并排显示
# plt.figure(figsize=(14, 8))

# plt.subplot(1,2,1)
# plt.imshow(img_bgr[:,:,::-1])
# plt.axis('off')

# plt.subplot(1,2,2)
# plt.imshow(img_bgr[:,:,::-1])
# plt.imshow(pred_mask, alpha=0.6) # alpha 高亮区域透明度，越小越接近原图
# plt.axis('off')
# plt.savefig('outputs/K1-2.jpg')


##叠加在原图显示
palette = [
    ['background', [127,127,127]],
    ['bubble', [0,0,200]],
    
]

palette_dict = {}
for idx, each in enumerate(palette):
    palette_dict[idx] = each[1]
    
opacity = 0.4


# 将预测的整数ID，映射为对应类别的颜色
pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
for idx in palette_dict.keys():
    pred_mask_bgr[np.where(pred_mask==idx)] = palette_dict[idx]
pred_mask_bgr = pred_mask_bgr.astype('uint8')

# 将语义分割预测图和原图叠加显示
pred_viz = cv2.addWeighted(img_bgr, opacity, pred_mask_bgr, 1-opacity, 0)

cv2.imwrite('output/mmseg/1-3-2.png', pred_viz)

plt.figure(figsize=(8, 8))
plt.imshow(pred_viz[:,:,::-1])
