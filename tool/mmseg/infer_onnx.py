import onnxruntime
import numpy as np
import cv2

onnx_path = "/home/txy/code/TXY_code/onnx/mmseg/kent-640-sim.onnx"

# 创建ONNX运行时会话
ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

img_path = "/home/txy/code/TXY_code/pic/mmseg/894-640.png"

# 读取图片并调整大小
img_bgr_crop = cv2.imread(img_path)
img_bgr_resize = cv2.resize(img_bgr_crop, (640, 640))

# 预处理
img_tensor = img_bgr_resize
mean = (123.675, 116.28, 103.53)  # BGR三通道的均值
std = (58.395, 57.12, 57.375)  # BGR三通道的标准差

# 归一化
img_tensor = (img_tensor - mean) / std
img_tensor = img_tensor.astype('float32')

# BGR转RGB
img_tensor = cv2.cvtColor(img_tensor, cv2.COLOR_BGR2RGB)

# 调整维度
img_tensor = np.transpose(img_tensor, (2, 0, 1))
# 扩充batch-size维度
input_tensor = np.expand_dims(img_tensor, axis=0)

# 预测
ort_input = {'input': input_tensor}  
ort_output = ort_session.run(['output'], ort_input)[0]
pred_mask = ort_output[0][0]

# 可视化
palette = [
    ['background', [127, 127, 127]],
    ['bubble', [0, 0, 255]],
]

palette_dict = {}
for idx, each in enumerate(palette):
    palette_dict[idx] = each[1]

opacity = 0.3  # 透明度

pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
for idx in palette_dict.keys():
    pred_mask_bgr[np.where(pred_mask == idx)] = palette_dict[idx]
pred_mask_bgr = pred_mask_bgr.astype('uint8')

# 将语义分割预测图和原图叠加显示
pred_viz = cv2.addWeighted(img_bgr_resize, opacity, pred_mask_bgr, 1 - opacity, 0)
cv2.imwrite('pic/mmseg/849_sim-onnx.png', pred_viz)
