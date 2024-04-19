import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

# Create a MMagicInferencer instance and infer
img = '/home/txy/code/TXY_code/pic/mmagic/qx/00000074.png'
result_out_dir = '/home/txy/code/TXY_code/output/mmagic/74.jpg'
editor = MMagicInferencer(model_ckpt="/home/txy/code/TXY_code/pth/mmagic/NAFNet-GoPro-midc64.pth",
                          model_name="nafnet",
                          device="cuda:0",
                          model_config="/home/txy/code/TXY_code/configs/config_mmagic/图像去模糊/nafbet_1280×720_gopro.py",)
                          
results = editor.infer(img=img, result_out_dir=result_out_dir)