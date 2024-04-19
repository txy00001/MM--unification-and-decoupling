import os
from mmagic.apis.inferencers import VideoRestorationInferencer
from mmengine import mkdir_or_exist


# Create a MMagicInferencer instance and infer
video_path = "/home/txy/code/TXY_code/pic/mmagic/video/demo.mp4"
result_out_dir = "/home/txy/code/TXY_code/output/mmagic/demo_out.mp4"
model_ckpt="/home/txy/code/TXY_code/pth/mmagic/realbasicvsr.pth"
model_config="configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py"
mkdir_or_exist(os.path.dirname(result_out_dir))
editor = VideoRestorationInferencer(ckpt=model_ckpt,
                          config=model_config,
                          device='cuda:0',
                          extra_parameters={
                            'max_seq_len': 2,
                            'start_idx': 0,} 
)
#预处理
preprocessed_data = editor.preprocess(video=video_path)##处理输入视频，将其转换为模型可以接受的格式
##
##前向推理
preds = editor.forward(inputs=preprocessed_data)##将处理后的输入送入模型进行前向传播，得到恢复后的视频帧
##可视化
output_video_path = os.path.join(result_out_dir, "output/mmagic/restored_video.mp4")#可视化模型的预测结果，并将结果保存到指定目录
editor.visualize(preds=preds, result_out_dir=output_video_path)
