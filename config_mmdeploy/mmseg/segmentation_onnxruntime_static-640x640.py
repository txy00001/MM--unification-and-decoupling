_base_ = ['/home/txy/code/TXY_code/config_mmdeploy/mmseg/segmentation_static.py', '/home/txy/code/TXY_code/config_mmdeploy/_base_/backends/onnxruntime.py']

onnx_config = dict(input_shape=[640,640])
