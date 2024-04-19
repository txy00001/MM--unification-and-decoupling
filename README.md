# 项目名称

MM系列合并项目：致力于MM系列算法的统一管理和自定义模块的解耦


## 准备环境
### 安装mm系列库
首先，你需要安装必要的库和依赖。打开你的终端或命令提示符，执行以下命令：
requirements.txt里也有对应mm,mmcv,mmengine的安装版本链接

```
pip install -U openmim
mim install mmengine
mim install "mmcv"
pip install -U openmim
mim install mmengine
pip install "mmxxx"

```

### 安装 requirements.txt
在确保你的Python环境设置完毕后，通过以下命令安装项目所需的其他依赖：

```
pip install -r requirements.txt

```
## 运行
### 运行脚本

要运行主脚本，请确保你已经处于项目的根目录下，然后执行：

```
都按照不同的任务进行了解耦，在tool和train里有对应的执行文件

```
### model

现已合并castpose_53head（通用场景），castpose（特定场景）,seg-knet(小目标分割)，deblur（自定义创新算法），前三个已完成全流程，后一个还在改进loss，会持续更新。

### 说明

```
各项权重及预训练链接：
链接：https://pan.baidu.com/s/1FzsION_dLyRldQUoEEimpg?pwd=atar 
提取码：atar  将对应文件放入对应目录下
