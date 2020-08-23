# RM object detection
## 项目简介
本项目由哈工大HITer组开源。

本项目共提供两组模型：

- 部署在机器人平台的检测模型——TinyYOLOv3，该模型是由我们组对Darknet版本的TinyYOLOv3的PyTorch复现，添加了SPP模块。
- 部署在哨岗平台的检测模型——SlimYOLOv2，该模型是由我们组对Darknet版本的YOLOv2的PyTorch复现，并且做了轻量化处理和添加SPP模块。

其中，我们将YOLOv2中的backbone网络——Darknet19替换为由我们自主设计的轻量级网络Darknet_tiny，该网络在ImageNet数据集上进行预训练，在val上获得
top1精度63.5和top5精度85.06。为了提高网络的性能，同时不引入过多的计算量，在detection head部分添加了SPP模块。

## 数据集
本项目的数据集由我们自主采集，使用相机在不同视角、不同距离、不同姿态下采集。训练集共有2400张图片，目前尚未开源数据集。

## 模型
### Backbone 预训练模型
预训练模型请到下面的百度云链接下载：

链接：https://pan.baidu.com/s/1_iqu6YXyk91uMN98wtQrDw 

提取码：9b3i

### 检测模型
已训练好的模型请使用下面的百度云链接下载：

链接：https://pan.baidu.com/s/1o6_Kjv_PrTWCp4wZ_csbCg 

提取码：m2i3

## 训练
假定已有训练集，并且接口合适，则使用下面的命令即可训练，以TinyYOLOv3为例：

```Shell
python train.py -v tiny_yolo_v3 -hr --cuda --num_workers 8
```

其中，-hr参数用于载入在ImageNet上经过448分辨率图像finetune的高清模型；--cuda参数用于调用GPU；--num_workers参数用于设置PyTorch中的dataloader中的
线程数。

另外，本项目支持多尺度训练，例如：

```Shell
python train.py -v tiny_yolo_v3 -hr --cuda -ms --num_workers 0
```

调用-ms参数即可，由于本项目存在尚未解决的问题，调用多尺度训练时，需要将num_workers设为0，从而只调用主线程来进行训练，训练速度会变慢。

## 测试
测试代码请查看```demo.py```，其中，我们支持三种测试模型：
- 调用摄像头，对应命令行参数   --mode camera
- 使用静态图像，对应命令行参数 --mode image
- 使用离线视频，对应命令行参数 --mode video

以TinyYOLOv3为例：

```Shell
python demo.py -v tiny_yolo_v3 --cuda --mode --path_to_img [请输入您的路径] --trained_model [请输入您的保存模型的路径]
```

测试时，默认图像大小为 640x640，可视化阈值为0.2

## 其他
更多的配置参数，如训练次数，anchor box尺寸等，可打开```data/config.py```文件自行查看，