# SP14

本仓库用于记录本人毕业设计 SP14 中与 AI infra & deployment 相关的工作，重点是模型从导出到上板运行的过程。

## 项目简介
本项目围绕 **自动驾驶场景下的车载目标检测（On-device Object Detection for Autonomous Driving）** 展开，重点关注目标检测模型在边缘设备上的部署可行性，而不是单纯追求训练精度。项目以 **BDD100K** 作为实验背景，比较了不同目标检测模型在精度、模型体积、部署复杂度和实际运行性能上的差异，并尝试将模型部署到 **Raspberry Pi 5 + Hailo NPU** 平台上，实现从模型导出、量化编译到端侧推理的完整流程。

## 项目范围
- 任务：On-device Object Detection for Autonomous Driving
- 数据集：BDD100K（本仓库不提供原始数据）
- 主要模型：YOLOv5, YOLOv8, PicoDet, DETR-family
- 主要硬件：Raspberry Pi 5（CPU baseline）+ Hailo NPU
- 主要内容：ONNX 导出、calibration set 构建、HEF 编译、树莓派推理、结果截图与性能记录

## 说明
- 训练代码、完整数据、所有中间产物不会系统整理到这里。

## 目录
- ppt/            中期汇报 PPT
- code/           主要脚本
- models/         导出的 ONNX / HEF 类别文件
- data/           演示图像与 calibration 样本
- results/        截图、benchmark、日志
- notes/          模型结论、部署总结、问题记录

## 备注
- Hailo DFC / HailoRT 相关环境按本地实际安装，不通过 requirements.txt 统一管理。
