# RobuRCDet

This is the official implementation of ICLR2025 paper: [**RobuRCDet: Enhancing Robustness of Radar-Camera Fusion in Bird's Eye View for 3D Object Detection**](https://arxiv.org/abs/2502.13071).


## Abstract
In this paper, we first conduct a systematic analysis of robustness in radar-camera detection on five kinds of noises and propose RobuRCDet, a robust object detection
model in bird’s eye view (BEV).
To mitigate inaccuracies in radar points, including position, Radar Cross-Section (RCS), and velocity, we design a 3D Gaussian Expansion (3DGE) module to achieve radar denoising.
We further introduce a weather-adaptive fusion module, which adaptively fuses radar and camera features based on camera signal confidence. 
RobuRCDet achieves competitive results in both regular and noisy conditions.


## Getting Started

### Installation



## Acknowledgement
This project is based on excellent open source projects:
- [CRN](https://github.com/youngskkim/CRN)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
