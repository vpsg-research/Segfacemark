<div align="center">
<h1> [TCSVT'26] One-Identity-One-Key: Region-aware Watermark for Proactive Deepfake Detection and Robust Source Tracing </h1>
</div>


## 📢 News
* **[2026-6]** Our paper is accepted by **TCSVT 2026**! 🎉
* **[2026-5]** The code are being organized and will be released shortly. Please star this repo for updates!

## ✨ Contributions
⚠️ We propose a region-aware watermarking framework. While maintaining global embedding, the framework implements region-wise partitioning and redundant embedding specifically for key semantic facial regions, thereby enhancing its resistance to local deepfake tampering and improving overall robustness. During detection, it combines the consistency of global and regional watermarks, which not only increases the accuracy of authenticity verification but also makes the detection results interpretable.

🚀 We propose an identity-driven dynamic encryption mechanism that generates a unique key from facial identity features, enabling a “one-identity–one-key” security design and resisting forgery and reverse analysis. During decryption, identity and structural consistency are jointly checked, allowing the framework to distinguish different types of deepfake and improving the accuracy and reliability of forgery detection.

🧩The experimental results show that the proposed framework achieves excellent detection accuracy, robustness, and imperceptibility across multiple datasets, and even when trained on a single deepfake type, it still demonstrates strong generalization ability.


##  📻 Overview
Official repository for the TCSVT2026 paper “*One-Identity-One-Key: Region-aware Watermark for Proactive Deepfake Detection and Robust Source Tracing*” [[paper]](https://ieeexplore.ieee.org/document/11552736) 

<div align="center">
    <img width="1000" alt="image" src="Image\3.png">
</div>

<div align="center">
Illustration of the overall architecture.
</div>

## 🎮 Getting Started

### 1. Install Environment

```bash
python -m pip install -r requirements.txt
```
### 2.📁 Datasets Used

SegFaceMark is trained using CelebAMask-HQ and tested on CelebAMask-HQ and LFW. We do not own the datasets, and they can be downloaded from the official webpages.

* [Download CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
* [Download LFW](http://vis-www.cs.umass.edu/lfw/)

After splitting the image data, the directory should look like the following:

```text
SegFaceMark
├── CelebAMask-HQ/
│   └── train/
│       ├── train_128/
│       │   ├── 0.jpg
│       │   ├── 1.jpg
│       │   └── ...
│       ├── train_256/
│       │   ├── 0.jpg
│       │   ├── 1.jpg
│       │   └── ...
│       ├── val_128/
│       │   ├── 1000.jpg
│       │   └── ...         
│       └── val_256/
│           ├── 1000.jpg
│           └── ...    
```

### 3.📁 watermark Used

To facilitate reproduction, we provide the pre-generated watermark message used in our experiments. The released `.npy` file contains the fixed binary watermark adopted during training and evaluation. Researchers can directly load this watermark to reproduce our experimental settings without regenerating the watermark from scratch.

🌍 **Google Drive** [Download watermark](https://drive.google.com/file/d/18ErP5hXoB-8rucctMDd9I8nR7PnLenma/view?usp=sharing)

### 4.🎭 Noise / Deepfake Models

Since we don't own the source code, we recommend downloading and placing the model source code and weights by yourself. The source code can be found at the following links:

* [FSRT (CVPR 2024)](https://github.com/andrerochow/fsrt)
* [RAFSwap (CVPR 2022)](https://github.com/xc-csc101/RAFSwap)
* [SimSwap (ACM MM 2020)](https://github.com/neuralchen/SimSwap)
* [CSCS (ACM TOG 2024)](https://github.com/ICTMCG/CSCS)
* [HifiFace (IJCAI 2021)](https://github.com/johannwyh/HifiFace)
* [UniFace (ECCV 2022)](https://github.com/xc-csc101/UniFace)
* [GANimation (ECCV 2018)](https://github.com/albertpumarola/GANimation)
* [InfoSwap (CVPR 2021)](https://github.com/GGGHSL/InfoSwap-master)
* [StyleMask (FG 2023)](https://github.com/StelaBou/StyleMask)
* [StarGAN (CVPR 2018)](https://github.com/yunjey/stargan)

### 5.📦 Pre-trained Weights

This project relies on several pre-trained models for face detection, recognition, and segmentation. Please refer to their official webpages to download the required weights:

* **ArcFace**: [Pre-trained models (InsightFace)](https://github.com/deepinsight/insightface)
* **dlib**: [Trained model files for dlib](https://github.com/davisking/dlib-models)
* **SegFace**: [Segface: Face segmentation of long-tail classes (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/32661)

### 6. Train
```bash
#We employ a progressive training strategy. The model is trained in multiple stages, where the training configuration and loss weights can be gradually adjusted based on the checkpoint obtained from the previous stage. This enables progressive optimization of the model throughout the training process.
python segfacemark/main.py 
```
### 7. Test
```bash
#During testing, users may adjust the watermark embedding strength by modifying the wm_factor parameter in train.yaml.
python segfacemark/test.py
```

## Citation

If you find our code useful, please consider citing us and give us a star!

```
@ARTICLE{11552736,
  author={He, Ziyuan and Guo, Zhiqing and Wang, Liejun and Tao, Xiaoming and Liao, Xin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={One-Identity-One-Key: Region-aware Watermark for Proactive Deepfake Detection and Robust Source Tracing}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Watermarking;Signal detection;Faces;Deepfakes;Modeling;Bit error rate;Forgery;Robustness;Conferences;Computers;Deepfake Detection;Source Tracing;Region-aware Watermarking;Dynamic Encryption},
  doi={10.1109/TCSVT.2026.3700633}}
```
