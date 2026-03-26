# One-Identity-One-Key: Region-aware Watermark for Proactive Deepfake Detection and Robust Source Tracing

## 📢 Updates

- [x] Open project page.
- [ ] Update project page information.
- [ ] Update arXiv version.
- [ ] Release inference code.
- [ ] Release trained weights.

## 📁 Datasets Used

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

## 🎭 Noise / Deepfake Models

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

## 📬 Contact

If you have any questions, please contact:
📧 107552304059@stu.xju.edu.cn
