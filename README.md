# [CVPRW2023] Addressing the Occlusion Problem in Multi-Camera People Tracking with Human Pose Estimation
The official resitory for 7th NVIDIA AI City Challenge (Track1: Multi-Camera People Tracking) from team Netspresso ([Nota Inc.](https://www.nota.ai/))
![Alt Text](demos/demo.gif)

## Environment
- option 1: Install dependencies in your environment

```bash 
bash ./setup.sh
```

- option 2: Use our docker image
```bash
docker build -t aic2023/track1_nota:latest -f ./Dockerfile .
docker run -it --gpus all -v /path/to/AIC2023_Track1_Nota:/workspace/AIC2023_Track1_Nota aic2023/track1_nota:latest /bin/bash
```

## Data & Model Preparation
1. Download the dataset and extract frames  
```bash
# extract frames
python3 tools/extract_frames.py --path /path/to/AIC23_Track1_MTMC_Tracking/
```

2. Download the pre-trained models ([Google Drive](https://drive.google.com/drive/folders/1_VichQvhbmfuD4h8x4-e7Rwc560TzWqH?usp=share_link))  

Make sure the data structure is like:
```
├── AIC2023_Track1_Nota
    └── datasets
    |   ├── S001
    |   |   ├── c001
    |   |   |   ├── frame1.jpg
    |   |   |   └── ...
    |   |   ├── ...
    |   |   └── map.png
    |   ├── ...
    |   └── S022
    |
    └── pretrained
        ├── market_mgn_R50-ibn.pth
        ├── duke_sbs_R101-ibn.pth
        ├── msmt_agw_S50.pth
        ├── market_aic_bot_R50.pth
        ├── yolov8x6.pth
        ├── yolov8x6_aic.pth
        └── yolov8x_aic.pth
```

## Reproduce MCPT Results
Run `bash ./run_mcpt.sh`  

The result files will be saved as follows:

```
├── AIC2023_Track1_Nota
    └── results
        ├── S001.txt
        ├── ...
        └── track1_submission.txt
```

## Citation
```
@InProceedings{Kim_2023_CVPR,
    author    = {Jeongho Kim, Wooksu Shin, Hancheol Park and Jongwon Baek},
    title     = {Addressing the Occlusion Problem in Multi-Camera People Tracking with Human Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
}
```
