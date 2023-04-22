# [CVPRW2023] Addressing the Occlusion Problem in Multi-Camera People Tracking with Human Pose Estimation
The official resitory for 7th NVIDIA AI City Challenge (Track1: Multi-Camera People Tracking) from team Netspresso ([Nota Inc.](https://www.nota.ai/))

### Environment
- option 1: Install dependencies in your environment

``` bash 
sh setup.sh
pip install -r requirements.txt
```

- option 2: Using our docker image
``` 
docker build -t aic2023_track1/nota:latest -f ./Dockerfile .
docker run -it --gpus all -v /path/to/AIC2023_Track1_Nota:/workspace/AIC2023_Track1_Nota aic2023_track1/nota:latest /bin/bash
```

### Data & Model Preparation
1. Download the dataset and extract frames.  

2. Download the pre-trained models.  

Make sure the dataset structure is like:
```
├── AIC2023_Track1_Nota
    └── datasets
        ├── S001
        |   ├── c001
        |   |   ├── frame1.jpg
        |   |   └── ...
        |   ├── ...
        |   └── map.png
        ├── ...
        └── S022
```
Make sure the model structure is like:
```
├── AIC2023_Track1_Nota
    └── pretrained
        ├── market_mgn_R50-ibn.pth
        ├── duke_sbs_R101-ibn.pth
        ├── msmt_agw_S50.pth
        ├── market_aic_bot_R50.pth
        ├── yolov8x6.pth
        ├── yolov8x6_aic.pth
        └── yolov8x_aic.pth
```

### Reproduce MCPT Results
