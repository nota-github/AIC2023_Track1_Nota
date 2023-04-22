FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update
RUN apt-get install -y tzdata
RUN apt-get install -y sudo
RUN sudo apt update
RUN sudo apt install -y software-properties-common
RUN sudo add-apt-repository -y ppa:deadsnakes/ppa
RUN sudo apt install -y python3.9
RUN sudo apt-get install -y python3-pip

RUN pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN apt-get install -y libgl1-mesa-glx ffmpeg

RUN pip3 install Cython
Run pip3 install cython-bbox

RUN pip3 install -U openmim
RUN mim install mmengine && \
    mim install "mmcv>=2.0.0" && \
    mim install "mmdet>=3.0.0"

RUN pip3 install lap \
    ultralytics \
    mmpose \
    fastreid \
    opencv-python==4.5.5.64 \
    opencv-python-headless==4.5.5.64 \
    numpy==1.23.0