FROM tensorflow/tensorflow:2.6.1-gpu

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y git

WORKDIR /project

RUN git clone -b mymultiprocess https://github.com/dengshuibing/AlphaZero_Gomoku.git