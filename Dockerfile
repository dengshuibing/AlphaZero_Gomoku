FROM cnstark/pytorch:2.0.1-py3.9.17-cuda11.8.0-devel-ubuntu20.04


RUN pip install Flask==2.0.1 Werkzeug==2.0.1 mysql-connector==2.2.9 \
    scikit-image==0.21.0 lxml==4.9.3  \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip uninstall opencv-python -y & \
    pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install paho-mqtt==1.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
WORKDIR /path/to/project/gomoku
