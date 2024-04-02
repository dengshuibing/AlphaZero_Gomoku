FROM tensorflow/tensorflow:2.6.1-gpu

RUN pip install flask \
                paho-mqtt==1.6.1 \
                mysql-connector==2.2.9 \
                opencv-python-headless==4.9.0.80 \
                -i https://pypi.tuna.tsinghua.edu.cn/simple
