import tensorflow as tf
# import torch
import multiprocessing


# print(torch.cuda.is_available())


n_cpu = multiprocessing.cpu_count()
# print(n_cpu)

gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)
# tf.config.list_physical_devices('GPU')



# print(tf.__version__)