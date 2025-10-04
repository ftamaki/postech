import tensorflow as tf
print(tf.__version__)

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

import tensorflow as tf
from tensorflow.python.client import device_lib
import os

# Mostrar detalhes da lib CUDA usada
print("TensorFlow version:", tf.__version__)
print("CUDA visible devices:", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set'))
print("Physical GPU devices:", tf.config.list_physical_devices('GPU'))

# Mostrar todos os dispositivos visíveis (inclusive CPU)
print("\n=== Dispositivos disponíveis ===")
for d in device_lib.list_local_devices():
    print(f"{d.name} ({d.device_type})")

# Forçar log detalhado se necessário
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import tensorflow as tf
print("TF version:", tf.__version__)
print(tf.config.list_physical_devices('GPU'))
