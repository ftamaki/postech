import tensorflow as tf
import time

# Lista dispositivos disponíveis
print("Dispositivos disponíveis:", tf.config.list_physical_devices())
print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))

# Teste na GPU (se disponível)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    with tf.device('/GPU:0'):
        print("Executando na GPU...")
        a = tf.random.normal([10000, 10000])
        b = tf.random.normal([10000, 10000])
        start_time = time.time()
        result = tf.matmul(a, b)
        end_time = time.time()
        print(f"Multiplicação de matrizes na GPU concluída em {end_time - start_time:.2f} segundos")
else:
    print("Nenhuma GPU detectada, testando na CPU...")

# Teste na CPU
with tf.device('/CPU:0'):
    print("Executando na CPU...")
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    start_time = time.time()
    result = tf.matmul(a, b)
    end_time = time.time()
    print(f"Multiplicação de matrizes na CPU concluída em {end_time - start_time:.2f} segundos")