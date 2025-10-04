import tensorflow as tf
import numpy as np

# Verificar se a GPU está disponível
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU está disponível.")
    # Configurar para usar a GPU (opcional, TensorFlow tentará usar por padrão)
    # tf.config.experimental.set_memory_growth(physical_devices[0], True) # Opcional: Permite que o TensorFlow aloque memória GPU conforme necessário
else:
    print("GPU NÃO está disponível. O código rodará na CPU.")

# Exemplo de dados de entrada
# X: matriz com 100 amostras e 3 características
# y: vetor de saída com 100 valores
X = np.random.random((100, 3))
y = np.random.random((100, 1))

# Definindo o modelo usando Input
inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(10, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compilando o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinando o modelo
model.fit(X, y, epochs=5)