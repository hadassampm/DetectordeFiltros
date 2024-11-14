import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# Use caminhos absolutos para garantir que o Python encontre os diretórios
base_dir = r'C:\Users\hadas\OneDrive\Área de Trabalho\tcc python'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Defina os geradores de dados
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


# Carregue as imagens
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
class_mode='binary')


validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')


# Defina o modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


# Compile o modelo
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
validation_generator.reset()

# Definindo a função de treinamento com tf.function para otimizar a execução
@tf.function(reduce_retracing=True)
def predict_filter(img_array):
    prediction = model.predict(img_array)
    return prediction[0][0]


# Treine o modelo
history = model.fit(
    train_generator,
    #steps_per_epoch=10,  # número de passos por época (ajuste conforme necessário)
    epochs=15,  # número de épocas (ajuste conforme necessário)
    validation_data=validation_generator,
    steps_per_epoch = train_generator.samples // train_generator.batch_size,
    validation_steps = validation_generator.samples // validation_generator.batch_size

    #validation_steps=5  # número de passos de validação (ajuste conforme necessário)
)

# Salve o modelo
model.save('detector_de_filtros.h5')

# Avaliação no conjunto de validação.
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Acurácia no conjunto de validação: {val_accuracy}")

#Verificando a acurácia e plotando gráficos
import matplotlib.pyplot as plt

# Acurácia de treino e validação
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
print('------ Métricas ------')
print('Acurácia de treino: ',train_acc)
print('Acurácia de teste: ',val_acc)
print('----------------------')
# Perda (loss) de treino e validação
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Número de épocas
epochs = range(1, len(train_acc) + 1)

# Plotando a acurácia de treino e validação
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'bo-', label='Acurácia de Treino')
plt.plot(epochs, val_acc, 'ro-', label='Acurácia de Validação')
plt.title('Acurácia de Treino e Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

# Plotando a perda (loss) de treino e validação
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'bo-', label='Perda de Treino')
plt.plot(epochs, val_loss, 'ro-', label='Perda de Validação')
plt.title('Perda de Treino e Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Obtenha os rótulos verdadeiros do conjunto de validação
val_labels = validation_generator.classes

# Obtenha as previsões do modelo para o conjunto de validação
predictions = model.predict(validation_generator)
predictions = np.round(predictions).astype(int).flatten()  # Arredonda para 0 ou 1

# Calcule a matriz de confusão
conf_matrix = confusion_matrix(val_labels, predictions)

# Plote a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=validation_generator.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.show()
