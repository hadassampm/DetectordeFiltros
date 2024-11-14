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


# Treine o modelo
history = model.fit(
    train_generator,
    steps_per_epoch=10,  # número de passos por época (ajuste conforme necessário)
    epochs=15,  # número de épocas (ajuste conforme necessário)
    validation_data=validation_generator,
    validation_steps=5  # número de passos de validação (ajuste conforme necessário)
)


# Salve o modelo
model.save('detector_de_filtros.h5')


val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Acurácia no conjunto de validação: {val_accuracy}")
