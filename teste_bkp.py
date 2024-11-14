import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Caminho absoluto para os diretórios de treino e validação
#checkpoint = ModelCheckpoint(r'C:\Users\hadas\OneDrive\Área de Trabalho\tcc python\best_model.h5', monitor='val_loss', save_best_only=True)
checkpoint = ModelCheckpoint(r'detector_de_filtros.h5', monitor='val_loss', save_best_only=True)


train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Data Augmentation para evitar overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Para validação, apenas redimensionamos
validation_datagen = ImageDataGenerator(rescale=1./255)

# Geradores de imagem
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Definindo o modelo com mais camadas convolucionais e regularização
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),  # Regularização para evitar overfitting
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Outra camada de Dropout
    layers.Dense(1, activation='sigmoid')
])

# Compile o modelo
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Adicionar callbacks para interromper o treinamento cedo se o modelo não melhorar
early_stop = EarlyStopping(monitor='val_loss', patience=3)
#checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
checkpoint = ModelCheckpoint('detector_de_filtros.h5', monitor='val_loss', save_best_only=True)

# Treinamento do modelo
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,  # Aumentando o número de épocas
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stop, checkpoint]  # Callbacks para monitorar o treinamento
)

# Avaliar o modelo no conjunto de validação
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Acurácia no conjunto de validação: {val_accuracy}")

# Carregar o melhor modelo salvo
#best_model = tf.keras.models.load_model('best_model.h5')
best_model = tf.keras.models.load_model('detector_de_filtros.h5')



