import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Carrega o modelo salvo
model = tf.keras.models.load_model('best_model.h5')

def load_and_preprocess_image(img_path):
    """
    Carrega e processa a imagem: redimensiona, converte em array e normaliza.
    """
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma nova dimensão
    img_array /= 255.0  # Normaliza a imagem
    return img_array

# Caminho da imagem para teste
test_image_path = r'C:\Users\hadas\OneDrive\Área de Trabalho\tcc python\imagem\teste.jpg'

# Carrega e preprocessa a imagem
img = load_and_preprocess_image(test_image_path)

# Exibe a imagem para garantir que foi carregada corretamente
plt.imshow(image.array_to_img(img[0]))
plt.title("Imagem de Teste")
plt.show()

# Faz a previsão usando o modelo
prediction = model.predict(img)

# Exibe a predição bruta
print(f"Predição bruta do modelo: {prediction[0][0]}")

# Define e avalia a predição com diferentes limites
threshold = 1  # Pode ajustar esse valor para otimizar os resultados
if prediction[0] > threshold:
    print(f"Com limite {threshold}: A imagem tem filtro.")
else:
    print(f"Com limite {threshold}: A imagem não tem filtro.")
