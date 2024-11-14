import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


# Carregue o modelo salvo
model = tf.keras.models.load_model('detector_de_filtros.h5')


def load_and_preprocess_image(img_path):
    # Carregue a imagem
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma nova dimensão
    img_array /= 255.0  # Normaliza a imagem
    return img_array


# Caminho para a imagem que você quer testar
#test_image_path = r'C:\Users\hadas\OneDrive\Área de Trabalho\tcc python\imagem\teste4.jpg'
#Com filtro
test_image_path = r'C:\Users\hadas\OneDrive\Área de Trabalho\tcc python\imagem\foto.jpg'


# Preprocessa a imagem
img = load_and_preprocess_image(test_image_path)


# Visualize a imagem para verificar se está correta
plt.imshow(image.array_to_img(img[0]))
plt.title("Imagem de Teste")
plt.show()


# Faz a previsão
prediction = model.predict(img)


# Verifique o valor da predição
print(f"Predição bruta do modelo: {prediction}")


# Interpreta a previsão com diferentes limites
for threshold in [0.5]:
    if prediction[0] > threshold:
        print(f"Com limite {threshold}: A imagem não tem filtro.")
    else:
        print(f"Com limite {threshold}: A imagem tem filtro.")