import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import base64
from io import BytesIO

def image_to_base64(image):
    """Converte uma imagem em Base64"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Carregue o modelo salvo
model = tf.keras.models.load_model('detector_de_filtros.h5')

# Compilar o modelo, caso necessário (se não estiver compilado)
if not model.compiled:
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Funções para carregar e preprocessar a imagem
def load_and_preprocess_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_filter(img_array):
    # Certifique-se de que o modelo esteja compilado
    if not model.compiled:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    prediction = model.predict(img_array)
    return prediction[0][0]


# Controlando a navegação entre as páginas usando o session_state
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# Página de boas-vindas
if st.session_state.page == 'welcome':
    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open('foto/logo4.jpg') 
        st.markdown('<img class="logo-img" src="data:image/jpeg;base64,{}" />'.format(image_to_base64(image)), unsafe_allow_html=True)

    with col2:
        st.markdown("<h1 id='bem-vindo-ao'>Bem-vindo ao </h1>", unsafe_allow_html=True)
        st.markdown("<h1 id='detector'>Detector de Filtros</h1>", unsafe_allow_html=True)
        st.markdown("<h1 id='de-redes-sociais'>De Redes Sociais</h1>", unsafe_allow_html=True)

    if st.button('Entrar'):
        st.session_state.page = 'instructions'
        

# Página de instruções
elif st.session_state.page == 'instructions':

    # Criando duas colunas
    col1, empty_col, col2 = st.columns([1, 0.5, 1])
    
    # Conteúdo da primeira coluna
    with col1:
        st.write("""
        ### "O Detector de Filtros de Redes Sociais tem como objetivo identificar a aplicação de filtros ou o uso de inteligência artificial em fotos, garantindo a autenticidade das imagens compartilhadas e postadas nas Redes Sociais."
        """)

    # Conteúdo da segunda coluna
    with col2:
        st.write("""
        ### Instruções de como usar o Detector de filtros:
        - Selecione a foto de sua preferência.
        - Faça o upload da foto no Detector de Filtros.
        - O Detector analisará a imagem para determinar se há a aplicação de filtros.
        - O resultado será exibido logo abaixo da foto.
        """)

    if st.button('Iniciar'):
        st.session_state.page = 'detector'  

# Página do detector
elif st.session_state.page == 'detector':
    # Injetar o CSS para centralizar o título
    st.markdown("<p class='text-detector'>Detector de Filtros</p>", unsafe_allow_html=True)

    # Centralizar o file_uploader usando CSS e Streamlit separado
    st.markdown("""
        <div id="file-uploader">
            <label for="file">Escolha uma imagem...</label>
        </div>
    """, unsafe_allow_html=True)

    # Coloque o st.file_uploader fora do st.markdown
    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.markdown('<img class="uploaded-img" src="data:image/jpeg;base64,{}" />'.format(image_to_base64(img)), unsafe_allow_html=True)

        img_array = load_and_preprocess_image(img)
        prediction = predict_filter(img_array)

        print(prediction)


        if prediction > 0.5:
            st.markdown("<p class='resultado-filtro'>A imagem provavelmente não tem um filtro aplicado.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='resultado-filtro'>A imagem provavelmente tem filtro aplicado.</p>", unsafe_allow_html=True)
