import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
from keras.preprocessing import image
from keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.image import resize
from PIL import Image

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('DeepLearning Deployment',
                          
                          ['CNN','EfficientNet Unfreezing','VAE','DC GAN'],
                          default_index=0)
    

if (selected == 'CNN'):

    st.title('CNN')
model = load_model("CNN.h5")

class_labels = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

def predict(image_path):
    img_width, img_height = 48, 48
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]

st.title('Image Classification')
st.write('Upload an image for classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        prediction = predict("temp_image.jpg")
        
        st.success(f"Predicted class label: {prediction}")

if (selected == 'EfficientNet Unfreezing'):

model = load_model("Efficent_net.h5")

class_labels = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

def predict(image_path):
    img_width, img_height = 48, 48
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]

st.title('Image Classification')
st.write('Upload an image for classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        prediction = predict("temp_image.jpg")
        
        st.success(f"Predicted class label: {prediction}")
        

if (selected == 'VAE'):

model = load_model("vae_model.h5")

class_labels = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

def predict(image_path):
    img_width, img_height = 48, 48
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]

st.title('Image Classification')
st.write('Upload an image for classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        prediction = predict("temp_image.jpg")
        
        st.success(f"Predicted class label: {prediction}")
                


if (selected == 'DC GAN'):

model = load_model("discriminator_weights.h5")

class_labels = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

def predict(image_path):
    img_width, img_height = 48, 48
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]

st.title('Image Classification')
st.write('Upload an image for classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        prediction = predict("temp_image.jpg")
        
        st.success(f"Predicted class label: {prediction}")
        
