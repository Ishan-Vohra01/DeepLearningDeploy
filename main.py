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
                          
                          ['CNN','EfficientNet Unfreezing'],
                          default_index=0)
    
if (selected == 'CNN'):
        # page title
    st.title('CNN')
    # Load the pre-trained model
model = load_model("emotion_detection_model.h5")

# Define class labels
class_labels = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to predict the class label
def predict(image_path):
    img_width, img_height = 48, 48  # Assuming the dimensions used during training
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Reshape to (1, img_width, img_height, 3)
    img_array /= 255.0  # Normalize the image
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]

# Streamlit UI
st.title('Image Classification')
st.write('Upload an image for classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Perform prediction when 'Predict' button is clicked
    if st.button('Predict'):
        # Save the uploaded file to a temporary location
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Get prediction
        prediction = predict("temp_image.jpg")
        
        # Display predicted class label
        st.success(f"Predicted class label: {prediction}")
        
