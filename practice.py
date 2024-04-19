import os
import streamlit as st
import numpy as np

from openai import OpenAI
from PIL import Image
from keras.models import load_model

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Load CNN Models
model1 = load_model('official-models/LettuceModel.h5')  # saved model from training
model2 = load_model('official-models/CauliflowerModel.h5')  # saved model from training
model3 = load_model('official-models/SugarcaneModel-1.h5')  # saved model from training
model4 = load_model('official-models/PepperModel.h5')  # saved model from training

# Define plant class names
Lettuce_names = ["lettuce_BacterialLeafSpot", "lettuce_BotrytisCrownRot", "lettuce_DownyMildew", "lettuce_Healthy"]
Cauliflower_names = ["cauliflower_BlackRot", "cauliflower_DownyMildew", "cauliflower_Healthy", "cauliflower_SoftRot"]
Sugarcane_names = ["sugarcane_Healthy", "sugarcane_Mosaic", "sugarcane_RedRot", "sugarcane_Rust"]
Pepper_names = ["pepper_Healthy", "pepper_CercosporaLeafSpot", "pepper_Fusarium", "pepper_Leaf_Curl"]

# Function to preprocess input image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Resize image to match model's input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values to range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict disease using CNN model
def predict_disease(model, image_path, names):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    disease_index = np.argmax(prediction)  # Get the index of the predicted class
    disease_class = names[disease_index]  # Fetch the class name using the index
    return disease_class

# Function to display recommendations and predicted class
def display_recommendations(predicted_class):
    st.subheader("Recommendations:")
    for recommendation in recommendations.get(predicted_class, []):
        st.write(recommendation)

# Function to generate a Feedback form
def feedback():
    with st.form(key="my_form"):
        st.subheader("Feedback*",divider="gray")
        c1, c2 = st.columns(2)
        feed = c1.text_area("message",placeholder="Write a message...",label_visibility="collapsed",)
        b1 = c2.form_submit_button("Send",use_container_width=True)
        b2 = c2.form_submit_button("Cancel",use_container_width=True)

# Main content
tab1, tab2, tab3 = st.tabs(["Home", "Crop Health Assessment", "About Crop Health Assessment"])

# Insert ChatGPT-like integration code here

# Remaining code for the Crop Health Assessment app can remain unchanged

