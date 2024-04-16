import os
import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

model1 = load_model('official-models/LettuceModel.h5')  # saved model from training
model2 = load_model('official-models/CauliflowerModel.h5')  # saved model from training
model3 = load_model('official-models/SugarcaneModel-1.h5')  # saved model from training
model4 = load_model('official-models/PepperModel.h5')  # saved model from training

Lettuce_names = ["lettuce_BacterialLeafSpot", "lettuce_BotrytisCrownRot", "lettuce_DownyMildew", "lettuce_Healthy"]

Cauliflower_names = ["cauliflower_BlackRot", "cauliflower_DownyMildew", "cauliflower_Healthy", "cauliflower_SoftRot"]

Sugarcane_names = ["sugarcane_Healthy", "sugarcane_Mosaic", "sugarcane_RedRot", "sugarcane_Rust"]

Pepper_names = ["pepper_Healthy", "pepper_CercosporaLeafSpot", "pepper_Fusarium", "pepper_Leaf_Curl"]

folder_path = "saved_images"

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Resize image to match model's input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values to range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def L_predict_disease(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model1.predict(preprocessed_img)
    disease_index = np.argmax(prediction)  # Get the index of the predicted Lettuce class
    disease_class = Lettuce_names[disease_index] # Fetch the class name using the index
    return disease_class

def C_predict_disease(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model2.predict(preprocessed_img)
    disease_index = np.argmax(prediction)  # Get the index of the predicted Cauliflower class
    disease_class = Cauliflower_names[disease_index] # Fetch the class name using the index
    return disease_class

def S_predict_disease(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model3.predict(preprocessed_img)
    disease_index = np.argmax(prediction)  # Get the index of the predicted Sugarcane class
    disease_class = Sugarcane_names[disease_index] # Fetch the class name using the index
    return disease_class

def P_predict_disease(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model4.predict(preprocessed_img)
    disease_index = np.argmax(prediction)  # Get the index of the predicted Pepper class
    disease_class = Pepper_names[disease_index] # Fetch the class name using the index
    return disease_class


# =======================================

#sidebar
with st.sidebar:
    st.subheader('About Us', divider='gray')
    
    st.info(
        """
        Authors:
        
        Christian Jerome S. Detuya
        
        Albert James E. Mangcao
        
        Axel Bert E. Ramos
        
        """
        )



tab1, tab2, tab2 = st.tabs(["Home", "Crop Health Assessment", "About Crop Health Assessment"])

with tab1:
    # selecting method for health assessment
    st.subheader("SELECT A METHOD")
    pick = st.selectbox("Select Method",('Upload','Camera'),label_visibility="hidden")

    if pick == 'Camera':
        st.subheader("Camera Input")
        plantpic = st.camera_input("take a plant picture",label_visibility="hidden")
        
        st.subheader("Select A Plant")
        select = st.selectbox("Select Plant",('Lettuce','Cauliflower','Sugarcane','Pepper'),label_visibility="hidden")
         
        submit = st.button("submit",use_container_width=True)
        
        if submit:
            if not plantpic:
                st.write("take a photo!")
                
            elif select == 'Lettuce':
                # predicting Lettuce disease
                image_path = plantpic
                predicted_class = L_predict_disease(image_path)
                pred1 = "Predicted Disease Class: " + predicted_class
                st.image(plantpic,pred1)
                
            elif select == 'Cauliflower':
                # predicting Cauliflower disease
                image_path = plantpic
                predicted_class = C_predict_disease(image_path)
                pred2 = "Predicted Disease Class: " + predicted_class
                st.image(plantpic,pred2)
                
            elif select == 'Sugarcane':
                # predicting Sugarcane disease
                image_path = plantpic
                predicted_class = S_predict_disease(image_path)
                pred3 = "Predicted Disease Class: " + predicted_class
                st.image(plantpic,pred3)
                
            elif select == 'Pepper':
                # predicting Pepper disease
                image_path = plantpic
                predicted_class = P_predict_disease(image_path)
                pred4 = "Predicted Disease Class: " + predicted_class
                st.image(plantpic,pred4)
               
            
    elif pick == 'Upload':
        st.subheader("Upload Image File")
        plantpic = st.file_uploader("upload",['jpg','png','gif','webp','tiff','psd','raw','bmp','jfif'],False,label_visibility="hidden")
        
        st.subheader("Select A Plant")
        select = st.selectbox("Select Plant",('Lettuce','Cauliflower','Sugarcane','Pepper'),label_visibility="hidden")
        
        submit1 = st.button("submit",use_container_width=True)
        if submit1:
            if not plantpic:
                st.write("take a photo!")
                
            elif select == 'Lettuce':
                # predicting Lettuce disease
                image_path = plantpic
                predicted_class = L_predict_disease(image_path)
                pred1 = "Predicted Disease Class: " + predicted_class
                st.image(plantpic,pred1)
                
            elif select == 'Cauliflower':
                # predicting Cauliflower disease
                image_path = plantpic
                predicted_class = C_predict_disease(image_path)
                pred2 = "Predicted Disease Class: " + predicted_class
                st.image(plantpic,pred2)
                
            elif select == 'Sugarcane':
                # predicting Sugarcane disease
                image_path = plantpic
                predicted_class = S_predict_disease(image_path)
                pred3 = "Predicted Disease Class: " + predicted_class
                st.image(plantpic,pred3)
                
            elif select == 'Pepper':
                # predicting Pepper disease
                image_path = plantpic
                predicted_class = P_predict_disease(image_path)
                pred4 = "Predicted Disease Class: " + predicted_class
                st.image(plantpic,pred4)
                
    

with tab2:
    """
    The "Deep Learning and Machine Learning Integration: A Comprehensive Approach to Automated Crop Health Assessment Using CNN, ANN, and SVM" project aims to revolutionize crop health assessment in agriculture through cutting-edge technology and advanced machine learning techniques. By leveraging Convolutional Neural Networks (CNN), Artificial Neural Networks (ANN), and Support Vector Machines (SVM), the project endeavors to automate the process of evaluating crop health, thereby enhancing agricultural practices and promoting sustainability.


In this research, CNN serves as the primary tool for image processing and feature extraction from crop images, enabling the identification of visual cues indicative of crop health. The extracted features are then fed into ANN, which analyzes the data and generates recommendations or actions to be taken for diseased plants. These recommendations encompass treatment strategies, irrigation adjustments, and pest control measures, aimed at improving crop yield and minimizing losses.


Additionally, SVM plays a crucial role in the classification of crops based on the features extracted by CNN and processed by ANN. By training SVM to classify crops into categories such as healthy, diseased, or under stress, the project achieves a comprehensive assessment of crop health, facilitating informed decision-making for farmers and agricultural stakeholders.


Through the integration of CNN, ANN, and SVM, the project not only advances the field of automated crop health assessment but also contributes to the broader goals of precision agriculture and sustainable farming practices. By harnessing the power of deep learning and machine learning, this research endeavors to redefine how crop quality is evaluated, ultimately leading to higher crop yields, reduced losses, and a more resilient agricultural ecosystem.
    
    """