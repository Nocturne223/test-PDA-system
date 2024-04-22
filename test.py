import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sklearn
 
from PIL import Image
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load CNN Models
model1 = load_model('official-models/LettuceModel.h5')  # saved model from training
model2 = load_model('official-models/CauliflowerModel.h5')  # saved model from training
model3 = load_model('official-models/SugarcaneModel-1.h5')  # saved model from training
model4 = load_model('official-models/PepperModel.h5')  # saved model from training

# # Load GPT-2 model and tokenizer
# model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
# tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")

# Define plant class names
Lettuce_names = ["lettuce_BacterialLeafSpot", "lettuce_BotrytisCrownRot", "lettuce_DownyMildew", "lettuce_Healthy"]
Cauliflower_names = ["cauliflower_BlackRot", "cauliflower_DownyMildew", "cauliflower_Healthy", "cauliflower_SoftRot"]
Sugarcane_names = ["sugarcane_Healthy", "sugarcane_Mosaic", "sugarcane_RedRot", "sugarcane_Rust"]
Pepper_names = ["pepper_Healthy", "pepper_CercosporaLeafSpot", "pepper_Fusarium", "pepper_Leaf_Curl"]

folder_path = "saved_images"

# Define recommendations for each class
recommendations = {
    "lettuce_BacterialLeafSpot": [
        "Use disease-resistant lettuce varieties whenever possible.",
        "Practice crop rotation with non-host crops to reduce the buildup of bacterial pathogens in the soil.",
        "Avoid overhead irrigation to minimize leaf wetness, as bacterial leaf spot thrives in moist conditions.",
        "Apply copper-based fungicides or bactericides according to label instructions, especially during periods of high humidity or when symptoms first appear."
    ],
    "lettuce_BotrytisCrownRot": [
        "Practice proper spacing between plants to improve air circulation and reduce humidity levels around the lettuce crowns.",
        "Avoid overhead irrigation and water lettuce at the base to prevent water accumulation in the crown.",
        "Remove and destroy infected plants and debris to prevent the spread of the fungus.",
        "Apply fungicides containing active ingredients such as iprodione or boscalid to protect healthy plants from infection."
    ],
    "lettuce_DownyMildew": [
        "Plant lettuce varieties with genetic resistance to downy mildew if available.",
        "Ensure proper drainage to prevent waterlogging, as downy mildew thrives in wet conditions.",
        "Apply fungicides containing active ingredients such as metalaxyl, mandipropamid, or chlorothalonil according to label instructions, especially during periods of high humidity or when symptoms first appear."
    ],
    "lettuce_Healthy": [
        "Maintain consistent soil moisture levels to ensure even growth and prevent stress.",
        "Provide adequate spacing between plants to promote air circulation and reduce the risk of foliar diseases.",
        "Monitor for signs of pests and diseases regularly, and take prompt action if detected.",
        "Implement good sanitation practices, including removing debris and weeds, to reduce the risk of disease spread."
    ],
    "cauliflower_BlackRot": [
        "Practice crop rotation with non-cruciferous crops to reduce the buildup of black rot pathogens in the soil.",
        "Remove and destroy infected plant debris to prevent the spread of the disease.",
        "Apply fungicides containing active ingredients such as copper or chlorothalonil according to label instructions, especially during periods of high humidity or when symptoms first appear."
    ],
    "cauliflower_DownyMildew": [
        "Plant cauliflower varieties with genetic resistance to downy mildew if available.",
        "Ensure proper spacing between plants to improve air circulation and reduce humidity levels.",
        "Apply fungicides containing active ingredients such as mandipropamid, fosetyl-aluminum, or copper hydroxide according to label instructions, especially during periods of high humidity or when symptoms first appear."
    ],
    "cauliflower_SoftRot": [
        "Practice proper sanitation and hygiene to prevent contamination of cauliflower heads during harvesting and handling.",
        "Harvest cauliflower heads at the correct maturity stage and handle them carefully to avoid bruising or injury.",
        "Store harvested cauliflower heads in cool, dry conditions to minimize the risk of soft rot development.",
        "Apply fungicides containing active ingredients such as iprodione or thiophanate-methyl to protect harvested cauliflower heads from soft rot during storage."
    ],
    "cauliflower_Healthy": [
        "Provide consistent moisture and fertility to support healthy growth and development.",
        "Monitor for signs of pests and diseases regularly, and take prompt action if detected.",
        "Implement good cultural practices, including proper spacing and soil management, to promote plant health and vigor."
    ],
    "sugarcane_Healthy": [
        "Implement proper irrigation and drainage practices to ensure adequate moisture without waterlogging.",
        "Monitor for signs of pests and diseases regularly, and take prompt action if detected.",
        "Implement good cultural practices, including weed control and soil management, to promote plant health and vigor."
    ],
    "sugarcane_Mosaic": [
        "Plant mosaic-resistant sugarcane varieties whenever possible.",
        "Use certified disease-free seed cane to minimize the risk of mosaic virus transmission.",
        "Implement strict sanitation practices to prevent the spread of mosaic virus within the sugarcane plantation.",
        "Control aphid populations, which can transmit mosaic virus, through insecticide applications or cultural practices."
    ],
    "sugarcane_RedRot": [
        "Use disease-free seed cane from reputable sources to prevent the introduction of red rot pathogens.",
        "Practice proper sanitation and hygiene to prevent the spread of red rot within the sugarcane plantation.",
        "Apply fungicides containing active ingredients such as mancozeb or thiophanate-methyl to protect healthy plants from infection."
    ],
    "sugarcane_Rust": [
        "Plant rust-resistant sugarcane varieties whenever possible.",
        "Ensure proper spacing between sugarcane rows to promote air circulation and reduce humidity levels, which can favor rust development.",
        "Apply fungicides containing active ingredients such as propiconazole or tebuconazole according to label instructions, especially during periods of high humidity or when rust symptoms first appear."
    ],
    "pepper_Healthy": [
        "Provide consistent moisture and fertility to support healthy growth and development.",
        "Monitor for signs of pests and diseases regularly, and take prompt action if detected.",
        "Implement good cultural practices, including proper spacing and soil management, to promote plant health and vigor.",
        "Use mulch to conserve soil moisture, suppress weeds, and maintain even soil temperatures around pepper plants."
    ],
    "pepper_CercosporaLeafSpot": [
        "Use disease-resistant pepper varieties whenever possible.",
        "Practice crop rotation with non-host crops to reduce pathogen buildup in the soil.",
        "Apply organic mulch around plants to prevent soil splash, which can spread the fungus.",
        "Avoid overhead irrigation to minimize leaf wetness and create drier conditions unfavorable for fungal growth.",
        "Apply copper-based fungicides according to label instructions, especially during periods of high humidity or when symptoms first appear."
    ],
    "pepper_Fusarium": [
        "Plant Fusarium-resistant pepper varieties if available.",
        "Practice crop rotation with non-host crops to reduce soilborne pathogen populations.",
        "Maintain optimal soil drainage to minimize waterlogging, as Fusarium pathogens thrive in wet conditions.",
        "Use soil solarization to reduce soilborne pathogens before planting.",
        "Apply biofungicides containing beneficial microorganisms to the soil to suppress Fusarium growth."
    ],
    "pepper_Leaf_Curl": [
        "Remove and destroy infected plants to prevent the spread of the disease to healthy plants.",
        "Avoid planting peppers in areas prone to high humidity or where water stagnates, as this can exacerbate leaf curl symptoms.",
        "Maintain proper spacing between plants to improve air circulation and reduce humidity levels around plants.",
        "Apply neem oil or horticultural oils to the foliage according to label instructions, as they may help suppress viral vectors like aphids."
    ]
    # Add recommendations for other classes similarly
}

# Function to preprocess input image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Resize image to match model's input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values to range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to generate plot for SVM predictions
def generate_svm_plot(prediction_probabilities, class_names):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(10, 6))
    plt.plot(prediction_probabilities[0], label=class_names, linestyle='-', marker='o', markersize=5)
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.grid(True, which='both', linestyle='--', linewidth=0.1)
    plt.legend(fontsize=12)
    
    # Add explanation
    explanation = """
    This plot shows the probability distribution across different classes.
    Each class represents a potential disease or health status of the crop.
    The x-axis represents the classes, while the y-axis represents the probability of each class.
    The class with the highest probability is the predicted disease or health status.
    """
    st.write(explanation)
    
    # Display the plot
    st.pyplot()
    
# Function to calculate evaluation metrics
def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    return accuracy, precision, recall, f1

# Function to display evaluation metrics
def display_evaluation_metrics(true_labels, predicted_labels):
    accuracy, precision, recall, f1 = calculate_metrics(true_labels, predicted_labels)
    st.subheader("Evaluation Metrics")
    st.write("Accuracy:", accuracy)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1 Score:", f1)

# Function to predict disease using CNN model
def predict_disease(model, image_path, names):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    disease_index = np.argmax(prediction)  # Get the index of the predicted class
    disease_class = names[disease_index]  # Fetch the class name using the index
    return prediction, disease_class  # Return prediction along with disease class

# Function to display recommendations and predicted class
def get_recommendations(predicted_class):
    return recommendations.get(predicted_class, [])


# Function to generate a Feedback form
def feedback():
    with st.form(key="my_form"):
        st.subheader("Feedback*",divider="gray")
        c1, c2 = st.columns(2)
        feed = c1.text_area("message",placeholder="Write a message...",label_visibility="collapsed",)
        b1 = c2.form_submit_button("Send",use_container_width=True)
        b2 = c2.form_submit_button("Cancel",use_container_width=True)

# Sidebar content
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
    
    # Add interactive elements
    st.sidebar.subheader("Analysis Settings")
    threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=None, help="Adjust the confidence threshold for classification")
    # crop_type = st.sidebar.selectbox("Select Crop Type", ["Cauliflower", "Pepper", "Sugarcane", "Lettuce"], help="Choose the type of crop for analysis")

# Main content
tab1, tab2, tab3 = st.tabs(["Home", "Crop Health Assessment", "About Crop Health Assessment"])

with tab1:
    st.title("Welcome to Crop Health Assessment App",False)
    
    col1, col2 = st.columns(2)
    col1.image("screenshots/PPrediction1.jpeg")
    col2.image("screenshots/PPrediction2.jpeg")
    
    st.write("""
        Welcome to the Crop Health Assessment App! This application aims to assist you in analyzing the health of your crops using advanced machine learning techniques.

        ## How to Use Camera Method
        - Proceed to Crop Health Assessment Tab.
        - Select the "Camera" option to capture an image of your crop.
        - Click "Allow" in the Popup message to use your device camera.
        - Click "Take Photo" to capture an image of your crop.
        - Choose the type of crop from the dropdown menu.
        - Click on the "Submit" button to analyze the uploaded image.
        
        ## How to Use Upload Method
        - Proceed to Crop Health Assessment Tab.
        - Select the "Upload" option to upload an image of your crop.
        - Choose the image file from your device.
        - Choose the type of crop from the dropdown menu.
        - Click on the "Submit" button to analyze the uploaded image.

        ## Method
        You can either upload an image from your device or use your device's camera to capture an image directly.

        ## About Us
        This app is developed by Christian Jerome S. Detuya, Albert James E. Mangcao, and Axel Bert E. Ramos as part of the Crop Health Assessment project.
    """)

with tab2:
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
            else:
                if select == 'Lettuce':
                    # Predict Lettuce disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model1, image_path, Lettuce_names)
                    pred1 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred1, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Lettuce " + predicted_class + ":"
                    # generated_recommendations = generate_recommendations(input_text)
                    # st.write(generated_recommendations)

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    generate_svm_plot(prediction, predicted_class)
                    
                    st.write(Lettuce_names)
                    
                    true_labels = ["lettuce_BacterialLeafSpot", "lettuce_BotrytisCrownRot", "lettuce_DownyMildew", "lettuce_Healthy"]  # True label for Lettuce
                    predicted_labels = [predicted_class] * len(true_labels)  # Predicted label for Lettuce               
                    
                    # Add evaluation metrics display after generating SVM plot
                    display_evaluation_metrics(true_labels, predicted_labels)
                    
                    st.write(true_labels)
                    st.write(predicted_labels)
                    

                elif select == 'Cauliflower':
                    # Predict Cauliflower disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model2, image_path, Cauliflower_names)
                    pred2 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred2, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Cauliflower " + predicted_class + ":"
                    # generated_recommendations = generate_recommendations(input_text)
                    # st.write(generated_recommendations)

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    generate_svm_plot(prediction, predicted_class)
                    
                    st.write(Cauliflower_names)
                    
                    true_labels = ["cauliflower_BlackRot", "cauliflower_DownyMildew", "cauliflower_Healthy", "cauliflower_SoftRot"]  # True label for Cauliflower
                    predicted_labels = [predicted_class] * len(true_labels)  # Predicted label for Cauliflower
                    
                    # Add evaluation metrics display after generating SVM plot
                    display_evaluation_metrics(true_labels, predicted_labels)
                    
                    st.write(true_labels)
                    st.write(predicted_labels)

                elif select == 'Sugarcane':
                    # Predict Sugarcane disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model3, image_path, Sugarcane_names)
                    pred3 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred3, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Sugarcane " + predicted_class + ":"
                    # generated_recommendations = generate_recommendations(input_text)
                    # st.write(generated_recommendations)

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    generate_svm_plot(prediction, predicted_class)
                    
                    st.write(Sugarcane_names)
                    
                    true_labels = ["sugarcane_Healthy", "sugarcane_Mosaic", "sugarcane_RedRot", "sugarcane_Rust"]  # True label for Sugarcane
                    predicted_labels = [predicted_class] * len(true_labels)  # Predicted label for Sugarcane
                    
                    # Add evaluation metrics display after generating SVM plot
                    display_evaluation_metrics(true_labels, predicted_labels)
                    
                    st.write(true_labels)
                    st.write(predicted_labels)

                elif select == 'Pepper':
                    # Predict Pepper disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model4, image_path, Pepper_names)
                    pred4 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred4, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Pepper " + predicted_class + ":"
                    # generated_recommendations = generate_recommendations(input_text)
                    # st.write(generated_recommendations)

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    generate_svm_plot(prediction, predicted_class)
                    
                    st.write(Pepper_names)
                    
                    true_labels = ["pepper_Healthy", "pepper_CercosporaLeafSpot", "pepper_Fusarium", "pepper_Leaf_Curl"]  # True label for Pepper
                    predicted_labels = [predicted_class] * len(true_labels)  # Predicted label for Pepper
                    
                    # Add evaluation metrics display after generating SVM plot
                    display_evaluation_metrics(true_labels, predicted_labels)    
                    true_labels = ["pepper_Healthy", "pepper_CercosporaLeafSpot", "pepper_Fusarium", "pepper_Leaf_Curl"]  # True label for Pepper
                    predicted_labels = [predicted_class] * len(true_labels)  # Predicted label for Pepper
                    
                    # Generate SVM plot
                    generate_svm_plot(prediction, predicted_class)
                    
                    feedback()

    elif pick == 'Upload':
        st.subheader("Upload Image File")
        plantpic = st.file_uploader("upload",['jpg','png','gif','webp','tiff','psd','raw','bmp','jfif'],False,label_visibility="hidden")
        
        st.subheader("Select A Plant")
        select = st.selectbox("Select Plant",('Lettuce','Cauliflower','Sugarcane','Pepper'),label_visibility="hidden")
        
        submit1 = st.button("submit",use_container_width=True)
        if submit1:
            if not plantpic:
                st.write("take a photo!")
            else:
                if select == 'Lettuce':
                    # Predict Lettuce disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model1, image_path, Lettuce_names)
                    pred1 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred1, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Lettuce " + predicted_class + ":"
                    # generated_recommendations = generate_recommendations(input_text)
                   # st.write(generated_recommendations)

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    generate_svm_plot(prediction, predicted_class)
                    
                    st.write(Lettuce_names[0], Lettuce_names[1], Lettuce_names[2], Lettuce_names[3])
                    
                    true_labels = ["lettuce_BacterialLeafSpot", "lettuce_BotrytisCrownRot", "lettuce_DownyMildew", "lettuce_Healthy"]  # True label for Lettuce
                    predicted_labels = [predicted_class] * len(true_labels)  # Predicted label for Lettuce               
                    
                    # Add evaluation metrics display after generating SVM plot
                    display_evaluation_metrics(true_labels, predicted_labels)

                elif select == 'Cauliflower':
                    # Predict Cauliflower disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model2, image_path, Cauliflower_names)
                    pred2 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred2, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Cauliflower " + predicted_class + ":"
                    # generated_recommendations = generate_recommendations(input_text)
                   # st.write(generated_recommendations)

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    generate_svm_plot(prediction, predicted_class)
                    
                    st.write("Class 0.0:", Cauliflower_names[0], " Class 1.0:", Cauliflower_names[1], align='center')
                    st.write("Class 2.0:", Cauliflower_names[2], " Class 3.0:", Cauliflower_names[3], align='center')
                    
                    true_labels = ["cauliflower_BlackRot", "cauliflower_DownyMildew", "cauliflower_Healthy", "cauliflower_SoftRot"]  # True label for Cauliflower
                    predicted_labels = [predicted_class] * len(true_labels)  # Predicted label for Cauliflower
                    
                    # Add evaluation metrics display after generating SVM plot
                    display_evaluation_metrics(true_labels, predicted_labels)
                    
                    st.write(true_labels)
                    st.write(predicted_labels)

                elif select == 'Sugarcane':
                    # Predict Sugarcane disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model3, image_path, Sugarcane_names)
                    pred3 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred3, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Sugarcane " + predicted_class + ":"
                    # generated_recommendations = generate_recommendations(input_text)
                    # st.write(generated_recommendations)

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    generate_svm_plot(prediction, predicted_class)
                    
                    st.write(Sugarcane_names)
                    
                    true_labels = ["sugarcane_Healthy", "sugarcane_Mosaic", "sugarcane_RedRot", "sugarcane_Rust"]  # True label for Sugarcane
                    predicted_labels = [predicted_class] * len(true_labels)  # Predicted label for Sugarcane
                    
                    # Add evaluation metrics display after generating SVM plot
                    display_evaluation_metrics(true_labels, predicted_labels)
                    
                    st.write(true_labels)
                    st.write(predicted_labels)

                elif select == 'Pepper':
                    # Predict Pepper disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model4, image_path, Pepper_names)
                    pred4 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred4, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Pepper " + predicted_class + ":"
                    # generated_recommendations = generate_recommendations(input_text)
                    ## st.write(generated_recommendations)

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    generate_svm_plot(prediction, predicted_class)
                    
                    st.write(Pepper_names)
                    
                    true_labels = ["pepper_Healthy", "pepper_CercosporaLeafSpot", "pepper_Fusarium", "pepper_Leaf_Curl"]  # True label for Pepper
                    predicted_labels = [predicted_class] * len(true_labels)  # Predicted label for Pepper
                    
                    # Add evaluation metrics display after generating SVM plot
                    display_evaluation_metrics(true_labels, predicted_labels)
                    
                    feedback()

with tab3:
    """
    The "Deep Learning and Machine Learning Integration: A Comprehensive Approach to Automated Crop Health Assessment Using CNN, ANN, and SVM" project aims to revolutionize crop health assessment in agriculture through cutting-edge technology and advanced machine learning techniques. By leveraging Convolutional Neural Networks (CNN), Artificial Neural Networks (ANN), and Support Vector Machines (SVM), the project endeavors to automate the process of evaluating crop health, thereby enhancing agricultural practices and promoting sustainability.


In this research, CNN serves as the primary tool for image processing and feature extraction from crop images, enabling the identification of visual cues indicative of crop health. The extracted features are then fed into ANN, which analyzes the data and generates recommendations or actions to be taken for diseased plants. These recommendations encompass treatment strategies, irrigation adjustments, and pest control measures, aimed at improving crop yield and minimizing losses.


Additionally, SVM plays a crucial role in the classification of crops based on the features extracted by CNN and processed by ANN. By training SVM to classify crops into categories such as healthy, diseased, or under stress, the project achieves a comprehensive assessment of crop health, facilitating informed decision-making for farmers and agricultural stakeholders.


Through the integration of CNN, ANN, and SVM, the project not only advances the field of automated crop health assessment but also contributes to the broader goals of precision agriculture and sustainable farming practices. By harnessing the power of deep learning and machine learning, this research endeavors to redefine how crop quality is evaluated, ultimately leading to higher crop yields, reduced losses, and a more resilient agricultural ecosystem.
    
    """
