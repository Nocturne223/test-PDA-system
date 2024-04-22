import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sklearn
 
from PIL import Image
from keras.models import load_model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load Models
model1 = load_model('official-models/LettuceModel.h5')  # saved model from training
model2 = load_model('official-models/CauliflowerModel.h5')  # saved model from training
model3 = load_model('official-models/SugarcaneModel-1.h5')  # saved model from training
model4 = load_model('official-models/PepperModel.h5')  # saved model from training

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

def preprocess_image(image):
    img = image.resize((100, 100))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_class(image, model):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class

def display_recommendations(predicted_class):
    class_name = None
    if predicted_class == 0:
        class_name = "Healthy"
    elif predicted_class == 1:
        class_name = "Disease"
    elif predicted_class == 2:
        class_name = "Nutrient Deficient"
    else:
        st.error("Error: Invalid prediction class.")
        return

    st.subheader("Recommendations:")
    if class_name in recommendations:
        for recommendation in recommendations[class_name]:
            st.write(recommendation)

def home():
    st.title("Crop Health Assessment App")
    st.write("Welcome to the Crop Health Assessment App! Use this app to analyze the health of your crops.")
    st.write("Select the 'Upload' page to upload an image of your crop.")

def upload():
    st.title("Upload Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # SVM Classifier
        st.subheader("SVM Classifier Prediction:")
        img_array = preprocess_image(image)
        img_array = img_array.flatten().reshape(1, -1)

        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train_flat, y_train)

        predicted_class = svm_model.predict(img_array)[0]

        if predicted_class == 0:
            st.write("Healthy")
        elif predicted_class == 1:
            st.write("Disease")
        elif predicted_class == 2:
            st.write("Nutrient Deficient")
        else:
            st.error("Error: Invalid prediction class.")

        display_recommendations(predicted_class)


# Function to display recommendations and predicted class
def display_recommendations(predicted_class):
    class_name = None
    if predicted_class == 0:
        class_name = "Healthy"
    elif predicted_class == 1:
        class_name = "Disease"
    elif predicted_class == 2:
        class_name = "Nutrient Deficient"
    else:
        st.error("Error: Invalid prediction class.")
        return

    st.subheader("Recommendations:")
    if class_name in recommendations:
        for recommendation in recommendations[class_name]:
            st.write(recommendation)

def home():
    st.title("Crop Health Assessment App")
    st.write("Welcome to the Crop Health Assessment App! Use this app to analyze the health of your crops.")
    st.write("Select the 'Upload' page to upload an image of your crop.")

def upload():
    st.title("Upload Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # SVM Classifier
        st.subheader("SVM Classifier Prediction:")
        img_array = preprocess_image(image)
        img_array = img_array.flatten().reshape(1, -1)

        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train_flat, y_train)

        predicted_class = svm_model.predict(img_array)[0]

        if predicted_class == 0:
            st.write("Healthy")
        elif predicted_class == 1:
            st.write("Disease")
        elif predicted_class == 2:
            st.write("Nutrient Deficient")
        else:
            st.error("Error: Invalid prediction class.")

        display_recommendations(predicted_class)

def about():
    st.title("About")
    st.write("This app is developed for crop health assessment using machine learning techniques.")

def main():
    page = st.sidebar.selectbox("Choose a page", ["Home", "Upload", "About"])

    if page == "Home":
        home()
    elif page == "Upload":
        upload()
    elif page == "About":
        about()

if __name__ == "__main__":
    main()