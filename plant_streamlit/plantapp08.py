import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pickle
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import os

# Workaround for OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load the trained models
model1_vgg16 = load_model('best_model.h5', compile=False)  # VGG16 model
model2_cnn = load_model('plant_disease_model_simpler_epoch150_02.h5', compile=False)

# Load class indices for VGG16
with open('class_indices.pkl', 'rb') as handle:
    class_indices = pickle.load(handle)
train_gen_class_indices = class_indices['train_gen_class_indices']
data_gen_class_indices = class_indices['data_gen_class_indices']

# Invert the class indices for VGG16
index_to_label_train_gen = {v: k for k, v in train_gen_class_indices.items()}
index_to_class_name = {v: k for k, v in data_gen_class_indices.items()}

# Define class labels (General for CNN)
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def predict_image_vgg16(image_path):
    # Load and preprocess the image for VGG16
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    predictions = model1_vgg16.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Map predicted index to the label used during training
    predicted_label = index_to_label_train_gen[predicted_class_index]

    # Map the label to the actual class name using the original data_generator mapping
    predicted_class_name = index_to_class_name[int(predicted_label)]

    return predicted_class_name, predicted_class_index, img_array

def predict_image_cnn(img_rgb):
    # Resize and preprocess the image for CNN
    img_resized = cv2.resize(img_rgb, (62, 62))
    img_array = np.array(img_resized) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model2_cnn.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_labels[predicted_class_index]

    return predicted_class_name

def generate_gradcam(image_path):
    # Use the VGG16 prediction function to get predictions and preprocessed image
    predicted_class_name, predicted_class_index, img_array = predict_image_vgg16(image_path)

    # Initialize Grad-CAM
    gradcam = Gradcam(model1_vgg16, model_modifier=ReplaceToLinear(), clone=True)

    # Create a score for the predicted class
    score = CategoricalScore(predicted_class_index)

    # Generate Grad-CAM heatmap
    cam = gradcam(score, img_array)[0]

    # Display the image with Grad-CAM
    plt.figure(figsize=(8, 8))
    plt.imshow(img_array[0].astype("uint8"))
    plt.imshow(cam, cmap='jet', alpha=0.5)  # Overlay Grad-CAM heatmap
    plt.title(f"Predicted: {predicted_class_name}")
    plt.axis('off')
    return plt

# Set page configuration
st.set_page_config(
    page_title="Plant Recognition Project",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stApp {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stSidebar {
        background-color: #4CAF50;
        color: #ffffff;
        border-radius: 10px;
        padding: 10px;
    }
    .stTitle {
        font-family: 'Segoe UI', sans-serif;
        color: #4CAF50;
    }
    .stHeader {
        font-family: 'Segoe UI', sans-serif;
        color: #2c3e50;
        font-weight: 300;
    }
    .stMarkdown p {
        font-family: 'Segoe UI', sans-serif;
        font-size: 1rem;
        line-height: 1.5;
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

# CSS to animate the symbol
animation_css = '''
<style>
@keyframes move-sprout {
  0% { transform: translateX(-100%); opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { transform: translateX(100%); opacity: 0; }
}

.sprout {
  font-size: 50px;
  position: absolute;
  top: 50%;
  left: 0;
  transform: translateY(-50%);
  animation: move-sprout 5s linear infinite;
}
</style>
'''

# Inject the CSS into the Streamlit app
st.markdown(animation_css, unsafe_allow_html=True)

# HTML for the symbol
sprout_html = '''
<div class="sprout">🍃 </div>
'''

# Inject the HTML into the Streamlit app
st.markdown(sprout_html, unsafe_allow_html=True)

# Sidebar for Navigation
with st.sidebar:
    selected = option_menu(
        "Navigation", ["Home", "Introduction", "Data Collection", "Methodology - Data Preprocessing",
                       "Methodology - Model Selection", "Model Architecture", "Selected Model", 
                       "Challenges and Outlook", "Conclusion", "Prediction", "Grad-CAM Interpretability"],
        icons=["house", "info-circle", "database", "gear", "bar-chart-line", "braces", "box", "list-task", 
               "check2-circle", "graph-up", "eye"], # "eye" icon for Grad-CAM
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#4CAF50"},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#567"},
            "nav-link-selected": {"background-color": "#2b7a78"},
        }
    )

# Home Page
if selected == "Home":
    st.title("🌱 Plant Recognition Project")
    st.subheader("Cohort: DataScientest May 2024")
    st.markdown("""
    **Team Members:**
    - Fares Naem
    - Nathalie Zahran
    - Mohammadamin Nooralikheirzad
    - Philippe Masindet
    
    **Supervisor:** Romain Lesieur
    
    **Date:** 09.08.2024
    """)

# Introduction Page
elif selected == "Introduction":
    st.title("Introduction")
    st.write("""
    **Overview:** The Plant Recognition Project is designed to address challenges in plant species identification and disease detection using advanced AI techniques.
    
    **Goal:** To develop a robust AI system capable of accurately classifying plant species and detecting diseases from digital images.
    
    **Significance:** This tool aids in sustainable farming by reducing chemical use and supports both agricultural experts and hobbyists.
    """)

# Data Collection Page
elif selected == "Data Collection":
    st.title("Data Collection")
    st.write("""
    **Datasets Used:**
    - **PlantVillage Dataset:** 55,000 images across 38 categories.
    - **New Plant Diseases Dataset:** 70,297 training images, 17,573 validation images.
    - **Plant Disease Dataset:** 43,457 training images, 10,850 test images.
    
    **Importance:** The diverse datasets ensured comprehensive training, allowing for accurate and reliable model performance across varied conditions.
    """)

    # Load the image from a file
    image = Image.open("imagesNumberFolders.png")
    # Display the image
    st.image(image, caption='Accuracy and Loss of the VGG16 Model Over Epochs', use_column_width=False)

    # Load the new image from a file
    merged_image = Image.open("merged.png")
    # Display the new image with a caption
    st.image(merged_image, caption='Final Dataset after augmenting merged colored and segmented datasets', use_column_width=True)

# Methodology - Data Preprocessing Page
elif selected == "Methodology - Data Preprocessing":
    st.title("Methodology - Data Preprocessing")
    st.write("""
    **Key Steps:**
    - Merging and augmenting datasets to address class imbalance.
    - Downscaling images for manageable training within computational limits.
    - Applying data augmentation to enhance model exposure to different scenarios.
    
    **Challenges:** Balancing image detail with training time and computational efficiency.
    """)

    # Load the image from a file
    image = Image.open("Preprocess.png")
    # Display the image with a caption
    st.image(image, caption='The difference of segmentation on color histogram', use_column_width=True)

# Methodology - Model Selection Page
elif selected == "Methodology - Model Selection":
    st.title("Methodology - Model Selection")
    st.write("""
    **Machine Learning Models:** Evaluated traditional algorithms like Decision Trees, SVM, KNN, and Random Forest.
    
    **Deep Learning Models:** Transitioned to CNNs, ultimately adopting VGG16 due to its superior performance in image classification.
    
    **Results:** VGG16 emerged as the most effective model, achieving high accuracy and reliability.
    """)

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv("models_comp.csv")
    # Display the DataFrame in the Streamlit app
    st.write("Comparison between different machine learning models and the two deep learning models:")
    st.dataframe(df)  # You can also use st.table(df) for a static table

    # Load the image from a file
    image = Image.open("accuracy_comp.png")
    # Display the image
    st.image(image, caption='Model Performance Comparsion', use_column_width=False)

# Model Architecture Page
if selected == "Model Architecture":
    st.title("Model Architecture")
    st.header("TensorBoard Graph")
    # Display TensorBoard Graph
    tensorboard_image = Image.open("tensorboard.png")
    st.image(tensorboard_image, caption="This TensorBoard graph depicts the detailed flow of operations and layers within the neural network, showcasing the complexity and connectivity of the model's architecture.")

    st.header("VGG16 Architecture Diagram")
    # Display VGG16 Architecture Diagram
    vgg16_image_path = Image.open("vgg16_architecture.png")
    st.image(vgg16_image_path, caption="Schematic representation of the VGG16 architecture as visualized by PlotNeuralNet, highlighting the sequential layers from input to output, including convolutional, pooling, and fully-connected layers.")

# Selected Model Page
elif selected == "Selected Model":
    st.title("Selected Model")

    st.write("""
    **Model Structure and Configuration:**
    - We used VGG16 pre-trained on ImageNet, customized by adding pooling layers, 
             dense layers with ReLU, dropout for regularization, 
             and a final SoftMax layer for classification, while freezing the base layers.
    """)
    st.write("""
    **Optimization Strategies and Techniques:**
    - **Learning Rate & Training:** Used exponential decay for stable training with the Adam optimizer
              and categorical cross-entropy loss.
    
    - **Model Monitoring:** Implemented Model Checkpoint to save the best-performing iterations. 
             
    - **Optimization Techniques:** Limited parameter search due to long training times, data 
             augmentation (rotation, flipping, zooming) for better generalization, 
             and mixed precision training for efficiency
    """)

    st.write("""
    **Model Interpretability:**
    - Grad-CAM analysis ensured that the model focused on relevant image sections,
              enhancing classification accuracy.
    """)

    # Load the image from a file
    image = Image.open("interpretab01.png")
    # Display the image
    st.image(image, caption='Areas affected by diseases', use_column_width=False)

    # Load the image from a file
    image = Image.open("interpretab02.png")
    # Display the image
    st.image(image, caption='Areas affected by diseases', use_column_width=False)

    st.write("""
    **Performance and results:**
    - Achieved 96% accuracy and a high F1-score, outperforming initial models.
    """)
    # Load the image from a file
    image = Image.open("acc_loss.png")
    # Display the image
    st.image(image, caption='Accuracy and Loss of the VGG16 Model Over Epochs', use_column_width=True)

    # Load the image from a file
    image = Image.open("confusion_Mat.png")
    # Display the image
    st.image(image, caption='Confusion matrix', use_column_width=True)

    # Load and display the classification report
    st.write("**Classification Report:**")
    classification_report_df = pd.read_csv('classification_report.csv')
    st.dataframe(classification_report_df)

# Challenges and Outlook Page
elif selected == "Challenges and Outlook":
    st.title("Challenges and Outlook")
    st.write("""
    **Challenges:**
    - Distinguishing between similar diseases.
    - Extensive training times due to computational limits.
    
    **Future Directions:**
    - Upgrade computational resources for faster training.
    - Expand datasets with real-world images to improve model robustness.
    - Enhance the user interface for mobile applications with real-time processing.
    """)

# Conclusion Page
elif selected == "Conclusion":
    st.title("Conclusion")
    st.write("""
    **Summary:** The Plant Recognition Project successfully demonstrated the application of deep learning in agriculture, paving the way for future innovations.
    
    **Impact:** The project’s success emphasizes the potential of AI in sustainable farming practices and contributes to the broader Agri-Tech field.
    """)

    st.markdown("**Thank you for your attention!**")

# Prediction Page
elif selected == "Prediction":
    st.title("Plant Recognition - Prediction Tool")
    st.write("""
    This page will be used to predict plant species and detect diseases using the trained model.
    """)

    # Select model
    model_choice = st.selectbox(
        "Choose a model for prediction:",
        ("Model 1 - VGG16", "Model 2 - CNN")
    )

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Read the image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)  # Decode image from the bytes
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB

        if model_choice == "Model 1 - VGG16":
            # Use the VGG16 prediction function
            predicted_class_name = predict_image_vgg16(uploaded_file)[0]
        else:
            # Use the CNN prediction function
            predicted_class_name = predict_image_cnn(img_rgb)

        # Display the uploaded image
        st.image(img_rgb, caption='Uploaded Image.', width=600)
        st.write(f"Prediction: {predicted_class_name}")

# Grad-CAM Interpretability Page
elif selected == "Grad-CAM Interpretability":
    st.title("Grad-CAM Interpretability")
    st.write("""
    This page will visualize the Grad-CAM heatmap to interpret the VGG16 model's predictions.
    """)

    uploaded_file = st.file_uploader("Choose an image for Grad-CAM...", type="jpg")

    if uploaded_file is not None:
        # Generate Grad-CAM visualization using the VGG16 prediction function
        gradcam_plot = generate_gradcam(uploaded_file)

        # Display the uploaded image and Grad-CAM
        st.pyplot(gradcam_plot)