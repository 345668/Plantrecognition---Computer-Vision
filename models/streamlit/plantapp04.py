import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
from keras.models import load_model
import numpy as np
import cv2

# Load the trained models
model1_vgg16 = load_model('best_model.h5')  # VGG16 model
model2_cnn = load_model('plant_disease_model_simpler_epoch150_02.h5')


# Define class labels (update these based on your model's training)
class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Set page configuration
st.set_page_config(
    page_title="Plant Recognition Project",
    layout="centered"  # Keep default layout
)

# Sidebar for Navigation
with st.sidebar:
    selected = option_menu(
        "Navigation", ["Home", "Prediction"],
        icons=["house", "graph-up"],
        menu_icon="cast", default_index=0
    )

# Home Page
if selected == "Home":
    st.title("Plant Recognition Project")
    st.subheader("Cohort: DataScientest May 2024")
    st.markdown("""
    **Team Members:**
    - Fares Naem
    - Nathalie Zahran
    - Mohammadamin Nooralikheirzad
    - Philippe Masindet
    
    **Supervisors:** [Names of Supervisors]
    
    **Date:** [Presentation Date]
    """)

    with st.expander("Introduction", expanded=False):
        st.write("""
        **Overview:** The Plant Recognition Project is designed to address challenges in plant species identification and disease detection using advanced AI techniques.
        
        **Goal:** To develop a robust AI system capable of accurately classifying plant species and detecting diseases from digital images.
        
        **Significance:** This tool aids in sustainable farming by reducing chemical use and supports both agricultural experts and hobbyists.
        """)

    with st.expander("Data Collection", expanded=False):
        st.write("""
        **Datasets Used:**
        - **PlantVillage Dataset:** 55,000 images across 38 categories.
        - **New Plant Diseases Dataset:** 70,297 training images, 17,573 validation images.
        - **Plant Disease Dataset:** 43,457 training images, 10,850 test images.
        
        **Importance:** The diverse datasets ensured comprehensive training, allowing for accurate and reliable model performance across varied conditions.
        """)

    with st.expander("Methodology - Data Preprocessing", expanded=False):
        st.write("""
        **Key Steps:**
        - Merging and augmenting datasets to address class imbalance.
        - Downscaling images for manageable training within computational limits.
        - Applying data augmentation to enhance model exposure to different scenarios.
        
        **Challenges:** Balancing image detail with training time and computational efficiency.
        """)

    with st.expander("Methodology - Model Selection", expanded=False):
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
        st.image(image, caption='Sample Image', use_column_width=True)

    with st.expander("Selected Model", expanded=False):

        st.write("""
        **Model Structure and Configuration:**
        - We used VGG16 pre-trained on ImageNet, customized by adding pooling layers, 
                 dense layers with ReLU, dropout for regularization, 
                 and a final SoftMax layer for classification, while freezing the base layers..
        """)
        st.write("""
        **Optimization Strategies and Techniques :**
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
        st.image(image, caption='areas affected by diseases', use_column_width=True)

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
        st.image(image, caption='confusion matrix', use_column_width=True)


    with st.expander("Challenges and Outlook", expanded=False):
        st.write("""
        **Challenges:**
        - Distinguishing between similar diseases.
        - Extensive training times due to computational limits.
        
        **Future Directions:**
        - Upgrade computational resources for faster training.
        - Expand datasets with real-world images to improve model robustness.
        - Enhance the user interface for mobile applications with real-time processing.
        """)

    with st.expander("Conclusion", expanded=False):
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

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Read the image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)  # Decode image from the bytes

        # Convert image to RGB (OpenCV loads images in BGR format by default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image based on selected model
        if model_choice == "Model 1 - VGG16":
            
            img_resized = cv2.resize(img_rgb, ( 256,  256))
            model = model1_vgg16
        else:
            img_resized = cv2.resize(img_rgb, (62, 62))
            model = model2_cnn

        # Convert the image to a numpy array and normalize
        img_array = np.array(img_resized)
        img_array = img_array.astype('float32') / 255.0  # Normalize the image to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]

        # Display the uploaded image
        st.image(img_rgb, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Show the result
        st.write(f"Prediction: {predicted_class}")
