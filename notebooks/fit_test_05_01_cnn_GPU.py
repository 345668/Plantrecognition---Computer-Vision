import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import seaborn as sns
import random
import tensorflow as tf
import mahotas
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def read_image(base_path, key, imgName):
    """
     Reads an image from a specified file path constructed from base path, key, and image name.

     Parameters:
     base_path (str): The base directory path where images are stored.
     key (tuple): A tuple containing two strings used to create a subdirectory name.
     imgName (str): The name of the image file to be read.

     Returns:
     img (ndarray): The image read from the specified file path.
     """
    # Construct the file path by joining the base path, a subdirectory (constructed from key), and the image name
    file_path = os.path.join(base_path, key[0] + "___" + key[1], imgName)

    # Read the image from the constructed file path using OpenCV
    img = cv2.imread(file_path)

    return img


# function returns a dictionary of images names sorted in this form {(plant, disease) : [img1, img2, ....]}
# where images stored in folders having names in this form plant___disease
def read_images_names(folder_path):
    # List all files in the dataset directory
    all_folder_names = os.listdir(folder_path)

    # store the names of training images depending on their folder names (plant___disease)
    images_names_dict = {
        (folder_name.split("___")[0], folder_name.split("___")[1]): os.listdir(os.path.join(folder_path, folder_name))
        for folder_name in all_folder_names}
    return images_names_dict


def sample_2Dimgs_gray_resized(images_names_dict, new_dimensions, n, train_dir):
    """
      Samples, converts to grayscale, resizes, and normalizes a subset of images from a given directory.

      Parameters:
      images_names_dict (dict): Dictionary where keys are tuples and values are lists of image file names.
      new_dimensions (tuple): The desired dimensions (width, height) to resize the images.
      n (int): The number of images to sample from each list in images_names_dict.
      train_dir (str): The base directory path where images are stored.

      Returns:
      imgs_sample (list): List of processed image arrays.
      label (list): List of labels corresponding to the sampled images.
      """
    label = []
    imgs_sample = []
    for key, values in images_names_dict.items():
        for imgName in values[0:n]:
            label.append(list(key))
            img = read_image(train_dir, key, imgName)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA) / 255
            imgs_sample.append(img)

    # Return the read image
    return imgs_sample, label


# Directory containing the dataset
# Define paths relative to the script's location
current_dir = os.getcwd()
# Navigate to the parent directory
parent_dir = os.path.dirname(current_dir)

# train data dir
train_dir = os.path.join(parent_dir, 'plantReco/data/images/Plant_Diseases_Dataset/train')

images_names_dict = read_images_names(train_dir)

n = 400 # number of samples for each plant type - disease type
new_dim = (64, 64) # images new dimension
# reading images and their labels
imgs_sample, imgs_label = sample_2Dimgs_gray_resized(images_names_dict, new_dim, n, train_dir)

imgs_label_plant = [em[0] for em in imgs_label]
imgs_label_disease = [em[1] for em in imgs_label]

# Plant type branch
plant_types = list(set(imgs_label_plant))  # get unique values
# Disease branch
disease_types = list(set(imgs_label_disease))  # get unique values
# stor labels as dataframes
imgs_label_df = pd.DataFrame()
imgs_label_df['plant'] = imgs_label_plant
imgs_label_df['disease'] = imgs_label_disease

# initialize an encoder for plant type and disease type
plant_encoder = LabelEncoder()
disease_encoder = LabelEncoder()
# encode the labels
imgs_label_df['plant'] = plant_encoder.fit_transform(imgs_label_df['plant'])
imgs_label_df['disease'] = disease_encoder.fit_transform(imgs_label_df['disease'])
# convert images list to numpy arrray
imgs_sample = np.array(imgs_sample)
X_train, X_test, y_train, y_test = train_test_split(imgs_sample, imgs_label_df, test_size=0.2, random_state=42, shuffle=True)

# Performing one-hot encoding with dtype as float
y_train_plant_encoded = pd.get_dummies(y_train['plant'], dtype=float)
# Performing one-hot encoding with dtype as float
y_train_disease_encoded = pd.get_dummies(y_train['disease'], dtype=float)

# Performing one-hot encoding with dtype as float
y_test_plant_encoded = pd.get_dummies(y_test['plant'], dtype=float)
# Performing one-hot encoding with dtype as float
y_test_disease_encoded = pd.get_dummies(y_test['disease'], dtype=float)

# number of Plant type branch
num_plant_types = len(plant_types)
# number of Disease branch
num_diseases = len(disease_types)

new_width = new_dim[0]
new_height = new_dim[1]

# define the model

# Define the input
input_layer = Input(shape=(new_width, new_height, 1))

# Shared convolutional layers with reduced filters
x = Conv2D(8, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Separate branches for plant type and disease prediction
# Plant type branch
plant_type_output = Dense(num_plant_types, activation='softmax', name='plant_type_output')(x)

# Disease branch
disease_output = Dense(num_diseases, activation='softmax', name='disease_output')(x)

# Define the model
model = Model(inputs=input_layer, outputs=[plant_type_output, disease_output])

# Compile the model
model.compile(optimizer='adam',
              loss={'plant_type_output': 'categorical_crossentropy', 'disease_output': 'categorical_crossentropy'},
              metrics={'plant_type_output': 'accuracy', 'disease_output': 'accuracy'})

# train the model
history = model.fit(X_train,
                    {'plant_type_output': y_train_plant_encoded, 'disease_output': y_train_disease_encoded},
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, {'plant_type_output': y_test_plant_encoded, 'disease_output': y_test_disease_encoded}),verbose=1)

# save the entire cnn model after training
model.save('cnn_model.h5')

# Assuming 'history' is the object returned by model.fit()
# Extract accuracy and validation accuracy
history_dict = history.history
accuracy = history_dict['plant_type_output_accuracy']  # Replace 'plant_type_output_accuracy' with the actual key if different
val_accuracy = history_dict['val_plant_type_output_accuracy']  # Replace 'val_plant_type_output_accuracy' with the actual key if different

# save the history as dataframe
# Convert the history dictionary to a DataFrame
history_df = pd.DataFrame(history_dict)
# Save the DataFrame to a CSV file
history_df.to_csv('cnn_training_history.csv', index=False)

# If you have 'disease_output_accuracy' as well, you can extract that similarly
# For example:
disease_accuracy = history_dict['disease_output_accuracy']
val_disease_accuracy = history_dict['val_disease_output_accuracy']

# Define the number of epochs
epochs = range(1, len(accuracy) + 1)

# Plot accuracy for both plant type and disease type
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'bo', label='Training accuracy (Plant Type)')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy (Plant Type)')
plt.title('Training and Validation Accuracy (Plant Type)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, disease_accuracy, 'ro', label='Training accuracy (Disease)')
plt.plot(epochs, val_disease_accuracy, 'r', label='Validation accuracy (Disease)')
plt.title('Training and Validation Accuracy (Disease)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("model_cnnn.png")
plt.show()
