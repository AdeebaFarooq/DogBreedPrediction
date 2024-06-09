import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import shutil

# 1. Welcome message and brief description
st.title("Dog Breed Predictor")
st.subheader("Upload the image of a dog and we will predict its breed!")
st.write("You can also try uploading the image of a human to see what happens. Have Fun!")

# 2. Add a file uploader widget to the sidebar
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
uploaded_image_path = ''


# Check if a file was uploaded
if uploaded_file is not None:
    col1, col2 = st.columns([2, 1])

    with col1:
        # Display the uploaded image
        image = Image.open(uploaded_file)

        # Calculate the new width based on the aspect ratio
        aspect_ratio = image.width / image.height
        new_width = int(400 * aspect_ratio)
        
        # Resize the image while maintaining aspect ratio
        image_resized = image.resize((new_width, 400))

        st.image(image_resized, caption='Uploaded Image')

        # Save the uploaded file to a directory named 'uploaded_files'
        save_dir = 'uploaded_files'
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        uploaded_image_path = file_path

    with col2:
        # Create a button for breed detection
        if st.button('Detect Breed'):
            # Perform breed detection when the button is clicked
            
            # Use the pre-trained CNN model to predict the dog breed from the uploaded image
            # Load the pre-trained CNN model Xception
            Xception_model = load_model('weights.best_adam.Xception.hdf5')

            # Pre-process the data
            from keras.preprocessing import image
            from tqdm import tqdm

            def path_to_tensor(img_path):
                # loads RGB image as PIL.Image.Image type
                img = image.load_img(img_path, target_size=(224, 224))
                # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
                x = image.img_to_array(img)
                # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
                return np.expand_dims(x, axis=0)

            def paths_to_tensor(img_paths):
                list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
                return np.vstack(list_of_tensors)

            # Write a function that takes a path to an image as input
            # and returns the dog breed that is predicted by the model.
            from extract_bottleneck_features import *
            from glob import glob

            # Load list of dog names
            # Read the file and load dog names into a list
            file_path = "dog_names.txt"
            dog_names = []
            with open(file_path, "r") as f:
                for line in f:
                    dog_names.append(line.strip())

            def Xception_Predict_Breed(img_path):
                # extract bottleneck features
                bottleneck_feature = extract_Xception(path_to_tensor(img_path))
                # obtain predicted vector
                predicted_vector = Xception_model.predict(bottleneck_feature)
                # return dog breed that is predicted by the model
                return dog_names[np.argmax(predicted_vector)]

            # Algorithm to detect dog or human
            # --------------------------------------------------------------------------
            # returns "True" if face is detected in image stored at img_path
            def face_detector(img_path):
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # extract pre-trained face detector
                face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
                faces = face_cascade.detectMultiScale(gray)
                return len(faces) > 0 

            from keras.applications.resnet50 import ResNet50
            # define ResNet50 model
            ResNet50_model_dog = ResNet50(weights='imagenet')

            from keras.applications.resnet50 import preprocess_input

            def ResNet50_predict_labels(img_path):
                # returns rprediction vector for image located at img_path
                img = preprocess_input(path_to_tensor(img_path))
                return np.argmax(ResNet50_model_dog.predict(img))

            # returns "True" if a dog is detected in the image stored at img_path
            def dog_detector(img_path):
                prediction = ResNet50_predict_labels(img_path)
                return ((prediction <=268) & (prediction >= 151))

            def dog_human_detector(img_path):
                if(dog_detector(img_path)):
                    st.success("Dog detected with breed " + Xception_Predict_Breed(img_path))
                elif(face_detector(img_path)):
                    st.success("Human detected with resembling dog breed " + Xception_Predict_Breed(img_path))
                else:
                    st.error("There is no human or dog in the image.")

            dog_human_detector(uploaded_image_path)  # Placeholder for breed detection result
            
            # Delete the 'uploaded_files' directory after breed detection
            if os.path.exists('uploaded_files'):
                shutil.rmtree('uploaded_files')