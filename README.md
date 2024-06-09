# Dog Breed Predictor
This repository contains a dog breed prediction app created using a pre-trained convolutional neural network for image classification and deployed using streamlit.

The model is being trained at google collab. We used a model from tensorflow hub.
Model Architecture:

The model employs transfer learning by utilizing a pre-trained CNN (e.g., VGG16, InceptionV3) as a feature extractor.
The pre-trained layers are frozen to prevent re-training.
New fully-connected layers are added on top of the pre-trained model to learn dog breed classification.

The model is trained on the provided dog breed dataset.
Data augmentation techniques (e.g., random cropping, flipping) are optionally used to increase the size of the training set and improve generalization.
