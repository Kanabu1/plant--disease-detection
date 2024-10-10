# plant--disease-detection
# Plant Disease Detection using Deep Learning

This project demonstrates how to build a deep learning model for detecting plant diseases in images using TensorFlow and Keras. The model is trained on a dataset of images of healthy and diseased plants and can be used to predict the disease of new images.

## Dataset

The dataset used in this project is the PlantVillage dataset, which contains images of 38 different plant diseases.

## Model

The model used in this project is a convolutional neural network (CNN). The CNN consists of three convolutional layers, followed by three max-pooling layers, a flatten layer, and two dense layers. The convolutional layers are used to extract features from the images, and the max-pooling layers are used to reduce the dimensionality of the feature maps. The flatten layer converts the feature maps into a one-dimensional vector, and the dense layers are used to classify the images into different disease categories.

## Training

The model is trained using the Adam optimizer and the categorical cross-entropy loss function. The model is trained for 10 epochs with a batch size of 16. The model's performance is evaluated on the validation set after each epoch.

## Evaluation

The model's performance is evaluated on the testing set after training. The model's accuracy, precision, and recall are calculated. 
