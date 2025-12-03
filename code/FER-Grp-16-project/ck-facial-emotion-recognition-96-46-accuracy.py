#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# ### Objective
# 
# - This notebook demonstrates a methodology for classifying images based on facial expressions into distinct emotion categories. The primary focus is on Facial Emotion Recognition (FER), a challenging yet fascinating area of machine learning.
# 
# ### Dataset Overview
# 
# - For this task, I have utilized the CK+ (Cohn-Kanade Plus) dataset. It's crucial to note that CK+ is a relatively small dataset comprising 981 images. Despite its limited size, CK+ is widely recognized for its high-quality, labeled facial expression images, making it a suitable choice for deep learning experiments in emotion classification.
# 
# ### Model Architecture and Implementation
# 
# - The core of this project is the implementation of a 2D Convolutional Neural Network (CNN2D). CNNs are particularly effective for image classification tasks due to their ability to extract features from spatial data. The architecture of the CNN2D model used in this project is designed to efficiently process and classify facial expression images.
# 
# ### Results
# 
# - The model achieved a remarkable validation accuracy of 96.46%. This high level of accuracy indicates the model's effectiveness in recognizing and classifying emotions from facial expressions.

# ### Importing Necessary Libraries

# In[ ]:


import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations and working with arrays
import cv2 as cv  # For image processing tasks
import os  # For interacting with the operating system, like file paths
import tensorflow as tf  # For building and training neural network models

from keras.preprocessing.image import ImageDataGenerator  # For real-time data augmentation
from tensorflow.keras.models import load_model  # For loading a saved Keras model
from keras.models import Sequential  # For creating a linear stack of layers in the model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten  # For building model layers
# from keras.optimizers import Adam  # For optimization algorithms
from keras.layers import BatchNormalization  # For applying Batch Normalization in neural network layers
from keras.regularizers import l2  # For applying L2 regularization to prevent overfitting
from keras.callbacks import ReduceLROnPlateau, EarlyStopping  # Importing specific callback functions

import warnings  # For handling warnings
import sys  # For interacting with the Python interpreter
if not sys.warnoptions:
    warnings.simplefilter("ignore")  # Ignore simple warnings if not already done
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignore deprecation warnings


# ### Reading and Splitting the Data using 'os' and 'shutil'
# 
# 
# > ### Why *shutil*?
# > 
# > - The notebook uses *shutil* to copy image files from the original dataset directory to separate training and testing directories as part of the dataset preparation process for a machine learning task. This module is chosen for its efficiency and ability to handle file operations in a high-level, user-friendly manner.

# In[ ]:


import shutil

# Define the path to your original dataset and the paths where you want to store your train and test datasets
original_dataset_dir = 'CK_plus_dataset'
train_dir = 'CK+48_train'
test_dir = 'CK+48_test'

# Create directories for training and testing datasets if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the split ratio
train_ratio = 0.8

# Loop through each emotion category in the original dataset
for emotion in os.listdir(original_dataset_dir):
    emotion_dir = os.path.join(original_dataset_dir, emotion)
    if os.path.isdir(emotion_dir):
        # Get a list of all the image filenames in the emotion category
        images = [f for f in os.listdir(emotion_dir) if os.path.isfile(os.path.join(emotion_dir, f))]

        # Randomly shuffle the list of image filenames
        np.random.shuffle(images)

        # Split the list of image filenames into training and testing sets
        train_size = int(len(images) * train_ratio)
        train_images = images[:train_size]
        test_images = images[train_size:]

        # Create directories for the emotion category in the train and test datasets
        train_emotion_dir = os.path.join(train_dir, emotion)
        test_emotion_dir = os.path.join(test_dir, emotion)
        os.makedirs(train_emotion_dir, exist_ok=True)
        os.makedirs(test_emotion_dir, exist_ok=True)

        # Copy the images into the corresponding directories
        for image in train_images:
            shutil.copy(os.path.join(emotion_dir, image), os.path.join(train_emotion_dir, image))
        for image in test_images:
            shutil.copy(os.path.join(emotion_dir, image), os.path.join(test_emotion_dir, image))

print("Dataset splitting complete")


# ### Extracting the Dimensions of the image

# In[ ]:


import cv2

# Load the image
img = cv2.imread('CK+48_train/disgust/S098_003_00000011.png')

# Get dimensions
height, width, channels = img.shape
print(f'Dimensions: {width}x{height}')


# ### Creating a Training Image Data Generator

# In[ ]:


# Create a data generator with augmentation
train_data_generator = ImageDataGenerator(
    rescale=1./255,  # Rescale the pixel values (normalization)
    rotation_range=15,  # Random rotation in the range of 15 degrees
    width_shift_range=0.15,  # Random horizontal shifts (15% of total width)
    height_shift_range=0.15,  # Random vertical shifts (15% of total height)
    shear_range=0.15,  # Random shearing transformations
    zoom_range=0.15,  # Random zoom range
    horizontal_flip=True,  # Randomly flip inputs horizontally
)

# Load images from the directory and apply the defined transformations
fer_training_data = train_data_generator.flow_from_directory(
    'CK+48_train',  # Path to the training data
    target_size=(48, 48),  # Resize images to 48x48
    batch_size=64,  # Number of images to yield per batch
    color_mode='grayscale',  # Load images in grayscale
    class_mode='categorical'  # Labels will be returned in categorical format
)

fer_training_data


# ### Creating a Test Image Data Generator

# In[ ]:


# Initialize an ImageDataGenerator for test data with rescaling
# Rescales images by dividing pixel values by 255 (normalization)
test_data_generator = ImageDataGenerator(rescale=1./255)

# Creates a data generator for the test dataset
# flow_from_directory method loads images from a directory,
# in this case, '/kaggle/working/CK+48_test'
fer_test_data = test_data_generator.flow_from_directory(
    'CK+48_test',  # Directory path for test images
    target_size = (48, 48),  # Resizes images to 48x48 pixels
    batch_size = 64,  # Number of images to yield per batch
    color_mode = 'grayscale',  # Specifies that images are in grayscale
    class_mode = 'categorical'  # Images are classified categorically
)

# fer_test_data is now a generator that yields batches of test images and their labels
fer_test_data


# # Fundamental Machine Learning Techniques
# 
# > 1. **Batch Normalization**
# >     - **What it is**: Batch Normalization is a technique to normalize the inputs of each layer within a neural network. It adjusts and scales the activations.
# >     - **How it works**: During training, it normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation. Then, it applies a scale factor and an offset. This helps in mitigating the problem of internal covariate shift, where the distribution of each layer’s inputs changes during training.
# >     - **Benefits**: It stabilizes the learning process and dramatically reduces the number of training epochs required to train deep networks. It can also help with gradient problems, allowing for higher learning rates.
# 
# > 2. **L2 Regularization**
# >    - **What it is**: L2 regularization, also known as Ridge Regression or weight decay, is a technique used to prevent overfitting by penalizing large weights in your model.
# >    - **How it works**: It adds a regularization term to the loss function. This term is the sum of the squares of all the feature weights multiplied by a regularization parameter (lambda). The effect is to shrink the weights of less important features.
# >    - **Benefits**: It's effective at preventing overfitting by ensuring that the model weights stay small, which can lead to a simpler model that generalizes better to new data.
# 
# > 3. **Early Stopping**
# >    - **What it is**: Early stopping is a form of regularization used to avoid overfitting when training a learner with an iterative method, like gradient descent.
# >    - **How it works**: This approach involves monitoring the performance of the model on a validation set during the training process and stopping the training once the performance on the validation set starts to degrade (e.g., the validation loss starts increasing).
# >    - **Benefits**: It prevents overfitting by terminating the training process before the model starts to overfit. This also saves computational resources and can result in a more generalized model.
# 
# > 4. **Max Pooling**
# >    - **What it is**: Max pooling is a downsampling operation commonly used in convolutional neural networks.
# >    - **How it works**: It reduces the spatial dimensions (width and height) of the input volume for the next convolutional layer. It works by sliding a window (usually 2x2) over the input and taking the maximum value from the window.
# >    - **Benefits**: It reduces the computational load, memory usage, and the number of parameters, helping to prevent overfitting. It also helps in extracting dominant features which are rotational and positional invariant.
# 
# > 5. **Dropout**
# >    - **What it is**: Dropout is a regularization technique used to prevent overfitting in neural networks.
# >    - **How it works**: During training, some number of layer outputs are randomly ignored or “dropped out”. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.
# >    - **Benefits**: As a form of regularization, dropout reduces the complexity of the model. It forces the network to not rely on any one feature, hence preventing overfitting. It makes the model more robust and can lead to better generalization.

# # Adam and Nadam Optimizers
# 
# > **1. Adam (Adaptive Moment Estimation):**
# > 
# >  - How it Works: Adam combines ideas from AdaGrad and RMSProp. It maintains two moving averages per parameter: one for the gradients (like RMSProp) and one for the square of the gradients (like AdaGrad). It uses these moving averages to adapt the learning rate for each parameter.
# > 
# > - Why Use it: Adam is particularly effective in cases with large datasets or parameters. It's known for its efficiency with large problems involving a lot of data or parameters. Adam is also quite robust to the choice of hyperparameters.
# 
# > **2. Nadam (Nesterov-accelerated Adaptive Moment Estimation):**
# >
# > - How it Works: Nadam is essentially Adam with Nesterov momentum. It incorporates the Nesterov accelerated gradient into the Adam update rule, which helps in gaining a faster convergence and smoother optimization trajectory.
# >
# > - Why Use it: Nadam can be more efficient than Adam, especially in scenarios where the momentum helps navigate the optimization landscape more effectively. This can be particularly useful in deep learning models where finding the optimal solution efficiently is crucial.

# In[ ]:


# Importing the optimizers module from TensorFlow's Keras library
from tensorflow.keras import optimizers

# Initializing a list of optimizers with specific configurations
optims = [
    optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
    optimizers.Adam(0.001),
]


# # Defining the Model

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(
        Conv2D(
            filters=512,
            kernel_size=(5,5),
            input_shape=(48, 48, 1),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_1'
        )
    )
model.add(BatchNormalization(name='batchnorm_1'))
model.add(
        Conv2D(
            filters=256,
            kernel_size=(5,5),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_2'
        )
    )
model.add(BatchNormalization(name='batchnorm_2'))

model.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_1'))
model.add(Dropout(0.25, name='dropout_1'))

model.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_3'
        )
    )
model.add(BatchNormalization(name='batchnorm_3'))
model.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_4'
        )
    )
model.add(BatchNormalization(name='batchnorm_4'))

model.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_2'))
model.add(Dropout(0.25, name='dropout_2'))

model.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_5'
        )
    )
model.add(BatchNormalization(name='batchnorm_5'))
model.add(
        Conv2D(
            filters=512,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_6'
        )
    )
model.add(BatchNormalization(name='batchnorm_6'))

model.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_3'))
model.add(Dropout(0.25, name='dropout_3'))

model.add(Flatten(name='flatten'))

model.add(
        Dense(
            256,
            activation='elu',
            kernel_initializer='he_normal',
            name='dense_1'
        )
    )
model.add(BatchNormalization(name='batchnorm_7'))

model.add(Dropout(0.25, name='dropout_4'))

model.add(
        Dense(
            7,
            activation='softmax',
            name='out_layer'
        )
    )

model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

model.summary()


# ### Callbacks

# In[ ]:


# Reduce learning rate when a metric has stopped improving
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=11,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [
    early_stopping,
    lr_scheduler,
]


# # Training the Model

# In[ ]:


batch_size = 64
history = model.fit(
    fer_training_data,
    epochs=60, 
    validation_data=fer_test_data,
    batch_size = 64,
    callbacks=callbacks,
)

model.save('CK_Model_Grp16/my_fer_model.h5')


# # Visualizing Loss and Accuracy

# In[ ]:


import plotly.graph_objects as go

# Create traces
fig = go.Figure()

# Training vs. Validation Accuracy
fig.add_trace(go.Scatter(x=list(range(1, len(history.history['accuracy']) + 1)), y=history.history['accuracy'], mode='lines+markers', name='Training Accuracy'))
fig.add_trace(go.Scatter(x=list(range(1, len(history.history['val_accuracy']) + 1)), y=history.history['val_accuracy'], mode='lines+markers', name='Validation Accuracy'))

# Layout for Accuracy
fig.update_layout(title='Training vs. Validation Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy', template="plotly_white")

# Show the plot
fig.show()

# New figure for loss
fig = go.Figure()

# Training vs. Validation Loss
fig.add_trace(go.Scatter(x=list(range(1, len(history.history['loss']) + 1)), y=history.history['loss'], mode='lines+markers', name='Training Loss'))
fig.add_trace(go.Scatter(x=list(range(1, len(history.history['val_loss']) + 1)), y=history.history['val_loss'], mode='lines+markers', name='Validation Loss'))

# Layout for Loss
fig.update_layout(title='Training vs. Validation Loss', xaxis_title='Epoch', yaxis_title='Loss', template="plotly_white")

# Show the plot
fig.show()


# # Model Evaluation

# In[ ]:


# Evaluate the model on the validation dataset
loss, accuracy = model.evaluate(fer_test_data, verbose=1)

# Print the loss and accuracy
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy * 100:.2f}%')


# # Conclusion
# 
# This notebook has presented an approach to Facial Emotion Recognition (FER) using a 2D Convolutional Neural Network (CNN2D), demonstrating the power of deep learning in interpreting and classifying human emotions from facial expressions. 
# 
# The model's impressive performance, achieving a 96.46% validation accuracy on the CK+ dataset, underscores the potential of CNN2D in accurately identifying emotions, even with a relatively small dataset.
# 
# However, it's important to recognize the limitations inherent in this study. The CK+ dataset, while high in quality, is limited in its size and diversity. This limitation raises questions about the model's generalizability and effectiveness when faced with more diverse and extensive datasets.
# 
# Your feedback and questions are highly appreciated. I am also open to collaborations on similar projects. For further discussions or networking opportunities, please do not hesitate to connect with me on LinkedIn: [Farneet Singh](https://www.linkedin.com/in/farneet-singh-6b155b208).
# 
# Thank you for going through this notebook, and I look forward to your valuable input and potential collaborations! Stay Curious :)
