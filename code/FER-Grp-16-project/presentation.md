
# Presentation: Real-Time Facial Emotion Recognition System

**Group:** 16
**Project:** Final Year Project

---

## Slide 1: Title Slide

*   **Title:** Real-Time Facial Emotion Recognition using Deep Learning
*   **Subtitle:** An FYP by Group 16
*   **Team Members:** [List your names here]
*   **University Logo**

---

## Slide 2: Project Overview & Agenda

*   **Introduction:** What is Facial Emotion Recognition (FER)? Why is it a significant field of study?
*   **Problem Statement:** The goal is to develop a system that can accurately identify human emotions from facial expressions in real-time.
*   **Agenda:**
    1.  Motivation & Applications
    2.  System Architecture
    3.  Dataset & Preprocessing
    4.  Model Architecture & Training
    5.  Real-Time Detection System
    6.  Results & Evaluation
    7.  Live Demonstration
    8.  Challenges & Future Work
    9.  Conclusion

---

## Slide 3: Motivation & Applications

*   **Why FER?**
    *   Human-Computer Interaction (HCI): Creating more empathetic and responsive machines.
    *   Mental Health: Assisting therapists by providing objective data on a patient's emotional state.
    *   Marketing & Customer Feedback: Gauging customer reactions to products or advertisements.
    *   Driver Safety: Detecting driver fatigue or distraction.
    *   Entertainment: Interactive gaming and experiences.

---

## Slide 4: System Architecture

*   **A High-Level View of Our Project Pipeline:**
    *   **Data Collection:** Utilized the standard FER2013 dataset.
    *   **Model Training:** Designed, trained, and evaluated a Convolutional Neural Network (CNN) using Python, Keras, and TensorFlow.
    *   **Real-Time Implementation:** Built a live video-feed application using OpenCV to capture facial data.
    *   **Emotion Classification:** Deployed the trained model to classify emotions in real-time.
    *   **Reporting:** Developed a system to log and summarize detected emotions for analysis.

*(You can create a simple block diagram to visualize this flow.)*

---

## Slide 5: The Dataset: FER2013

*   **Source:** A well-known public dataset for facial emotion recognition challenges.
*   **Specifications:**
    *   **Size:** 35,887 total images.
    *   **Training Data:** 28,709 images.
    *   **Public Test Data:** 7,178 images.
*   **Image Properties:**
    *   **Dimensions:** 48x48 pixels.
    *   **Color:** Grayscale.
*   **Classes (7 Emotions):**
    *   Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

*(Show sample images from the `FER 2013/train` directory for each emotion to illustrate.)*

---

## Slide 6: Model Architecture (The "Brain")

*   We designed a **Convolutional Neural Network (CNN)**, which is the industry standard for image classification tasks.
*   **Key Components:**
    *   **Input Layer:** Accepts 48x48 grayscale images.
    *   **5 Convolutional Blocks:** These layers automatically learn features from the face, like the shape of the eyes, mouth, and nose. Each block contains:
        *   `Conv2D` Layer (Feature extraction)
        *   `BatchNormalization` (Stabilizes training)
        *   `ReLU` Activation (Introduces non-linearity)
        *   `MaxPooling2D` (Reduces dimensionality)
        *   `Dropout` (Prevents overfitting)
    *   **2 Fully-Connected (Dense) Layers:** These layers take the learned features and perform the final classification.
    *   **Output Layer:** A `Softmax` activation function that outputs the probability for each of the 7 emotions.
*   **Total Parameters:** ~3.58 Million

---

## Slide 7: Model Training & Optimization

*   **Frameworks:** TensorFlow with Keras API.
*   **Training Process:**
    *   **Optimizer:** `Adam` (An efficient and popular choice).
    *   **Initial Learning Rate:** 0.0005.
    *   **Loss Function:** `categorical_crossentropy` (Standard for multi-class classification).
*   **Ensuring a Robust Model:**
    *   **`ReduceLROnPlateau`:** Automatically reduces the learning rate if the model's performance stops improving.
    *   **`EarlyStopping`:** Stops the training process when validation loss no longer improves, preventing overfitting and saving time.
    *   **`ModelCheckpoint`:** Saves the best version of the model during the training phase.
*   **Result:** The model was trained for 20 epochs before early stopping, achieving a validation accuracy of **~62%**.

---

## Slide 8: Training Performance

*   **Visualizing the Learning Process:**
    *   The graphs show the model's `Loss` and `Accuracy` on both the training and validation datasets over each epoch.
    *   **Loss Graph:** We can see the error rate decreasing steadily, indicating the model is learning.
    *   **Accuracy Graph:** The accuracy increases and then plateaus, showing the model has reached its peak performance on the given data.
    *   The close alignment of training and validation curves indicates that our use of Dropout and Batch Normalization was effective in preventing significant overfitting.

*(Here, you should display the `Results/Accuracy Curve.jpg` image.)*

---

## Slide 9: The Real-Time Detection System

*   **Technology Stack:**
    *   **`OpenCV`:** For capturing the webcam feed and performing initial face detection.
    *   **`Haar Cascade Classifier`:** A fast and efficient pre-trained algorithm used to find the location of faces in each frame.
*   **Workflow:**
    1.  Capture a frame from the webcam.
    2.  Convert the frame to grayscale.
    3.  Use the Haar Cascade to detect all faces and return their coordinates (x, y, w, h).
    4.  For each detected face:
        a.  Draw a bounding box around it.
        b.  Crop the face region and resize it to 48x48 pixels.
        c.  Preprocess the image to match the model's input format.
        d.  Feed the face into our trained Keras model (`FER_Model_Grp16.h5`).
        e.  Display the predicted emotion label on the video feed.

---

## Slide 10: Reporting & Analysis

*   **Going Beyond Detection:** Our system doesn't just display emotions; it logs them for analysis.
*   **Automated Report Generation:** At the end of each session, the system generates three files:
    1.  **`detailed_log_{timestamp}.csv`:** A frame-by-frame log of every emotion detected.
    2.  **`statistics_{timestamp}.csv`:** A summary of emotion counts and percentages.
    3.  **`summary_report_{timestamp}.txt`:** A human-readable summary of the entire session, including the most frequently detected emotion.
*   **Purpose:** This provides valuable quantitative data for analysis and evaluation of the system's performance in a real-world scenario.

---

## Slide 11: Results & Example Detections

*   This slide demonstrates the model's performance on sample images.
*   The images show the bounding box correctly identifying the face and the corresponding emotion label predicted by our model.

*(Display the images from the `Results` folder, such as `Happy Face.jpg`, `Sad Face.jpg`, `Surprise Face.jpg`, etc. Arrange them in a grid.)*

---

## Slide 12: Live Demonstration

*   At this point, you will run the `main.py` script.
*   **Demonstration Checklist:**
    1.  Start the script and explain what the console output means.
    2.  Position your face in front of the webcam.
    3.  Clearly show the bounding box and the predicted emotion label.
    4.  Try to express different emotions (happy, sad, surprise, neutral) to show the system's responsiveness.
    5.  Quit the application by pressing 'q'.
    6.  Briefly open and show the generated report files (`summary_report.txt`) to the audience.

---

## Slide 13: Web Application (TF.js)

*   **Extending the Project:** To demonstrate versatility, the Keras model was converted to a web-friendly format using TensorFlow.js (`tfjs_model`).
*   **`convert_model.py`:** This script handles the conversion from `.h5` to the `model.json` and binary shard files.
*   **`example.js`:** A simple JavaScript application that loads the converted model and can perform emotion recognition directly in a web browser.
*   **Benefit:** This showcases how the model can be deployed on different platforms (Desktop App vs. Web App), making it highly accessible.

*(If you have time, you can also demonstrate this web version.)*

---

## Slide 14: Challenges & Future Work

*   **Challenges Faced:**
    *   **Data Quality:** The FER2013 dataset contains some noisy and mislabeled images.
    *   **Emotion Ambiguity:** Certain emotions (e.g., 'fear' and 'surprise') can have similar facial features, making them hard to distinguish.
    *   **Real-world Conditions:** Lighting, head pose, and occlusions (like glasses or hands) can affect detection accuracy.
*   **Future Improvements:**
    *   **Advanced Models:** Experiment with more complex architectures like ResNet or Vision Transformers.
    *   **Data Augmentation:** Apply more advanced data augmentation techniques to make the model more robust.
    *   **Larger Datasets:** Train the model on newer, larger, and more diverse datasets.
    *   **Head Pose Correction:** Implement algorithms to normalize head orientation before classification.

---

## Slide 15: Conclusion

*   **Achievements:**
    *   Successfully designed and trained a deep learning model for Facial Emotion Recognition with ~62% validation accuracy.
    *   Developed a real-time application in Python to deploy the model on a live webcam feed.
    *   Implemented a comprehensive logging and reporting system for data analysis.
    *   Demonstrated model portability by converting and preparing it for a web-based application.
*   **Final Thoughts:** This project provides a solid foundation in building and deploying end-to-end deep learning systems, tackling a challenging computer vision problem with practical applications.

---

## Slide 16: Thank You & Q&A

*   **Thank you for your attention.**
*   **Any Questions?**

