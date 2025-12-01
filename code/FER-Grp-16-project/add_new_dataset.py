
import os
import cv2
import shutil

# --- CONFIGURATION ---
# 1. Path to the extracted CK+ dataset folders
# This should contain 'cohn-kanade-images' and 'Emotion' folders
CK_PLUS_BASE_PATH = 'CK_plus_dataset'

# 2. Path to your project's training data folder
# This is where the new images will be added
PROJECT_TRAIN_PATH = 'FER 2013/train'

# 3. Image size for your model
IMG_SIZE = 48

# --- EMOTION MAPPING ---
# Maps the emotion codes from CK+ to your project's folder names
# 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
EMOTION_MAP = {
    '1': 'angry',
    '3': 'disgust',
    '4': 'fear',
    '5': 'happy',
    '6': 'sad',
    '7': 'surprise'
    # We are ignoring 'neutral' (0) and 'contempt' (2) as they may not align perfectly
    # or have fewer samples. You can add them if you wish.
}


def preprocess_and_add_data():
    """
    Finds labeled images in the CK+ dataset, processes them, and adds them
    to the project's training folders.
    """
    emotion_label_path = os.path.join(CK_PLUS_BASE_PATH, 'Emotion')
    image_path = os.path.join(CK_PLUS_BASE_PATH, 'cohn-kanade-images')

    if not os.path.isdir(emotion_label_path) or not os.path.isdir(image_path):
        print(f"Error: Could not find 'Emotion' or 'cohn-kanade-images' folders in '{CK_PLUS_BASE_PATH}'")
        print("Please make sure you have extracted the CK+ dataset correctly.")
        return

    added_files_count = 0
    # Walk through the emotion labels directory
    for subdir, _, files in os.walk(emotion_label_path):
        for file in files:
            if file.endswith('.txt'):
                try:
                    # Read the emotion label from the text file
                    with open(os.path.join(subdir, file), 'r') as f:
                        emotion_code = f.read().strip().split('.')[0]

                    # Check if this emotion is one we want to use
                    if emotion_code in EMOTION_MAP:
                        emotion_name = EMOTION_MAP[emotion_code]

                        # Construct the path to the corresponding image sequence folder
                        # e.g., S005/001
                        parts = subdir.split(os.sep)
                        subject = parts[-2]
                        session = parts[-1]
                        image_session_path = os.path.join(image_path, subject, session)

                        # Find the last image in that sequence (the peak expression)
                        image_files = sorted([f for f in os.listdir(image_session_path) if f.endswith('.png')])
                        if not image_files:
                            continue
                        
                        peak_image_path = os.path.join(image_session_path, image_files[-1])

                        # --- Pre-processing ---
                        # 1. Read the image
                        img = cv2.imread(peak_image_path)
                        if img is None:
                            continue
                        # 2. Convert to grayscale
                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # 3. Resize to model's input size
                        resized_img = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))

                        # --- Save to project folder ---
                        output_dir = os.path.join(PROJECT_TRAIN_PATH, emotion_name)
                        if not os.path.isdir(output_dir):
                            print(f"Warning: Emotion folder '{emotion_name}' not found. Skipping.")
                            continue
                        
                        # Create a unique filename
                        new_filename = f"ckplus_{subject}_{session}.png"
                        output_path = os.path.join(output_dir, new_filename)
                        
                        cv2.imwrite(output_path, resized_img)
                        added_files_count += 1

                except Exception as e:
                    # print(f"Could not process file {file}. Error: {e}")
                    continue

    print("-" * 50)
    print("Data Addition Complete!")
    print(f"Successfully added {added_files_count} new images to your training set.")
    print("You can now re-run your 'TrainingModelFER.ipynb' notebook.")
    print("-" * 50)

if __name__ == '__main__':
    preprocess_and_add_data()
