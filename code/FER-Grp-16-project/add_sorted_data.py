
import os
import cv2
import shutil

# --- CONFIGURATION ---
# 1. Path to your new dataset, which is already sorted into emotion folders
NEW_DATA_SOURCE_PATH = 'CK_plus_dataset'

# 2. Path to your project's training data folder
# This is where the new images will be added
PROJECT_TRAIN_PATH = 'FER 2013/train'

# 3. Image size for your model
IMG_SIZE = 48


def add_sorted_data():
    """
    Processes images from pre-sorted emotion folders and adds them to the project's
    training set.
    """
    if not os.path.isdir(NEW_DATA_SOURCE_PATH):
        print(f"Error: Source folder '{NEW_DATA_SOURCE_PATH}' not found.")
        print("Please make sure the folder with your new sorted data exists.")
        return

    added_files_count = 0
    # Get the list of emotion folders in the source directory (e.g., 'anger', 'happy')
    emotion_folders = [d for d in os.listdir(NEW_DATA_SOURCE_PATH) if os.path.isdir(os.path.join(NEW_DATA_SOURCE_PATH, d))]

    # Dictionary to map source folder names to destination folder names
    NAME_MAP = {
        "anger": "angry",
        "sadness": "sad"
    }

    for emotion_name in emotion_folders:
        source_emotion_folder = os.path.join(NEW_DATA_SOURCE_PATH, emotion_name)
        
        # Map the name to the correct destination name (e.g., 'sadness' -> 'sad')
        dest_name = NAME_MAP.get(emotion_name, emotion_name)
        dest_emotion_folder = os.path.join(PROJECT_TRAIN_PATH, dest_name)

        # Check if the corresponding destination folder exists in your training set
        if not os.path.isdir(dest_emotion_folder):
            print(f"Warning: Destination folder for '{emotion_name}' not found in '{PROJECT_TRAIN_PATH}'. Skipping this emotion.")
            continue

        print(f"Processing folder: '{emotion_name}'...")
        image_files = os.listdir(source_emotion_folder)

        for i, filename in enumerate(image_files):
            try:
                source_image_path = os.path.join(source_emotion_folder, filename)

                # --- Pre-processing ---
                # 1. Read the image
                img = cv2.imread(source_image_path)
                if img is None:
                    continue
                # 2. Convert to grayscale
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 3. Resize to model's input size
                resized_img = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))

                # --- Save to project folder ---
                # Create a unique filename to avoid conflicts
                new_filename = f"newset_{emotion_name}_{i}.png"
                output_path = os.path.join(dest_emotion_folder, new_filename)
                
                cv2.imwrite(output_path, resized_img)
                added_files_count += 1

            except Exception as e:
                # print(f"Could not process file {filename}. Error: {e}")
                continue

    print("-" * 50)
    print("Data Addition Complete!")
    print(f"Successfully added {added_files_count} new images to your training set.")
    print("You can now re-run your 'TrainingModelFER.ipynb' notebook.")
    print("-" * 50)

if __name__ == '__main__':
    add_sorted_data()
