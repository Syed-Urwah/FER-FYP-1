from tensorflow.keras.models import load_model
from time import sleep, time
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os


def main():
    try:
        # Initialize face classifier and emotion model
        face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
        classifier = load_model(r'FER_Model_Grp16.h5')

        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

        # Initialize statistics
        emotion_stats = {emotion: 0 for emotion in emotion_labels}
        total_frames = 0
        frames_with_faces = 0
        start_time = time()
        session_log = []

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Facial Expression Recognition System started. Press 'q' to quit.")

        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from webcam.")
                    break

                total_frames += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray)
                current_emotions = []

                for (x, y, w, h) in faces:
                    frames_with_faces += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                    if np.sum([roi_gray]) != 0:
                        roi = roi_gray.astype('float') / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)

                        prediction = classifier.predict(roi, verbose=0)[0]
                        label = emotion_labels[prediction.argmax()]
                        emotion_stats[label] += 1
                        current_emotions.append(label)

                        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Log current frame
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                session_log.append({
                    'timestamp': timestamp,
                    'faces_detected': len(faces),
                    'emotions_detected': ', '.join(current_emotions) if current_emotions else 'None'
                })

                cv2.imshow('Emotion Detector', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as frame_error:
                print(f"Error processing frame: {frame_error}")
                continue

    except Exception as main_error:
        print(f"Initialization error: {main_error}")
    finally:
        # Ensure resources are released
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

        # Generate reports if we processed any frames
        if total_frames > 0:
            generate_csv_report(session_log)
            generate_statistics_report(emotion_stats, total_frames, frames_with_faces, start_time)
            generate_formatted_report(emotion_stats, total_frames, frames_with_faces, start_time)
        else:
            print("No frames processed - reports not generated")


def generate_csv_report(session_log):
    """Generate detailed CSV log of all frames"""
    try:
        os.makedirs('reports', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"reports/detailed_log_{timestamp}.csv"

        df = pd.DataFrame(session_log)
        df.to_csv(filename, index=False)
        print(f"\nDetailed frame log saved to: {filename}")
    except Exception as e:
        print(f"Error generating CSV report: {e}")


def generate_statistics_report(emotion_stats, total_frames, frames_with_faces, start_time):
    """Generate CSV with statistics"""
    try:
        os.makedirs('reports', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"reports/statistics_{timestamp}.csv"

        duration = time() - start_time
        stats_data = {
            'Metric': ['Total Frames', 'Frames with Faces', 'Session Duration (seconds)'],
            'Value': [total_frames, frames_with_faces, duration]
        }

        # Add emotion statistics
        for emotion, count in emotion_stats.items():
            stats_data['Metric'].append(f"{emotion} Detections")
            stats_data['Value'].append(count)
            percentage = count / frames_with_faces * 100 if frames_with_faces > 0 else 0
            stats_data['Metric'].append(f"{emotion} Percentage")
            stats_data['Value'].append(f"{percentage:.1f}%")

        df = pd.DataFrame(stats_data)
        df.to_csv(filename, index=False)
        print(f"Statistics report saved to: {filename}")
    except Exception as e:
        print(f"Error generating statistics report: {e}")


def generate_formatted_report(emotion_stats, total_frames, frames_with_faces, start_time):
    """Generate nicely formatted text report"""
    try:
        os.makedirs('reports', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"reports/summary_report_{timestamp}.txt"

        duration = time() - start_time

        with open(filename, 'w') as f:
            # Header
            f.write("=" * 60 + "\n")
            f.write("FACIAL EXPRESSION RECOGNITION SESSION REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Session info
            f.write(f"Session Duration: {duration:.2f} seconds\n")
            f.write(f"Total Frames Processed: {total_frames}\n")
            f.write(
                f"Frames with Faces Detected: {frames_with_faces} ({frames_with_faces / total_frames * 100:.1f}%)\n\n")

            # Emotion statistics
            f.write("Emotion Detection Statistics:\n")
            f.write("-" * 50 + "\n")
            for emotion, count in emotion_stats.items():
                percentage = count / frames_with_faces * 100 if frames_with_faces > 0 else 0
                f.write(f"{emotion:<10}: {count:>4} detections ({percentage:.1f}%)\n")

            # Most common emotion
            if frames_with_faces > 0:
                most_common = max(emotion_stats.items(), key=lambda x: x[1])
                f.write(f"\nMost common emotion: {most_common[0]} ({most_common[1]} detections)\n")

            # Footer
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")

        print(f"Formatted text report saved to: {filename}")

        # Also print to console
        with open(filename, 'r') as f:
            print("\n" + f.read())

    except Exception as e:
        print(f"Error generating formatted report: {e}")


if __name__ == "__main__":
    main()