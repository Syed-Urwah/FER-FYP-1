# convert_model.py
import os
import sys
import subprocess


def check_and_install_packages():
    """Check and install required packages"""
    required_packages = {
        'tensorflow': 'tensorflow==2.7.0',
        'tensorflowjs': 'tensorflowjs',
        'numpy': 'numpy',
        'h5py': 'h5py'
    }

    for package, install_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])


def convert_fer_model():
    """Convert the FER model to TensorFlow.js format"""
    import tensorflow as tf
    from tensorflow import keras
    import tensorflowjs as tfjs

    print("\n" + "=" * 60)
    print("FACIAL EXPRESSION RECOGNITION MODEL CONVERTER")
    print("=" * 60 + "\n")

    # Check if model file exists
    if not os.path.exists('FER_Model_Grp16.h5'):
        print("ERROR: FER_Model_Grp16.h5 not found!")
        print("Please ensure the model file is in the current directory.")
        return False

    try:
        # Load the model
        print("Loading FER_Model_Grp16.h5...")
        model = keras.models.load_model('FER_Model_Grp16.h5')
        print("✓ Model loaded successfully!")

        # Display model information
        print("\nModel Information:")
        print(f"- Input shape: {model.input_shape}")
        print(f"- Output shape: {model.output_shape}")
        print(f"- Total parameters: {model.count_params():,}")

        # Create output directory
        output_dir = '../nodejs/fer_model_tfjs'
        os.makedirs(output_dir, exist_ok=True)

        # Convert to TensorFlow.js format
        print(f"\nConverting to TensorFlow.js format...")
        tfjs.converters.save_keras_model(model, output_dir)

        print(f"✓ Model converted successfully!")
        print(f"✓ Output saved to: {output_dir}")

        # Verify conversion
        model_json_path = os.path.join(output_dir, 'model.json')
        if os.path.exists(model_json_path):
            print(f"✓ model.json created successfully")

            # List all created files
            print("\nCreated files:")
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"  - {file} ({file_size:.2f} KB)")

        return True

    except Exception as e:
        print(f"\nERROR during conversion: {e}")
        return False


def main():
    print("Checking required packages...")
    check_and_install_packages()

    print("\nStarting model conversion...")
    success = convert_fer_model()

    if success:
        print("\n" + "=" * 60)
        print("CONVERSION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Navigate to the nodejs directory")
        print("2. Run: npm install")
        print("3. Run: npm start")
    else:
        print("\nConversion failed. Please check the error messages above.")


if __name__ == "__main__":
    main()