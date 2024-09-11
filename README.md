
# Deepfake Image Detection

## Overview
This project implements a deep learning model for detecting deepfake images. The model is trained on a dataset of real and fake images to classify whether an input image is real or fake.

## Requirements
- Python 3.x
- TensorFlow 2.x
- scikit-learn
- NumPy
- Matplotlib
- OpenCV
- Gradio

##Dataset Link -> https://drive.google.com/drive/folders/1Cjc83ExrfafkNEjK2MmdJPYmg8Irbj3a?usp=drive_link

## Tech Stack
- **Deep Learning Framework:** TensorFlow
- **Model Architecture:** Convolutional Neural Network (CNN)
- **Data Augmentation:** ImageDataGenerator
- **Optimization:** Adam optimizer
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy
- **Regularization:** Dropout, L2 Regularization
- **Image Preprocessing:** Rescaling, Data Augmentation

## Project Structure
- `README.md`: Overview of the project, requirements, and tech stack.
- `deepfake_detection.py`: Main script for training and evaluating the deepfake detection model.
- `load_and_preprocess_image.py`: Utility functions for loading and preprocessing images.
- `model.py`: Definition of the Meso4 deepfake detection model.
- `data/`: Directory containing the dataset of real and fake images.
- `weights/`: Directory to save the trained model weights.

## Usage
1. Clone the repository: `git clone https://github.com/your-username/deepfake-image-detection.git`
2. Install the required packages: `pip install -r requirements.txt`
3. Run the `deepfake_detection.py` script to train and evaluate the model.

## Future Improvements
- Experiment with different model architectures and hyperparameters.
- Explore advanced data augmentation techniques.
- Implement more sophisticated deepfake detection algorithms.
- This project is created for educational perposes only.
## Contributors
- [Your Name](https://github.com/your-username)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
