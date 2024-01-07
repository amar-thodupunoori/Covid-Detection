# Covid Detection

## Overview

Implemented a Covid-19 detection model using VGG16 architecture and transfer learning. The project involves data preprocessing, model building, and evaluation.

## Project Structure

- `Data/`: Directory containing labeled images of Covid and Normal cases.
- `covid5.h5`: Saved model checkpoint.
- `accuracy.png`: Plot of model accuracy over epochs.

## Code Highlights

- **Data Loading:** Utilized OpenCV and NumPy to load and preprocess X-ray images of Covid and Normal cases.

- **Data Augmentation:** Applied image data augmentation techniques using Keras `ImageDataGenerator` to improve model generalization.

- **Transfer Learning:** Employed the VGG16 pre-trained model as the base for feature extraction, with a custom fully connected head for classification.

- **Model Training:** Trained the model using Stochastic Gradient Descent (SGD) as the optimizer and binary crossentropy as the loss function.

- **Evaluation:** Assessed model performance using accuracy, classification report, and confusion matrix. Achieved 100% accuracy on the test set.

## Results

- **Accuracy Plot:** ![accuracy.png](accuracy.png)

- **Sample Predictions:**
  ![Prediction Samples](sample_predictions.png)

## Usage

1. Clone the repository.
2. Ensure required dependencies are installed (`numpy`, `matplotlib`, `seaborn`, `opencv`, `scikit-learn`, `keras`).
3. Run the provided Jupyter notebook or script to train and evaluate the model.

Feel free to explore the code for more details!
