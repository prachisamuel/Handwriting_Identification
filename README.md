# Handwriting Identification

This project focuses on text-independent handwriting identification using deep learning. We use a modified **ResNet-18** architecture fine-tuned on the **IAM Handwritten Forms Dataset** to identify writers based on their handwriting. The model is trained and evaluated using the dataset, with performance visualized through accuracy plots.

---

## Features

- **Text-independent Writer Identification**: Identifies the writer of a handwritten text.
- **Deep Learning Model**: Fine-tuned ResNet-18 architecture.
- **Dataset**: The project uses the IAM Handwritten Forms Dataset for training and evaluation.
- **Accuracy**: The model achieves an accuracy of 97.64% during training and 91.50% during testing.

---

## Technologies Used
- **Deep Learning Framework**: PyTorch
- **Model Architecture**: ResNet-18
- **Data Preprocessing**: Torchvision transforms
- **Optimizer**: Adam optimizer for efficient training
- **Visualization**: Matplotlib for plotting training/validation results

---

## Project Overview

The project is divided into several stages:
1. **Data Preprocessing**: The dataset is preprocessed by resizing images, normalizing them, and splitting them into training, validation, and test sets.
2. **Model Training**: A ResNet-18 model is fine-tuned for writer identification.
3. **Evaluation**: The model is evaluated using test and validation sets, and performance metrics such as accuracy are tracked.
4. **Visualization**: The training and validation accuracy/loss curves are plotted, and some correct predictions are visualized.

---

## Dataset

The IAM Handwritten Forms Dataset is a publicly available dataset consisting of handwritten text images.

The dataset is not included directly in this repository due to size limitations. However, you can download the dataset from Kaggle by following these steps:

1. Go to [IAM Handwritten Forms Dataset on Kaggle](https://www.kaggle.com/naderabdalghani/iam-handwritten-forms-dataset).
2. Download and unzip the dataset.

---

## Model Architecture

We use the **ResNet-18** architecture:

- **Input**: Handwritten text images (128x128 grayscale).
- **Feature Extraction**: Convolutional layers extract features.
- **Classification**: Fully connected layers classify features into writer classes.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/prachisamuel/Handwriting_Identification.git
cd Handwriting_Identification
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure that you have access to the dataset as described above.

---

## Usage

### Run the Notebook
Open the `handwriting-identification.ipynb` notebook and run each cell to train, validate, and evaluate the model.

---

## Results

- **Training Accuracy**: 97.64%
- **Validation Accuracy**: 46.10%
- **Test Accuracy**: 91.50%

---

## Future Improvements

- Experiment with advanced architectures like **ResNet-50** or **EfficientNet**.
- Apply **data augmentation** techniques for better generalization.
- Fine-tune **hyperparameters** to improve validation accuracy further.
- Incorporate **feature fusion techniques** to enhance the model's ability to identify diverse handwriting styles.

---

## Acknowledgements

- **ResNet-18**: Pre-trained model used for fine-tuning in this project.
- **IAM Handwritten Forms Dataset**: Used for training and testing the handwriting identification model.
- **PyTorch**: Framework used to implement the model and training pipeline.
