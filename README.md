# Emotion Classification with Deep Learning

This project demonstrates emotion classification from text data using a deep learning model built with TensorFlow. The goal is to classify emotions in text samples based on a set of predefined emotion labels. The model is trained using a dataset consisting of labeled text and evaluates its performance on a test set.

## Requirements

To run the project, you'll need the following libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `tensorflow`
- `scikit-learn`

## Dataset
The project uses three text files:

train.txt: Contains the training data.

val.txt: Contains the validation data.

test.txt: Contains the test data.

Each text file consists of two columns:

Text: The text sample to be classified.

Emotions: The emotion label associated with the text sample.

The text samples and their corresponding emotions are separated by a semicolon (;).

## Script Overview
### Data Preprocessing:

Loads the training, validation, and test datasets.

Combines the training and validation datasets for model training.

Tokenizes and pads the text data.

Encodes the emotion labels into one-hot vectors.

### Model Definition:

Defines a simple Sequential model using TensorFlow Keras.

The model consists of:

Embedding layer for word representation.

Flatten layer for dimensionality reduction.

Dense layers for classification.

### Training:

The model is trained using the combined dataset (train + validation) for 10 epochs with a batch size of 32.

### Evaluation:

The model's performance is evaluated on the test dataset.

Outputs the test loss and accuracy.

## Usage
To train the emotion classification model, simply run the script emotion_classifier.py. The script will:

Load and preprocess the data.

Train the model using the training and validation data.

Evaluate the model on the test data and print the test accuracy and loss.
