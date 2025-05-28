# Potato Leaf Disease Identification using Deep Learning

This repository contains a Python application for detecting potato leaf diseases using a pre-trained deep learning model. The application uses PyQt5 for the GUI and TensorFlow for model inference. 

## Requirements

- Python 3.11.6
- PyQt5
- Pillow
- NumPy
- TensorFlow

**Note: This code is designed to run without errors only in Python 3.11.6 version.**

## Installation

1. Install the required packages:
    ```bash
    pip install PyQt5 Pillow numpy tensorflow
    ```
2. Download the pre-trained model [here](https://drive.google.com/drive/folders/1sWTD2RgQHXCxLnaZD2C6RQqrQDZKbfHC?usp=sharing) and place `potato_leaf_disease_model.h5` in the project directory.

## Usage

1. Ensure that the pre-trained model `potato_leaf_disease_model.h5` is in the project directory.

2. Run the application:
    ```bash
    python app.py
    ```

3. Use the GUI to select an image of a potato leaf and detect the disease.

## Dataset

The dataset used for training the model can be downloaded from [here](https://kaggle.com/datasets/c5eba7bed24ecf791e66c0de929b63fe8ae1af7758847b357ee06b06f873de8c). 

## Pre-trained Models (required)

The pre-trained model used for this application can be downloaded from [this link](https://drive.google.com/uc?id=your_pretrained_model_link).

## Model Training (optional)

Training the model is optional. The application comes with a pre-trained model, but if you wish to train the model yourself, follow these steps:

1. Download and extract the dataset from [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).

2. Update the paths in `model_training.py` to point to your dataset directories.

3. Run the training script:
    ```bash
    python Model Training Code.py
    ```

4. The trained model will be saved as `potato_leaf.h5`.
     

