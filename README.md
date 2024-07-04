# MNIST Handwritten Digit Classification

This project demonstrates how to build a simple deep learning model using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The MNIST dataset is a classic dataset used in the field of machine learning and contains images of handwritten digits from 0 to 9.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In this project, we will:
- Load and preprocess the MNIST dataset.
- Build a neural network using TensorFlow and Keras.
- Train the model on the training data.
- Evaluate the model on the test data.

## Dataset

The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits. Each image is 28x28 pixels.

## Project Structure

MNIST-Classification/
├── MNIST.ipynb # Jupyter Notebook containing the full project code
└── README.md # This README file


## Getting Started

To run this project, you can either use Google Colab or Jupyter Notebook.

### Using Google Colab

1. Open Google Colab: [Google Colab](https://colab.research.google.com/)
2. Upload the `MNIST.ipynb` file.
3. Run the notebook cells step by step.

### Using Jupyter Notebook

1. Make sure you have Jupyter Notebook installed on your system. If not, you can install it using Anaconda or pip:

    ```bash
    pip install notebook
    ```

2. Clone the repository:

    ```bash
    git clone https://github.com/parmarsunny125/MNIST-Classification.git
    cd MNIST-Classification
    ```

3. Open the Jupyter Notebook:

    ```bash
    jupyter notebook MNIST.ipynb
    ```

4. Follow the instructions in the notebook to run the code cells step by step.

## Usage

The notebook contains all the necessary code to load, preprocess, build, train, and evaluate the model. Simply follow the cells in the notebook.

## Model Architecture

The model is a simple neural network with the following layers:
- Flatten layer: Converts the 28x28 images to a 1D array.
- Dense layer: 100 neurons with ReLU activation.
- Dense layer: 100 neurons with ReLU activation.
- Output layer: 10 neurons with sigmoid activation (one for each digit).

## Training the Model

The model is trained using the Adam optimizer and the sparse categorical crossentropy loss function. We train the model for 5 epochs.

## Evaluating the Model

After training, the model is evaluated on the test data to determine its accuracy.

## Results

The trained model achieves an accuracy of over 97% on the test dataset. This means the model correctly classifies over 97% of the handwritten digits.

## Contributing

If you have any suggestions or improvements, feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
