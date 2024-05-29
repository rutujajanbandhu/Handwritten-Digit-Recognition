# HandWritten Digit Recognition
The goal is to develop a deep learning model capable of accurately classifying handwritten
digits from the MNIST dataset, which contains grayscale images of handwritten digits ranging from 0
to 9. The challenge lies in training a model that can effectively capture the subtle variations in writing
styles across different individuals while maintaining robustness to noise and distortions in the images.

## Dataset
The MNIST dataset is a widely used benchmark in the field of machine learning, particularly for
image classification tasks. It consists of 28x28 grayscale images of handwritten digits (0-9) collected
from various sources. 
* The dataset is divided into training and test sets, with 60,000 images for training and 10,000 images for testing.
  
![App Screenshot](https://github.com/rutujajanbandhu/Handwritten-Digit-Recognition/blob/main/Screenshots/Data.jpg)

For the handwritten digit recognition task, I've implemented several traditional
and deep learning approaches and compared their performances. Here's an
overview of my solution strategy and the comparisons made:
## Traditional Approaches:

* Convolutional Neural Network (CNN): Utilized CNN architecture with appropriate convolutional layers, activation functions (e.g., ReLU), pooling layers (e.g., max pooling), and fully connected layers. Used suitable loss functions like cross-entropy and optimized the model using algorithms like stochastic gradient descent (SGD) or Adam.
* Multilayer Perceptron (MLP): Implemented an MLP architecture with multiple layers, activation functions (e.g., ReLU, sigmoid), and appropriate loss functions and optimizers for training, such as categorical cross-entropy and SGD.

## Comparison Metrics:

* Evaluated the models based on their training and test losses (e g., mean squared error, categorical cross-entropy) to assess their convergence and generalization capabilities.
* Assessed model accuracies on both training and test sets to measure their performance in correctly classifying handwritten digits.

Accuracy Comparison:
![App Screenshot](https://github.com/rutujajanbandhu/Handwritten-Digit-Recognition/blob/main/Screenshots/Traditional_Approach_Comparision.jpg)

## Deep Learning Architectures:

* LeNet-5: Implemented the LeNet-5 architecture, which comprises convolutional layers followed by max-pooling and fully connected layers, designed specifically for handwritten digit recognition tasks.
* ResNet (Residual Network): Utilized ResNet architecture with residual connections, enabling training of very deep networks while mitigating the vanishing gradient problem.
* DenseNet (Densely Connected Convolutional Network): Implemented DenseNet architecture with densely connected layers, facilitating feature reuse and gradient flow throughout the network.

## Comparison of CNN Models:

* Evaluated and compared the performances of LeNet-5, ResNet, and DenseNet based on their training and test losses to analyze their convergence rates and generalization capabilities.
* Assessed the accuracies of these models on both training and test datasets to determine their effectiveness in classifying handwritten digits accurately.

Accuracy Comparison:
![App Screenshot](https://github.com/rutujajanbandhu/Handwritten-Digit-Recognition/blob/main/Screenshots/Training_accuracy.jpg)

![App Screenshot](https://github.com/rutujajanbandhu/Handwritten-Digit-Recognition/blob/main/Screenshots/Testing_accuracy.jpg)


## GNN Visualization
* Connectivity Visualization: Visualizing the connectivity of pixels forming a digit
using the GNN model can provide qualitative insights into how the model processes
and understands image data.

![App Screenshot](kjj)
