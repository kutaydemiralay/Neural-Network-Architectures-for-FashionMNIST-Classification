# Neural-Network-Architectures-for-FashionMNIST-Classification

This repository contains code and documentation for solving image classification tasks using various neural network architectures.

## Project Overview




This project explores deep learning techniques for  fashion items (FashionMNIST). I implemented and evaluated both fully connected neural networks (FCNs) and a convolutional neural network (CNN), comparing different optimizers, regularization techniques, and initialization strategies.

We are tasked with:

- Determining the architecture of our models.
- Fine-tuning hyperparameters for optimal performance.
- Comparing the performance of different models against each other.

---

##  1. Fully Connected Neural Network (FCN)

- **Architecture**:
  - Hidden Layer 1: 300 neurons (ReLU)
  - Hidden Layer 2: 200 neurons (ReLU)
  - Output Layer: 10 neurons (Softmax)
- **Loss Function**: Cross-Entropy Loss
- **Training Setup**:
  - Epochs: 50
  - Batch Size: 512 (train), 256 (test)
- **Optimizers**:
  - SGD (base)
  - RMSprop
  - Adam

---

##  2. Hyperparameter Tuning

- **Learning Rate Selection**:
  - Tuned for each optimizer
  - Balanced stability vs. accuracy
- **Dropout Regularization**:
  - Used `p = 0.1` to prevent overfitting
- **Weight Initialization**:
  - Random Normal
  - Xavier Normal
  - Kaiming (He) Uniform — **Best test accuracy**
- **Batch Normalization**:
  - Used to improve training stability and reduce sensitivity

---

##  3. Evaluation Metrics & Visualization

- Accuracy and loss tracked on:
  - **Training**
  - **Validation**
  - **Test** datasets
- Monitored:
  - Underfitting / Overfitting via gap in loss curves
  - Standard deviation and curve smoothness
- Compared computation time, loss convergence, and final accuracy

---

##  4. Convolutional Neural Network (CNN - AlexNet Style)

- Implemented CNN using PyTorch
- Features:
  - Convolutional layers with ReLU
  - Max pooling
  - Fully connected output
- **Test Accuracy**: **93.13%** in 10 epochs (FashionMNIST)
- Outperformed FCN in both speed and accuracy

---

## 📊 Summary of Results

| Optimizer   | Final Test Accuracy |
|-------------|---------------------|
| SGD         | 86.03%              |
| RMSprop     | 87.94%              |
| Adam        | 88.41%              |
| **CNN**     | **93.13%**          |

---

## Libraries Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

##  Key Concepts

- Neural Network Architectures
- Optimization Algorithms (SGD, RMSprop, Adam)
- Dropout and Batch Normalization
- Weight Initialization Strategies
- Overfitting vs Underfitting Detection
- Loss & Accuracy Curve Analysis

---

## Documentation

You can find the detailed documentation in the PDF below:

[View the Project Paper](./Neural%20Network%20Architectures%20for%20FashionMNIST%20Classification_KutayDemiralay.pdf)

## Code

You can find the relevant Python codes below:

[View the Project Code](./Neural_Network_Architectures_for_FashionMNIST_Classification_KutayDemiralay.ipynb)


## Results

 Plots of loss over epochs and accuracy through epochs for training, validation, and testing were generated. From these values, the optimal network for classifying the FashionMNIST dataset was chosen across various architectures, optimizers, initialization techniques, regularizations, and normalizations.

The Adam optimizer, with an appropriate learning rate, Kaiming Uniform initialization, and batch normalization proved to be the most effective for our base model with FCN  neural network architecture, even though it exhibited some underfitting characteristics

![Adam Optimizer](./images/Adam.png)


*Figure 1: Plots of loss over epochs and accuracy through epochs Base Model with Adam Optimizer with  Kaiming Uniform initialization, and batch normalization*



But when I used the AlexNet CNN architecture for classifying the same FashionMNIST dataset, I achieved significantly better classification performance. The testing accuracy with AlexNet CNN reached up to 93.14% in only 10 epochs, whereas with FCN, the highest testing accuracy I could achieve was 88.46% in 50 epochs, even with optimal tuning.


![Adam Optimizer](./images/CNNAlexNet.png)


*Figure 2: Plots of loss over epochs and accuracy through epochs with AlexNet CNN architecture*

![Adam Optimizer](./images/AlexNet.png)


*Figure 3:AlexNet CNN architecture explained [2]*
