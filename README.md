# Traffic Sign Recognition Classifier
The project developed a Traffic Sign Classification model using neural network architectures, including AlexNet, DenseNet and CNN. The objective was accurate traffic sign classification based on visual features. The model accuracies ranged from 96% to 98% during training, indicating their ability to learn and classify the various traffic sign classes accurately. Furthermore, the test accuracy results ranged between 86% to 97%, demonstrating the models' generalization performance on unseen data.Different architectures demonstrated strengths and weaknesses. This study highlights the effectiveness of diverse neural network models for traffic sign classification, contributing to intelligent transportation systems and enhanced road safety.

## Dataset Description:
This dataset consists of images of traffic signs captured from roads, with each image labeled according to its corresponding class. The dataset focuses on single-image, multi-class classification challenges and does not include any temporal information from the original video footage.

## Dataset Description:
This dataset consists of images of traffic signs captured from roads, with each image labeled according to its corresponding class. The dataset focuses on single-image, multi-class classification challenges and does not include any temporal information from the original video footage.

## Dataset
https://www.kaggle.com/datasets/harbhajansingh21/german-traffic-sign-dataset?select=train.p

## Data Description:
Each image in the dataset is sized at 32 x 32 pixels and is presented in RGB format, featuring three color channels. The pixel values are stored as unsigned 8-bit integers, offering a range of 256 potential values for each pixel.

The dataset encompasses a total of 43 unique classes or labels, determined by the design or significance of the traffic signs.

The training set includes 34,799 images, each linked to its corresponding label.

The validation set encompasses 4,410 images, each with its respective labels.

Finally, the test set comprises 12,630 images, with each image labeled according to its corresponding class.

## Model used
### Model Architecture -1 : ALEXNET¶
The code implements AlexNet, a convolutional neural network architecture used for image classification. It utilizes Keras to construct the model, which consists of convolutional, fully connected, and activation layers. The network is designed to classify images into 43 different classes.

### Model Architecture -2 : DenseNET
Changes as compared to Architecture-1:

Hyperparameter Modification: The second model incorporates changes in the hyperparameters. It includes early stopping and model checkpointing callbacks, which help in optimizing the training process by monitoring the validation loss and saving the best model.

### Model Architecture - 3: CNN
The below model is a convolutional neural network (CNN) designed for image classification. It consists of multiple convolutional layers with ReLU activation, max pooling layers, and dropout layers for regularization. The final architecture includes three sets of convolutional layers, followed by max pooling and dropout, and ends with a fully connected layer and an output layer with softmax activation. The model is capable of processing grayscale images of size 32x32x1 and has a total of 43 output units, representing the number of classes in the classification task. The summary provides a concise overview of the model's structure and the number of parameters in each layer.

## Accuracy of different models are:
AlexNet: 88.84%

DenseNet: 95.12%

CNN Model: 97.31%
