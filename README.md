Brief Description: Gradient Descent and Neural Network Implementation

This project demonstrates a machine learning workflow for predicting insurance purchase likelihood using a custom implementation of gradient descent and a neural network built with TensorFlow/Keras.
Key Features:

    Data Preprocessing:
        Scales input features (age and affordability) to ensure they are on the same scale.
        Splits data into training and test sets.

    Neural Network Model:
        Built with TensorFlow/Keras using a single dense layer with sigmoid activation.
        Trained with binary cross-entropy loss and evaluated for accuracy.

    Custom Gradient Descent:
        Implements a gradient descent algorithm from scratch.
        Optimizes weights (w1, w2) and bias for binary classification.
        Includes custom log_loss and sigmoid functions.

    Custom Neural Network Class:
        Defines a class myNN that trains using custom gradient descent.
        Enables prediction for test data.

    Comparison:
        Results from TensorFlow's model and the custom neural network are compared for consistency.

Output:

    Final weights, biases, and loss values are displayed during training.
    Predictions are generated for the test set using both TensorFlow and the custom model.

This project highlights foundational concepts of neural networks, gradient descent, and binary classification, combining theoretical understanding with practical implementation.

