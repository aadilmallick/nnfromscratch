
import numpy as np
import matplotlib.pyplot as plt
from nn import NeuralNetTrainer


if __name__ == "__main__":

    # Input data for XOR function
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Corresponding labels for XOR function
    y = np.array([[0], [1], [1], [0]])

    """
    When vectorizing across all training examples, use the convention [n^l, m],
    where: 
        n^l = number of nodes in layer l
        m = number of training samples
    
    When creating output and input data, both the features and labels must be in that convention
    """
    X = X.T
    y = y.T

    trainer = NeuralNetTrainer(X, y, layers=[4, 3, 1], alpha=0.01)
    trainer.output_shapes()
    costs, epochs = trainer.train(epochs=1000)
    plt.plot(range(epochs), costs)
    plt.show()