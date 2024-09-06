import numpy as np

class Sigmoid:
  @staticmethod
  def g(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
  
  @staticmethod
  def g_prime(x: np.ndarray) -> np.ndarray:
    return Sigmoid.g(x) * (1 - Sigmoid.g(x))

class Tanh:
  @staticmethod
  def g(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)
  
  @staticmethod
  def g_prime(x: np.ndarray) -> np.ndarray:
    return 1 - np.power(Tanh.g(x), 2)

class ReLu:
  @staticmethod
  def g(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)
  
  @staticmethod
  def g_prime(x: np.ndarray) -> np.ndarray:
    return 0 if x <= 0 else 1

class BinaryCrossEntropy:
  @staticmethod
  def loss(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = y_hat.reshape(-1).shape[0]
    return -1 * (1/m) * np.sum( y*np.log(y_hat) + (1-y)*(np.log(1-y_hat)) )
  
  @staticmethod
  def loss_prime(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = y_hat.reshape(-1).shape[0]
    return -1 * (1/m) * (y/y_hat - (1-y)/(1-y_hat))

class MeanSquaredError:
  @staticmethod
  def loss(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.mean(np.sum( (y_hat - y)**2 , axis=1))
  
  @staticmethod
  def loss_prime(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = y_hat.reshape(-1).shape[0]
    return (1/m) * 2 * (y_hat - y)

from typing import List, Dict

class NeuralNetTrainer:
  """Class attributes and methods

    **Part 1: creating the network**

    - `self.X` : an n^0 x m matrix representing all the training data
    - `self.y` : an n^0 x m matrix representing all the training data
  """
  def __init__(self, X: np.ndarray, y: np.ndarray, layers : List[int], alpha = 0.1) -> None:
    """
    X = n^0 x m matrix
    y = n^L x m matrix
    layers = list of hidden layer sizes and output layer size
    alpha = learning rate
    """
    self.X = X
    self.y = y
    self.alpha = alpha
    self.m = self.X.shape[1]
    self.initialization_factor = 0.1

    # custom activations and cost
    self.g = Tanh.g
    self.g_prime = Tanh.g_prime
    self.cost = MeanSquaredError.loss
    self.cost_prime = MeanSquaredError.loss_prime

    input_size = self.X.shape[0]
    output_size = self.y.shape[0]
    assert layers[-1] == output_size
    
    self.n : List[int] = [self.X.shape[0], *layers]
    self.L = len(self.n) - 1 # only include hidden layers

    self.W : Dict[int, np.ndarray] = {}
    self.b : Dict[int, np.ndarray] = {}
    for l in range(1, self.L+1):
      self.W[l] = np.random.randn(self.n[l], self.n[l-1]) * self.initialization_factor
      self.b[l] = np.random.randn(self.n[l], 1)
    
  
  def output_shapes(self):
    print("layer shapes:", self.n)
    print("number layers:", self.L)
    print("number training samples:", self.m)
    print("_"*40, "\n")
    for layer in self.W:
      print(f"weights for layer {layer}",self.W[layer].shape)
    print("_"*40, "\n")
    for layer in self.b:
      print(f"bias for layer {layer}",self.b[layer].shape)
  
  """
  X must be an n x m matrix
  """
  def forward(self, X):
    m = X.shape[1]
    self.A : Dict[int, np.ndarray] = {}
    self.Z : Dict[int, np.ndarray] = {}
    # 1. do calculations for first layer
    self.Z[1] = self.W[1] @ X + self.b[1]
    self.A[1] = self.g(self.Z[1])

    for l in range(2, self.L):
      # 2. calculate Zl = Wl * Al-1 + bl
      self.Z[l] = self.W[l] @ self.A[l-1] + self.b[l]
      assert self.Z[l].shape == (self.n[l], m), f"shape mismatch: {self.Z[l].shape}"

      # 3. calculate Al = g(Zl)
      self.A[l] = self.g(self.Z[l])
    
    # 4. calculate output for last layer using sigmoid
    L = self.L
    self.Z[L] = self.W[L] @ self.A[L-1] + self.b[L]
    self.A[L] = Sigmoid.g(self.Z[L])

    self.A_shapes = [self.A[key].shape for key in self.A.keys()]
    self.Z_shapes = [self.Z[key].shape for key in self.Z.keys()]

    # to prevent indexing issues
    self.A[0] = X

    return self.A[L]
  
  def __assert_two_dimensions(arr: np.ndarray):
    assert arr.ndim == 2, f"expected 2 dimensions, got {arr.ndim}"
  
  # dC/dA^l -> dC/dA^(l-1), dC/dW^l, dC/db^l
  def __propagate_derivatives(self, dC_dAl: np.ndarray, layer_index: int):
    A_l = self.A[layer_index]
    Z_l = self.Z[layer_index]
    m = A_l.shape[1]

    """
    # propagation calculations
    # dC/dA^(l-1) = dC/dA^l * dA^l/dZ^l * dZ^l/dA^(l-1)
    # [n^(l-1) x m ] = ([n^l x m] * [n^l x m]) @ [n^(l-1) x n^l]
    # [n^(l-1) x m ] = [n^(l-1) x n^l]^T @ [n^l x m] 


    # dC/dA^(l-1) = (dZ^l/dA^(l-1))^T @ (dC/dA^l * dA^l/dZ^l)
    """
    dAl_dZl = self.g_prime(Z_l)
    dC_dZl = dC_dAl * dAl_dZl
    dC_dA_l_minus_one = self.W[layer_index].T @ dC_dZl

    # weight and bias calculations for layer
    dZl_dWl = self.A[layer_index - 1]
    dZL_dbl = 1
    dC_dWL = dC_dZl @ dZl_dWl.T
    dC_dbL = np.sum(dC_dZl, axis=1, keepdims=True)
    assert dC_dWL.shape == self.W[layer_index].shape
    assert dC_dbL.shape == self.b[layer_index].shape
    assert dC_dA_l_minus_one.shape == (self.n[layer_index-1], m)
    return {
        "gradW": dC_dWL,
        "gradb": dC_dbL,
        "gradA-minus-one": dC_dA_l_minus_one
    }
  
  def backward(self, y_hat: np.ndarray, y: np.ndarray):
    NeuralNetTrainer.__assert_two_dimensions(y_hat)
    NeuralNetTrainer.__assert_two_dimensions(y)

    A_L = y_hat.reshape(self.A_shapes[-1])
    pred = y.reshape(self.A_shapes[-1])
    m = y_hat.shape[1]

    """
    First layer calculations
    """
    
    dC_dAL = self.cost_prime(A_L, pred)

    dAL_dZL = Sigmoid.g_prime(self.Z[self.L])
    assert dAL_dZL.shape == (self.A_shapes[-1])

    dC_dZL = dC_dAL * dAL_dZL
    assert dC_dZL.shape == (self.A_shapes[-1])

    dZL_dWL = self.A[self.L-1]
    dZL_dbL = 1

    dC_dWL = dC_dZL @ dZL_dWL.T
    dC_dbL = np.sum(dC_dZL, axis=1, keepdims=True)
    assert dC_dWL.shape == self.W[self.L].shape
    assert dC_dbL.shape == self.b[self.L].shape

    gradW : Dict[int, np.ndarray] = {}
    gradb : Dict[int, np.ndarray] = {}

    gradW[self.L] = dC_dWL
    gradb[self.L] = dC_dbL

    

    """
    loop through rest of layers

    dC/dZl-1 = dC/dZl * dZl/dAl-1 * dAl-1/dZl-1
    dC/dWl-1 = C/dZl-1 @ (A^l-1)^T
    dC/dbl-1 = C/dZl-1
    """
    # starting from last layer
    dZL_dA_L_minus_one = self.W[self.L]
    dC_dA_L_minus_one = dZL_dA_L_minus_one.T @ dC_dZL
    assert dC_dA_L_minus_one.shape == (self.A_shapes[-2])

    # temp for looping
    dC_dA_l_minus_one = dC_dA_L_minus_one
    for l in range(self.L-1, 0, -1):
      propagated_grads = self.__propagate_derivatives(dC_dA_l_minus_one, l)
      gradW[l] = propagated_grads["gradW"]
      gradb[l] = propagated_grads["gradb"]
      dC_dA_l_minus_one = propagated_grads["gradA-minus-one"]
    
    return {
        "gradW": gradW,
        "gradb": gradb
    }
  
  def train(self, epochs=1000):
    costs = []
    for epoch in range(1, epochs+1):
      # 1. feed forward
      y_hat = self.forward(self.X)

      # 2. cost
      loss = self.cost(y_hat, self.y)
      costs.append(loss)

      # 3. backward prop
      grads = self.backward(y_hat, self.y)
      gradW = grads["gradW"]
      gradb = grads["gradb"]
      for l in range(1, self.L + 1):
        self.W[l] += -self.alpha * gradW[l]
        self.b[l] += -self.alpha * gradb[l]
      
    return costs, epochs