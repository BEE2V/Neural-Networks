import numpy as np

# Activation Function
def sign(number): 
    if number >= 0:
        return 1
    else:
        return -1

class Perceptron:
    lr = 0.1                # Learning Rate
    
    def __init__(self, NofConnections: int) -> None:
        self.weights = []
        self.NofConnections = NofConnections
        for i in range(NofConnections):
            self.weights.append(np.random.choice([-1.0, 1.0]))
    
    def guess(self, inputs: list[float]) -> int:
        sum = 0
        for i in range(self.NofConnections):
            sum += inputs[i] * self.weights[i]
            
        return sign(sum)
        
    def train(self, inputs: list[float], target: int) -> None:
      
      guess = self.guess(inputs)
      error = target - guess
      
      # TUne all the weights
      for i in range(self.NofConnections):
          self.weights[i] += error * inputs[i] * Perceptron.lr     
