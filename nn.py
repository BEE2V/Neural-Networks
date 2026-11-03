import numpy as np

class NeuralNetwork:
        
    def __init__(self, input_nodes, hidden_nodes, output_nodes) -> None:
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        self.weights_ih = np.random.uniform(-1.0, 1.0, size=(self.hidden_nodes, self.input_nodes + 1))
        self.weights_ho = np.random.uniform(-1.0, 1.0, size=(self.output_nodes, self.hidden_nodes + 1))
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
        
        
    def feed_forward(self, inputs) -> None:
        inputs_n_bias = np.array(inputs + [1.0])
        
        hidden = self.weights_ih @ inputs_n_bias
        hidden = np.append(hidden, 1.0)
        
        output = self.weights_ho @ hidden
        output = self.sigmoid(output)
        
        return output
        
    def train(self, inputs, answer):
        pass
        
        
nn = NeuralNetwork(2, 2, 1)

inp = [1, 0]
out = nn.feed_forward(inp)

print(out)