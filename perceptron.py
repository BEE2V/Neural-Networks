import numpy as np
import pygame as pg

# Activation Function
def sign(number): 
    if number >= 0:
        return 1
    else:
        return -1

class Perceptron:
    lr = 0.1                # Learning Rate
    
    def __init__(self, NofConnections: int) -> None:
        self.weights: list[float] = [] 
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

    def show(self, screen) -> None:
        weight_x, weight_y = self.weights
        if weight_x == 0 and weight_y == 0:
            return

        width, height = screen.get_size()
        intersections = []

        if weight_y != 0:
            for x in (-1.0, 1.0):
                y = -weight_x * x / weight_y
                if -1.0 <= y <= 1.0:
                    intersections.append((x, y))

        if weight_x != 0:
            for y in (-1.0, 1.0):
                x = -weight_y * y / weight_x
                if -1.0 <= x <= 1.0:
                    intersections.append((x, y))

        intersections = list(dict.fromkeys(intersections))
        if len(intersections) >= 2:
            pixel_points = [
                (
                    round((x + 1.0) * (width - 1) / 2),
                    round((1.0 - y) * (height - 1) / 2),
                )
                for x, y in intersections[:2]
            ]
            pg.draw.line(screen, "#FFFFFF", pixel_points[0], pixel_points[1], 2)