import numpy as np
import pygame as pg
from config import *

# Activation Function
def sign(number): 
    if number >= 0:
        return 1
    else:
        return -1

class Perceptron:
    lr = 0.1                # Learning Rate
    
    def __init__(self, NofInputs: int) -> None:
        self.weights: list[float] = []
        self.bias = 1.0
        self.NofInputs = NofInputs
        
        for i in range(NofInputs + 1):
            self.weights.append(np.random.choice([-1.0, 1.0]))
    
    def _inputs_with_bias(self, inputs: list[float]) -> list[float]:
        if len(inputs) != self.NofInputs:
            raise ValueError(
                f"Expected {self.NofInputs} inputs, got {len(inputs)}"
            )
        return inputs + [self.bias]

    def guess(self, inputs: list[float]) -> int:
        inputs_with_bias = self._inputs_with_bias(inputs)
        weighted_sum = sum(
            value * weight
            for value, weight in zip(inputs_with_bias, self.weights)
        )
        return sign(weighted_sum)
        
    def train(self, inputs: list[float], target: int) -> None:
        inputs_with_bias = self._inputs_with_bias(inputs)
        error = target - self.guess(inputs)

        for i, value in enumerate(inputs_with_bias):
            self.weights[i] += error * value * Perceptron.lr
                
    def guessY(self, x: float) -> float:
        w0, w1, w2 = self.weights
        return -(w2/w1) - (w0/w1) * x

    def toPixels(self, point: tuple[float, float]) -> tuple[int, int]:
        pix_x = round((point[0] + 1.0) * (WIDTH - 1) / 2)
        pix_y = round((1.0 - point[1]) * (HEIGHT - 1) / 2)
        
        return (pix_x, pix_y)
            
    def show(self, screen: pg.Surface) -> None:
        point0 = (-1, self.guessY(-1))
        point1 = ( 1, self.guessY( 1))
        
        pixel_point0 = self.toPixels(point0)
        pixel_point1 = self.toPixels(point1)
        
        pg.draw.line(screen, "#16C639", pixel_point0, pixel_point1, 2)
         
        
        
        