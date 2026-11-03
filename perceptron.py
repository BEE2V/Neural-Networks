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

    def show(self, screen) -> None:
        if self.NofInputs != 2:
            raise ValueError("Decision-boundary drawing requires exactly 2 inputs")

        weight_x, weight_y, bias_weight = self.weights
        if weight_x == 0 and weight_y == 0:
            return

        width, height = screen.get_size()
        intersections = []

        if weight_y != 0:
            for x in (-1.0, 1.0):
                y = -(weight_x * x + bias_weight * self.bias) / weight_y
                if -1.0 <= y <= 1.0:
                    intersections.append((x, y))

        if weight_x != 0:
            for y in (-1.0, 1.0):
                x = -(weight_y * y + bias_weight * self.bias) / weight_x
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
            pg.draw.line(screen, "#16C639", pixel_points[0], pixel_points[1], 2)