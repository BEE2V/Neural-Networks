import numpy as np
import pygame as pg

class Point:
    def __init__(self, xRange, yRange) -> None:
        
        self.x = np.random.randint(xRange)
        self.y = np.random.randint(yRange)
        self.label = -1
        
        if self.x > self.y:
            self.label = 1
        
    def show(self, canvas):
        color = (0, 255, 0) if self.label == 1 else (255, 0, 0)
        pg.draw.circle(canvas, color, (self.x, self.y), 4)
    