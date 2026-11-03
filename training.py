import numpy as np
import pygame as pg

class Point:
    def __init__(self, xRange, yRange) -> None:
        
        self.x = np.random.randint(xRange)
        self.y = np.random.randint(yRange)
        self.label = -1
        
        if self.x > self.y:
            self.label = 1
        
    def show(self, canvas, fill_color, outline_color):
        pg.draw.circle(canvas, fill_color, (self.x, self.y), 10)
        pg.draw.circle(canvas, outline_color, (self.x, self.y), 10, 5)
    