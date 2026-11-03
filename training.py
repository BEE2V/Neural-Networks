import numpy as np
import pygame as pg
from config import *

class Train:
    # y = mx + c
    m = 0.5
    c = 0.7

    def f(self, x) -> float:
        return Train.m * x + Train.c
    
    def toPixels(self, point: tuple[float, float]) -> tuple[int, int]:
        pix_x = round((point[0] + 1.0) * (WIDTH - 1) / 2)
        pix_y = round((1.0 - point[1]) * (HEIGHT - 1) / 2)
        
        return (pix_x, pix_y)
    
    def show_line(self, canvas) -> None:
        p1 = (-1.0, self.f(-1))
        p2 = ( 1.0, self.f( 1))
        
        p1_ = self.toPixels(p1)
        p2_ = self.toPixels(p2)
        
        pg.draw.line(canvas, "#00FF00", p1_, p2_, 2)
        
        

class Point(Train):
    def __init__(self) -> None:
        
        self.x = np.random.uniform(-1.0, 1.0)
        self.y = np.random.uniform(-1.0, 1.0)
        self.label = 1
        
        if self.y < self.f(self.x):
            self.label = -1
                        
    def getPixels(self) -> tuple[int, int]:
        return self.toPixels((self.x, self.y))
        
    def show(self, canvas, fill_color, outline_color):
        coords = self.getPixels()
        pg.draw.circle(canvas, fill_color, coords, 10)
        pg.draw.circle(canvas, outline_color, coords, 10, 5)
    