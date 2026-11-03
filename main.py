import pygame as pg
import numpy as np
from training import Point, Train
from perceptron import Perceptron
from config import *

points = []

Training_Index = 0
Points_Trained = 0
TRAINING_INTERVAL_MS = 100

training_line = Train(np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0))

for i in range(NOFPOINTS):
    points.append(Point(training_line.m, training_line.c))

percept = Perceptron(2)

# Train the perceptron
def train():
    global Training_Index, Points_Trained

    batch = points[Training_Index : Training_Index + BATCH_SIZE]
    errors = []
    for point in batch:
        inputs = [point.x, point.y]
        guess = percept.guess(inputs)
        errors.append(abs(point.label - guess))
        percept.train(inputs, point.label)

    Points_Trained += len(batch)
    if errors:
        batch_error = sum(errors) / len(errors)
        print(
            f"Weights: [ {percept.weights[0]:.2f}, {percept.weights[1]:.2f}, "
            f"bias weight: {percept.weights[2]:.2f} ] | "
            f"Batch error (mean absolute): {batch_error:.2f} | "
            f"Points trained so far: {Points_Trained}"
        )
        
    Training_Index = (Training_Index + BATCH_SIZE) % NOFPOINTS



def main() -> None:
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Perceptron")
    clock = pg.time.Clock()
    space_training_elapsed = 0
    space_was_down = False

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                
            if event.type == pg.MOUSEBUTTONDOWN:
                train()

        if pg.key.get_pressed()[pg.K_SPACE]:
            if not space_was_down:
                train()
                space_training_elapsed = 0
            else:
                space_training_elapsed += clock.get_time()
                if space_training_elapsed >= TRAINING_INTERVAL_MS:
                    train()
                    space_training_elapsed -= TRAINING_INTERVAL_MS
            space_was_down = True
        else:
            space_was_down = False
            space_training_elapsed = 0

        screen.fill(BLACK)

        training_line.show_line(screen)
        percept.show(screen)

        for point in points:
            fill_color = POINT_UP if point.label == 1 else POINT_DOWN
            guess = percept.guess([point.x, point.y])
            outline_color = GREEN if guess == point.label else RED
            point.show(screen, fill_color, outline_color)




        pg.display.flip()
        clock.tick(60)

    pg.quit()
if __name__ == "__main__":
    main()