import pygame as pg
from training import Point, Train
from perceptron import Perceptron
from config import *

points = []


Training_Index = 0
Points_Trained = 0

for i in range(NOFPOINTS):
    points.append(Point())

percept = Perceptron(2)
training_line = Train()

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
            f"Weights: [ {percept.weights[0]:.2f}, {percept.weights[1]:.2f} ]| "
            f"Batch error (mean absolute): {batch_error:.2f} | "
            f"Points trained so far: {Points_Trained}"
        )
        
    Training_Index = (Training_Index + BATCH_SIZE) % NOFPOINTS



def main() -> None:
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Perceptron")
    clock = pg.time.Clock()

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                
            if event.type == pg.MOUSEBUTTONDOWN:
                train()

        screen.fill(BLACK)

        training_line.show_line(screen)
        percept.show(screen)

        for point in points:
            fill_color = YELLOW if point.label == 1 else POINT_BLUE
            guess = percept.guess([point.x, point.y])
            outline_color = GREEN if guess == point.label else RED
            point.show(screen, fill_color, outline_color)




        pg.display.flip()
        clock.tick(60)

    pg.quit()
if __name__ == "__main__":
    main()