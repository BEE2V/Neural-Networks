import pygame as pg
from training import Point
from perceptron import Perceptron

WIDTH = 700
HEIGHT = 700
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
DASH_LENGTH = 12
DASH_SPACING = 20

NOFPOINTS = 100
points = []

for i in range(NOFPOINTS):
    points.append(Point(WIDTH, HEIGHT))

percept = Perceptron(2)

# Train the perceptron

for point in points:
    inputs = [point.x, point.y]
    percept.train(inputs, point.label)




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

        screen.fill(BLACK)

        for offset in range(0, min(WIDTH, HEIGHT), DASH_SPACING):
            end = min(offset + DASH_LENGTH, min(WIDTH, HEIGHT) - 1)
            pg.draw.line(screen, BLUE, (offset, offset), (end, end), 2)

        for point in points:
            point.show(screen)




        pg.display.flip()
        clock.tick(60)

    pg.quit()
if __name__ == "__main__":
    main()