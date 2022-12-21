import pygame
pygame.init()
pygame.font.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 255, 0)
GREEN = (0, 0, 255)

FPS = MIN_FPS = 60
MAX_FPS = 1000

ROWS = COLS = 28*3

WIDTH = min(56, ROWS)*10
HEIGHT = min(56, COLS)*10

SIZE=4

TOOLBAR_HEIGHT = HEIGHT - WIDTH

PIXEL_SIZE = WIDTH // COLS

BG_COLOR = BLACK


def get_font(size):
    return pygame.font.SysFont("comicsans", size)
