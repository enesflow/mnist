from utils import *
import threading
import PIL.Image
from matplotlib import pyplot as plt
import time
from tensorflow.keras import models, datasets
import tensorflow as tf
import numpy as np
from random import randint
from scipy.ndimage import zoom as zoom_sc

my_model = models.load_model("C:\\Users\\enesk\\Desktop\\mnist\\model.h5", compile=False)
my_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.003),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("Loaded the model")

res = []
font = pygame.font.SysFont(name="comicsans", size=32)

texts = []
textRects = []

for i in range(0,10):
    res.append(0.0)
    cr = res[i]
    texts.append(font.render(f"{i} => ?", True, (255-(cr*255),cr*255,100), (0,0,0)))
    textRects.append(texts[i].get_rect())
    textRects[i].center = (WIDTH + 200, (HEIGHT-ROWS) // 9 * i + ROWS)

def get(n):
    randomnumber = x_train[n]
    randomnumber = zoom_sc(randomnumber, ROWS // 28)
    for i in range(0, ROWS):
        for j in range(0,COLS):
            a = randomnumber[i][j]
            grid[i][j] = (a,a ,a )
    predict()
    buttons[3] = Button(500, button_y, 50, 50, WHITE, f":{y_train[n]}", BLACK)
def predict_number(grid):
    global res
    raw = PIL.Image.fromarray(np.array(grid))
    im = raw.resize((28,28))
    im = list(im.getdata())
    raw.close()
    im = np.array(im).reshape((-1,28,28))
    x1 = True
    prediction = (list(my_model.predict(im, verbose="0"))[0])
    res = {}
    for i in range(0,10):
        res[i] = prediction[i]

    res = dict(sorted(res.items(), key=lambda item: item[1])[::-1])


WIN = pygame.display.set_mode((WIDTH+400, HEIGHT+100))
pygame.display.set_caption("Drawing Program")

def init_grid(rows, cols, color):
    grid = []

    for i in range(rows):
        grid.append([])
        for _ in range(cols):
            grid[i].append(color)

    return grid


def draw_grid(win, grid):
    for i, row in enumerate(grid):
        for j, pixel in enumerate(row):
            pygame.draw.rect(win, pixel, (j * PIXEL_SIZE, i *
                                          PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))



def draw(win, grid, buttons):
    win.fill(BG_COLOR)
    draw_grid(win, grid)

    for button in buttons:
        button.draw(win)

    pygame.draw.line(WIN, WHITE, (0, HEIGHT), (WIDTH, HEIGHT), 2)
    pygame.draw.line(WIN, WHITE, (WIDTH, 0), (WIDTH, HEIGHT), 2)
    for i in range(0,10):
        WIN.blit(texts[i], textRects[i])
    pygame.display.update()


def get_row_col_from_pos(pos):
    x, y = pos
    row = y // PIXEL_SIZE
    col = x // PIXEL_SIZE

    if row >= ROWS:
        raise IndexError

    return row, col

def predict():
    global texts
    def predict_number_function():
        global texts
        for i in range(len(img)):
            for j in range(len(img[i])):
                img[i][j] = grid[i][j][0]
        predict_number(img)
        for i in range(0,10):
            cr = res[i]
            texts[i] = font.render(f"{i} => {round(cr * 100)}%", True, (255-(cr*255),cr*255,100), (0,0,0))
    predict_thread = threading.Thread(target=predict_number_function, name="Predict")
    predict_thread.start()


def get_circle(row,col):
    pixels = []
    for i in range(max(0,row-SIZE), row+SIZE):
        for j in range(max(0,col-SIZE), col+SIZE):
            if ((i-row)**2+(j-col)**2<=SIZE**2):
                pixels.append((i,j))
    return pixels

circles = []
for i in range(ROWS):
    circles.append([])
    for j in range(COLS):
        circles[i].append([])
        circles[i][j] = get_circle(i, j)

run = True
clock = pygame.time.Clock()
grid = init_grid(ROWS, COLS, BG_COLOR)
img = init_grid(ROWS, COLS, BG_COLOR)
drawing_color = WHITE
mouse_press = 0

button_y = (HEIGHT) - TOOLBAR_HEIGHT/2 - 25 + 50
buttons = [
    Button(10, button_y, 150, 50, WHITE, "clear", BLACK),
    Button(170, button_y, 150, 50, WHITE, "screenshot", BLACK),
    Button(340, button_y, 150, 50, WHITE, "random", BLACK),
    Button(500, button_y, 50, 50, WHITE, ":", BLACK)
]

while run:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        if pygame.mouse.get_pressed()[0] or pygame.mouse.get_pressed()[2]:
            mouse_press+=1
            pos = pygame.mouse.get_pos()
            FPS=MAX_FPS

            try:
                if (pygame.mouse.get_pressed()[2]):
                    drawing_color = (0,0,0)
                else:
                    drawing_color = (255,255,255)
                row, col = get_row_col_from_pos(pos)
                grid[row][col] = drawing_color
                for i in circles[row][col]:
                    grid[i[0]][i[1]] = drawing_color

                if (mouse_press % 15 == 0):
                    predict()
            except IndexError:
                for button in buttons:
                    if not button.clicked(pos):
                        continue

                    drawing_color = button.color
                    if button.text == "clear":
                        grid = init_grid(ROWS, COLS, BG_COLOR)
                    if button.text == "screenshot":
                        for i in range(len(img)):
                            for j in range(len(img[i])):
                                img[i][j] = grid[i][j][0]
                        raw = PIL.Image.fromarray(np.array(img))
                        im = raw.resize((28,28))
                        plt.imshow(im)
                        plt.show()
                    if button.text == "random":
                        n = randint(0,59999)
                        get(n)
                    if button.text[0] == ":" and len(button.text) == 2:
                        while True:
                            n = randint(0,59999)
                            if y_train[n] == int(button.text[1]):
                                break
                        get(n)

        else:
            FPS=MIN_FPS
            if (mouse_press > 0):
                predict()
            mouse_press = 0
    draw(WIN, grid, buttons)

pygame.quit()
