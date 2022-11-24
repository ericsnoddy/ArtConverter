# std lib
# import sys
import math

# reqs
import pygame as pg
import numpy as np
from numba import njit  # just-in-time compiler
import pygame.gfxdraw as gdraw
import cv2

# local
from settings import (
    # ascii art
    # ASCII_CHARS, FONT, FONT_COLOR, IS_BOLD, CHAR_STEP_FONT_SCALE,

    # pixel art
    FONT_SIZE,
    BG_COLOR,
    CV2_WIN_SCALE, 
    COLORS,
    PIXEL_SIZE
)

# decorator for numba's just-in-time method
# black magic that greatly speeds up computation for performance
@njit(fastmath=True)
def accelerate_conversion(image, width, height, color_coeff, step):
    array_of_values = []
    for x in range(0, width, step):
        for y in range(0, height, step):
            r, g, b = image[x, y] // color_coeff
            if sum((r, g, b)):
                array_of_values.append(((r, g, b), (x, y)))
    return array_of_values


class Pixelator:
    def __init__(self, path='input/sunny.mp4', font_size=FONT_SIZE, pixel_size=PIXEL_SIZE, color_level=COLORS):
        # pygame init
        pg.init()
        self.path = path
        self.capture = cv2.VideoCapture(path)
        self.PIXEL_SIZE = pixel_size
        self.COLOR_LVL = color_level
        self.image = self.get_image()
        self.RES = self.WIDTH, self.HEIGHT = self.image.shape[0], self.image.shape[1]  # cv2 attrs
        self.win = pg.display.set_mode(self.RES)
        self.clock = pg.time.Clock()

        # prerender chars for performance
        self.PALETTE, self.COLOR_COEFF = self.create_palette()


    def draw_converted_image(self):
        self.image = self.get_image()
        # convert array of color values into array of color tuples as indices
        array_of_values = accelerate_conversion(self.image, self.WIDTH, self.HEIGHT, self.COLOR_COEFF, self.PIXEL_SIZE)

        for color_key, (x, y) in array_of_values:
            # (0, 0, 0) does not exist in array so nothing is drawn in those squares; BG_COLOR shows
            color = self.PALETTE[color_key]
            gdraw.box(self.win, (x, y, self.PIXEL_SIZE, self.PIXEL_SIZE), color)


    def create_palette(self):
        # pre-render all the chars for each color value to improve performance
        
        # linspace generates COLOR_LVL (8) evenly spaced values (0-255), outputting an ndarray with int-type elements
        # also outputs a coeff equal to the space between values
        colors, color_coeff = np.linspace(0, 255, num=self.COLOR_LVL, dtype=int, retstep=True)        
        color_coeff = int(color_coeff)  # we don't want float

        # init a list of arrays of every combination of rgb in colors
        color_palette_list = [np.array([r, g, b]) for r in colors for g in colors for b in colors]

        # create dict = { keys: char, values: {keys: tuples of normalized color values, values: rendered font for key char} 
        palette = {}
        for color in color_palette_list:
            color_key = tuple(color // color_coeff)
            palette[color_key] = color
        return palette, color_coeff


    def get_image(self):
    
        ret, self.cv2_image = self.capture.read()  # Store a Numpy pixel array of frame, get next frame (ret)
        if not ret: exit()

        # transpose for pygame compatibility
        transposed_image = cv2.transpose(self.cv2_image)

        # convert to RGB array and return
        image = cv2.cvtColor(transposed_image, cv2.COLOR_BGR2RGB)
        return image


    def draw_cv2_image(self):
        # convert to lower res so doesn't interfere with pygame win
        resized_cv2_image = cv2.resize(self.cv2_image, (int(self.WIDTH * CV2_WIN_SCALE), int(self.HEIGHT * CV2_WIN_SCALE)), interpolation=cv2.INTER_AREA)
        cv2.imshow('img', resized_cv2_image)


    def draw(self):
        self.win.fill(BG_COLOR)  # BG_COLOR will substitute for color_index (0, 0, 0)
        self.draw_converted_image()
        self.draw_cv2_image()


    def save_image(self):
        pg_img = pg.surfarray.array3d(self.win)
        cv2_img = cv2.transpose(pg_img)
        cv2.imwrite('output/converted_image.jpg', cv2_img)


    def run(self):
        while True:
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    exit()
                elif e.type == pg.KEYDOWN:
                    if e.key == pg.K_s:
                        self.save_image()
            self.draw()
            pg.display.set_caption(f'{self.clock.get_fps():.1f}')
            pg.display.flip()
            self.clock.tick(60)


if __name__ == '__main__':
    app = Pixelator()
    app.run()

