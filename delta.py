# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:33:36 2020

@author: lambozsolty
"""

import os
import time
from PIL import Image
from matplotlib import pyplot
from matplotlib.image import imread

airplanes = os.listdir('airplanes/')
cars = os.listdir('cars/')
   
fig = pyplot.figure()

i = 1

for filename in airplanes:
    image = Image.open('airplanes/'+filename)
    image.thumbnail((64, 64), Image.ANTIALIAS)
    fig.add_subplot(10, 4, i).title.set_text("plane")
    i += 1
    pyplot.imshow(image)

for filename in cars:
    image = Image.open('cars/'+filename)
    image.thumbnail((64, 64), Image.ANTIALIAS)
    fig.add_subplot(5, 8, i).title.set_text("car")
    i += 1
    pyplot.imshow(image)
    
fig.suptitle('Vehicles', fontsize=16)
pyplot.imshow()



