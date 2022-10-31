#Dependencies
import math
import os
import sys
import numpy as np 
import cv2
import glob
import os
import matplotlib.pyplot as plt
import string
from mlxtend.plotting import plot_decision_regions



def euclidean_distance(x1, x2, y1, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def datacheck():
    print(os.listdir("./input"))
    return 

# print(euclidean_distance(1, 2, 3, 4))
datacheck()