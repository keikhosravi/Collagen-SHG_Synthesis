from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform, img_as_float, color
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from IPython import display
# Ignore warnings
import warnings
import csv
import copy
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

def get_csv_path(screened=False):
    if screened:
        path = os.path.join(os.getcwd(), 'Data', 'screened', 'image_files.csv')
    else:
        path = os.path.join(os.getcwd(), 'Data', 'image_files.csv')
    return path

def generate_csv(screened=False):
    cwd = os.getcwd()
    if screened:
        csvFilePath = os.path.join(cwd, 'Data', 'screened', 'image_files.csv')
        hePath = os.path.join(cwd, 'Data', 'screened', 'HE_JPEG')
        shgPath = os.path.join(cwd, 'Data', 'screened', 'SHG_JPEG')
    else:      
        csvFilePath = os.path.join(cwd, 'Data', 'image_files.csv')
        hePath = os.path.join(cwd, 'Data', 'HE_JPEG')
        shgPath = os.path.join(cwd, 'Data', 'SHG_JPEG')
    fileListHE = [name for name in os.listdir(hePath) if 
                      os.path.isfile(os.path.join(hePath, name))]
    fileListSHG = [name for name in os.listdir(shgPath) if 
                      os.path.isfile(os.path.join(shgPath, name))]
    numOfPatch = len(fileListHE)
    print("Number of Patches: " + str(numOfPatch))
    print(str(fileListHE[0]))
    with open(csvFilePath, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', 
                                quoting=csv.QUOTE_MINIMAL)
        for i in range(numOfPatch):
            he = os.path.join(hePath, str(fileListHE[i]))
            shg = os.path.join(shgPath, str(fileListSHG[i]))
            filewriter.writerow([he, shg])
