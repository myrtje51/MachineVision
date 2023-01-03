import cv2
import numpy as np
import zipfile
from PIL import Image
import io
from matplotlib import pyplot as plt
from scipy import ndimage, misc
import math
import pandas as pd
from ast import literal_eval
from numba import jit
import os

# A list to be able to loop through every date
dates = ["0521", "0528", "0601", "0604", "0608", "0611", "0615", "0618", "0622", "0625"]

# A list of corrections that are in the same position as the date 
# So: a correction of 24 pixels on each side needs to happen at date: 0528
pixel_width = [0,24,40,54,80,80,108,122,122,138]

for d,p in zip(dates,pixel_width):
    archive = zipfile.ZipFile('Heightmap/2021' + d + '_FX10_heightmap.zip', 'r')
    if d == "0521":
        files = archive.namelist()[1::]
    else: 
        files = archive.namelist()
        
    sativa_ILs = os.listdir("Sativa_AS") 
    sativa_ILs = [x[6:11] for x in sativa_ILs]
    def filter_sat(all_files):
        sat = []
        for f in all_files:
            for s in sativa_ILs:
                if s in f and "-001_" in f: 
                    sat.append(f)
        return sorted(set(sat)) 

    files = list(filter_sat(files))[150:300]
    for fi in files:
        imgdata = archive.read(fi)
        bytes_io = io.BytesIO(imgdata)
        img = Image.open(bytes_io).convert('L')
        imr = np.array(img)
        rows,cols = imr.shape
        corr = imr[0:rows, 0+p:cols-p]
        im = cv2.resize(corr, (1000, 700))

        name = "LinkLettuce/" + fi[18:23] + "_" + d + ".tsv"
        dfs = pd.read_csv(name, sep='\t')

        ul = dfs['Label'].unique()
        nb_lab = len(ul)
        height_dataframe = []
        for n in range(nb_lab):
            new_im = np.copy(im)
            soil_im = np.copy(im)
            new_im[:,:] = 0
            y = dfs.loc[dfs['Label'] == ul[n]]["Y"].values
            x = dfs.loc[dfs['Label'] == ul[n]]["X"].values
            label = ul[n]
            for yy, xx in zip(y,x):
                new_im[yy, xx] = im[yy,xx]
                soil_im[yy, xx] = 0
                height_dataframe.append([fi[18:23], label, im[yy,xx]])
                
        soil_corr = np.mean(soil_im)
        
        dfheight = pd.DataFrame(height_dataframe, columns = ['ID', 'Label', 'HeightPixel'])
        dfheight['HeightSoil'] = soil_corr
        name = "Heights/ClosestPixelHeight/" + fi[18:23] + "_" + d + "_height.tsv"
        dfheight.to_csv(name, sep="\t")

