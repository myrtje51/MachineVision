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
    # Open the zipped folders of each date with the height profiles in the folders
    archive = zipfile.ZipFile('Heightmap/2021' + d + '_FX10_heightmap.zip', 'r')
    # The first date, 0521, has an extra picture in the beginning which does not match to any other pictures so we skip it
    if d == "0521":
        files = archive.namelist()[1::]
    else: 
        files = archive.namelist()
    
    # Get the IL ids that are L. Sativa (since the folder also contains L. serriola) 
    sativa_ILs = os.listdir("Sativa_AS") 
    sativa_ILs = [x[6:11] for x in sativa_ILs]
    # Filter the files of the folder on only keeping the L. sativa ILs
    def filter_sat(all_files):
        sat = []
        for f in all_files:
            for s in sativa_ILs:
                if s in f and "-001_" in f: 
                    sat.append(f)
        return sorted(set(sat)) 

    # This can be changed to: 0:150. I did it in two batches so I could run the two batches parallel. 
    files = list(filter_sat(files))[150:300]
    # Loop through the batch files
    for fi in files:
        # Read the height profiles using the PIL package
        imgdata = archive.read(fi)
        bytes_io = io.BytesIO(imgdata)
        img = Image.open(bytes_io).convert('L')
        imr = np.array(img)
        # Change the shape of the image by resizing to make them all the same size and by correcting them for the pixels on the side
        rows,cols = imr.shape
        corr = imr[0:rows, 0+p:cols-p]
        im = cv2.resize(corr, (1000, 700))

        # Make a DataFrame for each date and IL id
        name = "LinkLettuce/" + fi[18:23] + "_" + d + ".tsv"
        dfs = pd.read_csv(name, sep='\t')

        # Get the unique labels of the DataFrame we just made for each crop in each picture for each date 
        ul = dfs['Label'].unique()
        nb_lab = len(ul)
        height_dataframe = []
        for n in range(nb_lab):
            # Copy the image for the crops
            new_im = np.copy(im)
            # Copy the image for soil correction
            soil_im = np.copy(im)
            # Make the entire array black (0)
            new_im[:,:] = 0
            # Get the x and y values (in lists) per crop
            y = dfs.loc[dfs['Label'] == ul[n]]["Y"].values
            x = dfs.loc[dfs['Label'] == ul[n]]["X"].values
            label = ul[n]
            # Get the x and y value per pixel
            for yy, xx in zip(y,x):
                # Fill in the black array with the height values for each crop pixel
                new_im[yy, xx] = im[yy,xx]
                # Fill in the soil array with a black color for each crop pixel 
                soil_im[yy, xx] = 0
                # add to the empty list for later dataframe making
                height_dataframe.append([fi[18:23], label, im[yy,xx]])
                
        # Get the average soil value so that later can be corrected
        soil_corr = np.mean(soil_im)
        
        # Make into a dataframe and then save the csv
        dfheight = pd.DataFrame(height_dataframe, columns = ['ID', 'Label', 'HeightPixel'])
        dfheight['HeightSoil'] = soil_corr
        name = "Heights/ClosestPixelHeight/" + fi[18:23] + "_" + d + "_height.tsv"
        dfheight.to_csv(name, sep="\t")

