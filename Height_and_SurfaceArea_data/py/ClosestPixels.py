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

# We loop through the dates and the pixels at the same time using zip() 
for d, p in zip(dates,pixel_width): 
    # We open all the images related to the date without unzipping (to save space)
    archive = zipfile.ZipFile('Heightmap/2021' + d + '_FX10_heightmap.zip', 'r')
    # 0521 has an extra file at the beginning without an accession linked to it so we ignore it
    if d == "0521":
        files = archive.namelist()[1::]
    else: 
        files = archive.namelist()
    
    # We filter all the images to only look at Sativa 
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
    
    # We loop through the individual images that are still zip files
    for fi in files: 
        # We read the files as images (numpy arrays)
        imgdata = archive.read(fi)
        bytes_io = io.BytesIO(imgdata)
        img = Image.open(bytes_io).convert('L')
        imr = np.array(img)
        
        # We determine the shape of the images and correct the images using the pixel information at the beginning
        rows,cols = imr.shape
        corr = imr[0:rows, 0+p:cols-p]
        
        # We resize the images so that they are all the same size 
        im = cv2.resize(corr, (1000, 700))

        # We try to look for local maxima in our height images
        data_max = ndimage.maximum_filter(im, 100, mode = 'nearest')
        maxima = (im == data_max)
        data_min = ndimage.minimum_filter(im, 100)
        diff = ((data_max - data_min) > 10)
        maxima[diff == 0] = 0
        labeled, num_objects = ndimage.label(maxima)
        xy = np.array(ndimage.center_of_mass(im, labeled, range(1, num_objects+1)))

        def dist2(p1, p2):
            """
            Input: Two sets of x and y values in Tuple
            Output: Euclidian distance
            """
            # We compute the Euclidian distance
            return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

        def fuse(points, d):
            """
            Input: 
                points --> List of X and Y values in tuples
                d --> the maximum distance
            Output: 
                A filtered list of X and Y values with the ones that are too close to each other removed
            """
            # In this function we try to eliminate local minima that are too close to each other
            # meaning they are probably the same height value. This is done by computing the Euclidian distance
            ret = []
            ret2 = []
            d2 = d * d
            n = len(points)
            taken = [False] * n
            for i in range(n):
                if not taken[i]:
                    count = 1
                    point = [points[i][0], points[i][1]]
                    taken[i] = True
                    for j in range(i+1, n):
                        if dist2(points[i], points[j]) < d2:
                            point[0] += points[j][0]
                            point[1] += points[j][1]
                            count+=1
                            taken[j] = True
                    point[0] /= count
                    point[1] /= count
                    ret.append([point[0], point[1]])
                    ret2.append((int(point[0]), int(point[1])))
            return np.array(ret), ret2

        filt, filt2 = fuse(xy, 80)

        # The height of the soil is different for some dates meaning that we need to take a different 
        # value for the soil for different dates. 
        mh = 15
        if d == "0521":
            mh = 10
        if d == "0601" or d == "0604":
            mh = 25

        # rows = y = i
        # columns = x = j
        def closestPixel(filt, im, mh):
            """
            Input:
                filt --> list of x and y values of the centers found using the local maxima
                im --> an image
                mh --> soil height to correct on
            Output: 
                row_df --> list of lists that can be turned into a dataframe
            """
            # for this function you input the centers found by local maxima, the image and the soil threshold
            # we determine the shape of the image
            rows,cols = im.shape
            # we initiate a list which will eventually become a Pandas dataframe
            row_df = []
            # We loop through the rows en the columns of our image to get specific pixel values.
            for i in range(rows):
                for j in range(cols):
                    # Here we determine the pixel value of place: i, j
                    k = im[i,j]
                    # If the value is higher than the soil threshold it is a plant. 
                    if k > mh:
                        close = []
                        for x in filt: 
                            # Here we compute the distance of this specific pixel and all determined centers using local maxima
                            d = (j-x[1])**2 + (i-x[0])**2 
                            l = str(x[1]) + "," + str(x[0])
                            close.append((l, d))
                        # Here we determine which center the pixel is closest to using min()
                        label = min(close, key = lambda t: t[1])
                        # And finally, we add it to our list that we initiated at the beginning. 
                        row_df.append([label[0], i, j])

            return row_df

        # We run the function
        row_df = closestPixel(filt, im, mh)
        
        # turn the list of lists into a dataframe
        dfs = pd.DataFrame(row_df, columns = ['Label', 'Y', 'X'])
        name = "LinkLettuce/" + fi[18:23] + "_" + d + ".tsv"
        # Save the dataframe in the directory: LinkLettuce/
        dfs.to_csv(name, sep="\t")
