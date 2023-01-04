"""
### ANOUNCEMENT
This is a test script that I did not end up using for the results of my project. Therefore the documentation is a bit messy.
### ANOUNCEMENT
"""
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
from scipy.spatial import Voronoi, voronoi_plot_2d
import pyclesperanto_prototype as cle

# initialize GPU
device = cle.select_device("GTX")
print("Used GPU: ", device)

dates = ["0521", "0528", "0601", "0604", "0608", "0611", "0615", "0618", "0622", "0625"]
for d in dates: 
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

    files = list(filter_sat(files))
    for fi in files: 
        imgdata = archive.read(fi)
        bytes_io = io.BytesIO(imgdata)
        img = Image.open(bytes_io).convert('L')
        imr = np.array(img) 
        im = cv2.resize(imr, (1000, 700))

        data_max = ndimage.maximum_filter(im, 100, mode = 'nearest')
        maxima = (im == data_max)
        data_min = ndimage.minimum_filter(im, 100)
        diff = ((data_max - data_min) > 10)
        maxima[diff == 0] = 0
        labeled, num_objects = ndimage.label(maxima)
        xy = np.array(ndimage.center_of_mass(im, labeled, range(1, num_objects+1)))

        def dist2(p1, p2):
            return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

        def fuse(points, d):
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
        
        #vor = Voronoi(filt)
        #fig = voronoi_plot_2d(vor)
        
        img_gaussian = cle.gaussian_blur(im, sigma_x=1, sigma_y=1, sigma_z=1)
        img_thresh = cle.threshold_otsu(img_gaussian)
        voronoi_separation = cle.masked_voronoi_labeling(filt, img_thresh)
        
        name = "LinkLettuce/" + fi[18:23] + "_" + d + ".png"
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))
        cle.imshow(im, plot=axs[0], color_map='gray')
        cle.imshow(voronoi_separation, labels=True, plot=axs[1])
        fig.savefig(name)
        

        #new_im = function(filt, im)
        #row_df = function(filt, im, mh)
        #dfs = pd.DataFrame(row_df, columns = ['Label', 'Y', 'X'])

        #name = "LinkLettuce/" + fi[18:23] + "_" + d + ".tsv"
        #dfs.to_csv(name, sep="\t")

        #new_im = np.copy(im)
        #for x in row_df:
        #    new_im[x[1], x[2]] = 255
        break
