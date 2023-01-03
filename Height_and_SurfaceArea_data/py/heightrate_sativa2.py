# The imports needed to run this script
import zipfile
import numpy as np
import skimage.io
import skimage.viewer
import skimage.filters
import cv2
from PIL import Image
import io
from plantcv import plantcv as pcv
import os
import glob
import math
from statistics import mean
import pandas as pd
import gc
from ast import literal_eval

# Import all time slots heightmaps
archive_ts1 = zipfile.ZipFile('Heightmap/20210521_FX10_heightmap.zip', 'r')
archive_ts2 = zipfile.ZipFile('Heightmap/20210528_FX10_heightmap.zip', 'r')
archive_ts3 = zipfile.ZipFile('Heightmap/20210601_FX10_heightmap.zip', 'r')
archive_ts4 = zipfile.ZipFile('Heightmap/20210604_FX10_heightmap.zip', 'r')
archive_ts5 = zipfile.ZipFile('Heightmap/20210608_FX10_heightmap.zip', 'r')
archive_ts6 = zipfile.ZipFile('Heightmap/20210611_FX10_heightmap.zip', 'r')
archive_ts7 = zipfile.ZipFile('Heightmap/20210615_FX10_heightmap.zip', 'r')
archive_ts8 = zipfile.ZipFile('Heightmap/20210618_FX10_heightmap.zip', 'r')
archive_ts9 = zipfile.ZipFile('Heightmap/20210622_FX10_heightmap.zip', 'r')
archive_ts10 = zipfile.ZipFile('Heightmap/20210625_FX10_heightmap.zip', 'r')

# Import all time slots NDVI
archive_ts1_NDVI = zipfile.ZipFile('NDVI/20210521_FX10_NDVI.zip', 'r')
archive_ts2_NDVI = zipfile.ZipFile('NDVI/20210528_FX10_NDVI.zip', 'r')
archive_ts3_NDVI = zipfile.ZipFile('NDVI/20210601_FX10_NDVI.zip', 'r')
archive_ts4_NDVI = zipfile.ZipFile('NDVI/20210604_FX10_NDVI.zip', 'r')
archive_ts5_NDVI = zipfile.ZipFile('NDVI/20210608_FX10_NDVI.zip', 'r')
archive_ts6_NDVI = zipfile.ZipFile('NDVI/20210611_FX10_NDVI.zip', 'r')
archive_ts7_NDVI = zipfile.ZipFile('NDVI/20210615_FX10_NDVI.zip', 'r')
archive_ts8_NDVI = zipfile.ZipFile('NDVI/20210618_FX10_NDVI.zip', 'r')
archive_ts9_NDVI = zipfile.ZipFile('NDVI/20210622_FX10_NDVI.zip', 'r')
archive_ts10_NDVI = zipfile.ZipFile('NDVI/20210625_FX10_NDVI.zip', 'r')

# Get the files in the archive
files_ts1 = archive_ts1.namelist()[1::]
files_ts2 = archive_ts2.namelist()
files_ts3 = archive_ts3.namelist()
files_ts4 = archive_ts4.namelist()
files_ts5 = archive_ts5.namelist()
files_ts6 = archive_ts6.namelist()
files_ts7 = archive_ts7.namelist()
files_ts8 = archive_ts8.namelist()
files_ts9 = archive_ts9.namelist()
files_ts10 = archive_ts10.namelist()

# Get the files in the archive
files_ts1_NDVI = archive_ts1_NDVI.namelist()[1::]
files_ts2_NDVI = archive_ts2_NDVI.namelist()
files_ts3_NDVI = archive_ts3_NDVI.namelist()
files_ts4_NDVI = archive_ts4_NDVI.namelist()
files_ts5_NDVI = archive_ts5_NDVI.namelist()
files_ts6_NDVI = archive_ts6_NDVI.namelist()
files_ts7_NDVI = archive_ts7_NDVI.namelist()
files_ts8_NDVI = archive_ts8_NDVI.namelist()
files_ts9_NDVI = archive_ts9_NDVI.namelist()
files_ts10_NDVI = archive_ts10_NDVI.namelist()

coord = pd.read_csv('Sativa_Coordinates2_FX10.csv')
coord = coord.fillna(0)

# Filter to only get the Sativa accessions
sativa_ILs = os.listdir("Sativa_AS") 
sativa_ILs = [x[6:11] for x in sativa_ILs]
def filter_sat(all_files):
    sat = []
    for f in all_files:
        for s in sativa_ILs:
            if s in f and "-001_" in f: 
                sat.append(f)
    return sorted(set(sat)) 

# Select two Time Slots to compute the mean growthrate for
files_ts1_sat = list(filter_sat(files_ts9))
files_ts2_sat = list(filter_sat(files_ts10))
files_ts1_sat_NDVI = list(filter_sat(files_ts9_NDVI))
files_ts2_sat_NDVI = list(filter_sat(files_ts10_NDVI))

# Check if both Time Slots have information for all accessions.
def filter_both(file1, file2):
    ffiles_ts1 = []
    ffiles_ts2 = []
    for x in file1:
        for y in file2:
            #print(x, y) 
            if x[18:23] == y[18:23]:
                ffiles_ts1.append(x)
                ffiles_ts2.append(y)
    
    return ffiles_ts1, ffiles_ts2

fi1, fi2 = filter_both(files_ts1_sat, files_ts2_sat)
fi1_NDVI, fi0_NVDI = filter_both(files_ts1_sat_NDVI, fi2)
fi00_NDVI, fi2_NDVI = filter_both(fi1, files_ts2_sat_NDVI)
            
def thresholding(image_gray):
    """ 
    Input: A gray image. 
    Output: A mask created using thresholding.
    """
    a_thresh = pcv.threshold.otsu(gray_img=image_gray, max_value=255, object_type='dark')
    a_fill = pcv.fill(bin_img=a_thresh, size=200)
    dilated = pcv.dilate(gray_img=a_fill, ksize=2, i=1)
    Mask = cv2.bitwise_not(dilated)
    return Mask

def read_image(zipped_file, archive):
    """
    Input: A zipped image that you want to pull from an archive (zipped directory)
    Output: A grey image and an RGB image
    """
    imgdata = archive.read(zipped_file)
    bytesio = io.BytesIO(imgdata)
    im_g = Image.open(bytesio).convert('L')
    im_rgb = Image.open(bytesio).convert('RGB')
    np_g = np.array(im_g) 
    np_rgb = np.array(im_rgb)
    im_g.close()
    im_rgb.close()
    return np_g, np_rgb

def find_objects(mask):
    # Find objects
    id_objects, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    # Sort objects
    sorted_objects = sorted(id_objects, key=lambda x: cv2.contourArea(x))
    sorted_objects.reverse()
    # Select the 30 biggest objects (since there are 30 crops in each picture) 
    sorted_objects = sorted_objects[0:30]
    area_vals = []
    label_coord = []
    for i, contour in enumerate(sorted_objects):
        # ID and store area values and centers of mass for labeling them
        m = cv2.moments(contour)
        # Skip iteration if contour area is zero
        # This is needed because cv2.contourArea can be > 0 while moments area is 0.
        if m['m00'] != 0:
            area_vals.append(m['m00'])
            label_coord.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))) 
    
    return area_vals, sorted_objects, label_coord

def extractCoordinates(filename, coordinates): 
    xy_present = []
    for i in range(30):
        if filename[18:23] in coordinates.values:
            coo_row = coordinates.loc[coordinates['id'] == filename[18:23]][str(i)].values[0]
            if coo_row != 0: 
                coo_tup = literal_eval(coo_row)
                xy_present.append(coo_tup)
            else:
                continue
        else:
            continue
            
    return xy_present

 # Lactuca sativa
pcv.params.debug = None
def height_rate(files_present, files_past, height_present, height_past, archive_present, archive_past, ah_present, ah_past,xys):
    # Create a list to store the growth rate and the different accessions
    list_gr = []
    list_acc = []
    list_height = []
    for file_present, file_past, hpr, hp in zip(files_present, files_past, height_present, height_past): 
        if file_present[18:23] == file_past[18:23]:
            # Present
            # Read the different images using PIL
            np_g_present, np_rgb_present = read_image(file_present, archive_present)
            np_g_presenth, np_rgb_presenth = read_image(hpr, ah_present)
            np_g_presenthh = cv2.resize(np_g_presenth, (1000, 700))
            np_g_presentt = cv2.resize(np_g_present, (1000, 700))
            mask_present = thresholding(np_g_presentt)
            # Find objects
            area_vals_present, sorted_objects_present, coord_present = find_objects(mask_present)   
            
            # If the objects are too small or too big they are probably not crops so remove them --> outliers
            sorted_objects_presentf=[x for x in sorted_objects_present if len(x)>=100 and len(x) <= 800]
            if len(sorted_objects_presentf) == 0: 
                continue

            # draw filled circles in white on black background as masks
            xy_present = extractCoordinates(file_present, xys)
                
            mask1 = np.zeros_like(np_g_presenthh)
            for xy in xy_present:
                mask2 = cv2.circle(mask1,xy,40,(255,255,255), -1)

            # put mask into alpha channel of input
            _, so_ind_present, crd_pres_circle = find_objects(mask2) 
            masked = cv2.bitwise_and(np_g_presenthh, np_g_presenthh, mask=mask2)
        
            #Past
            np_g_past, np_rgb_past = read_image(file_past, archive_past)
            np_g_pasth, np_rgb_pasth = read_image(hp, ah_past)
            np_g_pasthh = cv2.resize(np_g_pasth, (1000, 700))
            np_g_pastt = cv2.resize(np_g_past, (1000, 700))
            mask_past = thresholding(np_g_pastt)
            background = cv2.bitwise_not(mask_past)
            # Find objects
            area_vals_past, sorted_objects_past, coord_past = find_objects(mask_past)
            
            # Filter out the outliers
            sorted_objects_pastf=[x for x in sorted_objects_past if len(x)>=100 and len(x) <= 800]
            if len(sorted_objects_pastf) == 0: 
                continue
                
            mask1past = np.zeros_like(np_g_pasthh)
            for xy in xy_present:
                mask2past = cv2.circle(mask1past,xy,40,(255,255,255), -1)

            # put mask into alpha channel of input
            _, so_ind_past, crd_past_circle = find_objects(mask2past)
            maskedpast = cv2.bitwise_and(np_g_pasthh, np_g_pasthh, mask=mask2past)
            # put reverse mask (so background) into alpha channel of input
            mask_bg = cv2.bitwise_and(np_g_pasthh, np_g_pasthh, mask=background)
            # calculate the average pixel intensity of the background for later correction
            mnval = np.mean(mask_bg[np.nonzero(mask_bg)])
            
            for obj_pr, c in zip(so_ind_past, crd_past_circle): 
                crop_img = pcv.auto_crop(img=maskedpast, obj=obj_pr, padding_x=0, padding_y=0, color='image')
                calc = np.mean(crop_img[np.nonzero(crop_img)])
                height = calc-mnval
                if height > 0: 
                    list_height.append(height)
                    list_gr.append(c)
                    list_acc.append(file_present[18:23])

    return list_gr, list_acc, list_height

gr, acc, he = height_rate(fi2_NDVI, fi1_NDVI, fi2, fi1, archive_ts10_NDVI, archive_ts9_NDVI, archive_ts10, archive_ts9, coord)
df = pd.DataFrame()
df['acc'] = acc
df['height'] = he
df['coordinates'] = gr
#df_h = pd.DataFrame(he, index =acc, columns =['Height'])
#name = 'Sativa_20210622-20210625_heightrate.csv'
nameh = 'Sativa_20210622_height_corrected.csv'
df.to_csv(nameh, index=True)
                   