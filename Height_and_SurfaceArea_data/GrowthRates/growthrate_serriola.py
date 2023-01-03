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

# Import all time slots
archive_ts1 = zipfile.ZipFile('NDVI/20210521_FX10_NDVI.zip', 'r')
archive_ts2 = zipfile.ZipFile('NDVI/20210528_FX10_NDVI.zip', 'r')
archive_ts3 = zipfile.ZipFile('NDVI/20210601_FX10_NDVI.zip', 'r')
archive_ts4 = zipfile.ZipFile('NDVI/20210604_FX10_NDVI.zip', 'r')
archive_ts5 = zipfile.ZipFile('NDVI/20210608_FX10_NDVI.zip', 'r')
archive_ts6 = zipfile.ZipFile('NDVI/20210611_FX10_NDVI.zip', 'r')
archive_ts7 = zipfile.ZipFile('NDVI/20210615_FX10_NDVI.zip', 'r')
archive_ts8 = zipfile.ZipFile('NDVI/20210618_FX10_NDVI.zip', 'r')
archive_ts9 = zipfile.ZipFile('NDVI/20210622_FX10_NDVI.zip', 'r')
archive_ts10 = zipfile.ZipFile('NDVI/20210625_FX10_NDVI.zip', 'r')

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

# Filter to only get the Sativa accessions
sativa_ILs = os.listdir("Serriola_AS") 
sativa_ILs = [x[6:11] for x in sativa_ILs]
def filter_sat(all_files):
    sat = []
    for f in all_files:
        for s in sativa_ILs:
            if s in f and "001_ndvi" in f:
                sat.append(f)
    return sat

# Select two Time Slots to compute the mean growthrate for
files_ts1_sat = filter_sat(files_ts5)
files_ts2_sat = filter_sat(files_ts6)

# Check if both Time Slots have information for all accessions. 
ffiles_ts1 = []
ffiles_ts2 = []
for x in files_ts1_sat:
    for y in files_ts2_sat:
        if x[18:23] == y[18:23]:
            ffiles_ts1.append(x)
            ffiles_ts2.append(y)
            
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
    
def observations(image, obj, mask):
    """
    Input: RGB image, one of the objects created by findContours() and the mask created by thresholding() 
    Output: the Convex hull area of the individual plant
    """
    analysis_image = pcv.analyze_object(img=image, obj=obj, mask=mask, label="default")
    plant_area = pcv.outputs.observations
    cha = plant_area['default']['convex_hull_area']['value']
    return cha

def gr_calc(ec_list_past, cha_list_past, ec_list_present, cha_list_present): 
    """
    Input: list of x and y values for each individual plant for the past and present and the convex hull area of each plant
    Output: The growth rates for each individual plant
    """
    growth_rates = []
    
    # Loop through the coordinates and the sizes of the earlier date 
    for x_past, y_past in zip(ec_list_past, cha_list_past):
        dist_list = []
        close_past = []
        close_present = []
        size_past = []
        size_present = []
        
        # Loop through the coordinates and the sizes of the later date
        for x_present, y_present in zip(ec_list_present, cha_list_present):
            # Calculate the Euclidian distance between the past and present to select the individual crops
            distance = math.dist(x_past, x_present)
            # Append to the different lists so that these can be indexed later on
            dist_list.append(distance)
            close_past.append(x_past)
            close_present.append(x_present)
            size_past.append(y_past)
            size_present.append(y_present)
            
        min_value = min(dist_list)
        min_index = dist_list.index(min_value)
        # Calculate the growth rate
        growth_rates.append((size_present[min_index] - size_past[min_index]) / size_past[min_index])
    return growth_rates

 # Lactuca sativa
pcv.params.debug = None
def growth_rate(files_present, files_past, archive_present, archive_past):
    # Create a list to store the growth rate and the different accessions
    list_gr = []
    list_acc = []
    for file_present, file_past in zip(files_present, files_past):
        if file_present[18:23] == file_past[18:23]:
            # Read the different images using PIL 
            imgdata_present = archive_present.read(file_present)
            imgdata_past = archive_past.read(file_past)
            bytes_present = io.BytesIO(imgdata_present)
            bytes_past = io.BytesIO(imgdata_past)
            
            #Present
            im_g_present = Image.open(bytes_present).convert('L')
            im_rgb_present = Image.open(bytes_present).convert('RGB')
            np_g_present = np.array(im_g_present) 
            np_rgb_present = np.array(im_rgb_present)
            mask_present = thresholding(np_g_present)
            # Find objects
            id_objects_present, _past = cv2.findContours(mask_present, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
            # Sort objects
            sorted_objects_present = sorted(id_objects_present, key=lambda x: cv2.contourArea(x))
            sorted_objects_present.reverse()
            # Select the 30 biggest objects (since there are 30 crops in each picture) 
            sorted_objects_present = sorted_objects_present[0:30]
            area_vals_present = []
            for i, contour in enumerate(sorted_objects_present):
                # ID and store area values and centers of mass for labeling them
                m = cv2.moments(contour)
                # Skip iteration if contour area is zero
                # This is needed because cv2.contourArea can be > 0 while moments area is 0.
                if m['m00'] != 0:
                    area_vals_present.append(m['m00']) 
                
            
            # If the objects are too small or too big they are probably not crops so remove them --> outliers
            area_vals_presentf=[x for x in area_vals_present if x >=400 and x < 14000]
            if len(area_vals_presentf) == 0: 
                continue
                    
            im_g_present.close()
            im_rgb_present.close()

            #Past
            im_g_past = Image.open(bytes_past).convert('L')
            im_rgb_past = Image.open(bytes_past).convert('RGB')
            np_g_past = np.array(im_g_past) 
            np_rgb_past = np.array(im_rgb_past)
            mask_past = thresholding(np_g_past)
            # Find objects
            id_objects_past, _past = cv2.findContours(mask_past, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
            sorted_objects_past = sorted(id_objects_past, key=lambda x: cv2.contourArea(x))
            sorted_objects_past.reverse()
            # Select the 30 biggest objects 
            sorted_objects_past = sorted_objects_past[0:30]
            area_vals_past = []
            # Get the size of the objects
            for i, contour in enumerate(sorted_objects_past):
                m = cv2.moments(contour)
                if m['m00'] != 0:
                    area_vals_past.append(m['m00'])
            
            # Filter out the outliers
            area_vals_pastf=[x for x in area_vals_past if x >= 400 and x <= 14000]
            if len(area_vals_pastf) == 0: 
                continue
                
            # Calculate the growth rate
            growth_rate = (mean(area_vals_presentf) - mean(area_vals_pastf)) / mean(area_vals_pastf)
            # Add the growth rate to the list and the accompanying accession to the other list 
            list_gr.append(growth_rate)
            list_acc.append(file_present[18:23])
            im_g_past.close()
            im_rgb_past.close()
            
    return list_gr, list_acc

gr, acc = growth_rate(ffiles_ts2, ffiles_ts1, archive_ts6, archive_ts5)
df = pd.DataFrame(gr, index =acc, columns =['Growth_rate'])
name = 'Serriola_20210608-20210611_growthrate.csv'
df.to_csv(name, index=True)
                   