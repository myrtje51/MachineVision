import cv2
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
import os
import numpy as np
from spectral import imshow, view_cube
import spectral.io.envi as envi
import spectral.io.aviris as aviris
import spectral as spy
import pandas as pd
import zipfile

def MCARI(band, reflectance):
    ## Modified Chlorophyll Absorption Reflectance Index
    # Get the band closest to 750, 705 and 550
    upper = min(band[:-1], key=lambda x:abs(float(x)-750))
    lower = min(band[:-1], key=lambda x:abs(float(x)-705))
    correct = min(band[:-1], key=lambda x:abs(float(x)-550))
    # Get the index of the band 
    upper_index = band[:-1].index(upper)
    lower_index = band[:-1].index(lower)
    correct_index = band[:-1].index(correct)
    # Get the reflectance value of the band
    upper_pixel = reflectance[upper_index]
    lower_pixel = reflectance[lower_index]
    correct_pixel = reflectance[correct_index]
    # Calculate the MCARI with the reflectance values
    MCARI = ((upper_pixel-lower_pixel) - 0.2 * (upper_pixel - correct_pixel)) * (upper_pixel/lower_pixel)
    return MCARI

def ARI(band, reflectance):
    ## Anthocyanin Reflectance Index
    # Get the band closest to 700 and 550
    upper = min(band[:-1], key=lambda x:abs(float(x)-700))
    lower = min(band[:-1], key=lambda x:abs(float(x)-550))
    # Get the index of the band
    upper_index = band[:-1].index(upper)
    lower_index = band[:-1].index(lower)
    # Get the reflectance value of the band
    upper_pixel = reflectance[upper_index]
    lower_pixel = reflectance[lower_index]
    # Calculate the ARI with the reflectance values
    ARI = (1 / lower_pixel) - (1 / upper_pixel)
    return ARI

def CRI550(band, reflectance):
    ### Carotenoid Reflectance Index 550
    # Get the band closest to 550 and 510
    upper = min(band[:-1], key=lambda x:abs(float(x)-550))
    lower = min(band[:-1], key=lambda x:abs(float(x)-510))
    # Get the index of the band
    upper_index = band[:-1].index(upper)
    lower_index = band[:-1].index(lower)
    # Get the reflectance value of the band
    upper_pixel = reflectance[upper_index]
    lower_pixel = reflectance[lower_index]
    # Calculate the CRI550 with the reflectance values
    CRI550 = (1 / lower_pixel) - (1 / upper_pixel)
    return CRI550

def PSRI(band, reflectance):
    ## Plant Senescence Reflectance Index
    # Get the band closest to 750, 500 and 678
    upper = min(band[:-1], key=lambda x:abs(float(x)-750))
    lower = min(band[:-1], key=lambda x:abs(float(x)-500))
    middle = min(band[:-1], key=lambda x:abs(float(x)-678))
    # Get the index of the band
    upper_index = band[:-1].index(upper)
    lower_index = band[:-1].index(lower)
    middle_index = band[:-1].index(middle)
    # Get the reflectance value of the band
    upper_pixel = reflectance[upper_index]
    lower_pixel = reflectance[lower_index]
    middle_pixel = reflectance[middle_index]
    # Calculate the PSRI with the reflectance values
    PSRI = (middle_pixel - lower_pixel) / upper_pixel
    return PSRI

def main():
    # Open the zipped folder that contains the Nearest Pixel information previously determined 
    zf = zipfile.ZipFile('/home/myrthe/Height_and_SurfaceArea_data/LinkLettuce/Batch2_IL592-IL783.zip')
    # Skip every 9 files since we are only interested in the IL ids and we don't want to do the same calculations 9. 
    nl = zf.namelist()[1::9]
    #nl = nl[141::]
    # Now read the csv file
    for l in nl:
        # Initiate the list of dates and the pixel corrections on the sides (since the images differ in dimensions per date)
        dates = ["0521", "0528", "0601", "0604", "0608", "0611", "0615", "0618", "0622", "0625"]
        pixel_width = [0,24,40,54,80,80,108,122,122,138]
        # Loop through the list
        for d,p in zip(dates,pixel_width):
            # Open the csv's in the zipped file using pandas read_csv.
            pixinfo = "Batch2_IL592-IL783/"+ l[19:24] +"_"+ d + ".tsv"
            df = pd.read_csv(zf.open(pixinfo), sep="\t")

            # Open the two files needed for opening hyperspectral images: hdr and raw
            hdr = '/net/virus/linuxhome/basten-group/sarah/TraitSeeker_HSP_All/Combined_SpecimFX10_'+ l[19:24] +'_2021' + d + '-001.cmb.hdr'
            raw = '/net/virus/linuxhome/basten-group/sarah/TraitSeeker_HSP_All/Combined_SpecimFX10_'+ l[19:24] +'_2021' + d + '-001.cmb.raw'
            hyperspectral = envi.open(hdr, raw)

            # Resize the hyperspectral image data like we did with other nearest pixel methods
            nparr = np.array(hyperspectral.load())
            rows,cols,depth = nparr.shape
            corr = nparr[0:rows, 0+p:cols-p]
            reshaped = cv2.resize(nparr, (1000, 700))

            # use the spectral package to open the hdr file so to work with it more easily
            img = spy.open_image('/net/virus/linuxhome/basten-group/sarah/TraitSeeker_HSP_All/Combined_SpecimFX10_'+ l[19:24] +'_2021'+ d +'-001.cmb.hdr')

            # get the values of the different bands
            bands = img.metadata['wavelength']

            # Get the labels for each crop in each picture in the previously made Nearest Pixel 
            ul = df['Label'].unique()
            nb_lab = len(ul)
            hs_df = []
            for n in range(nb_lab):
                # Get the x and y values per label
                y = df.loc[df['Label'] == ul[n]]["Y"].values
                x = df.loc[df['Label'] == ul[n]]["X"].values
                label = ul[n]
                # Get the x and y values of a single pixel
                for yy, xx in zip(y,x):
                    # Get the value of the pixel
                    leaf_pixel = reshaped[yy:yy+1,xx:xx+1,:]
                    leaf_pixel_squeezed = np.squeeze(leaf_pixel)
                    name = "Hyperspectral_per_pixel_" + str(xx) + "_" + str(yy)
                    # Calculate the Spectral Indices
                    mcari = MCARI(bands, leaf_pixel_squeezed)
                    ari = ARI(bands, leaf_pixel_squeezed)
                    cri550 = CRI550(bands, leaf_pixel_squeezed)
                    psri = PSRI(bands, leaf_pixel_squeezed)
                    # add to an empty list so that we can later turn it into a dataframe
                    hs_df.append([l[19:24], label, mcari, ari, cri550, psri])

            # Turn the list into a DataFrame 
            dfs = pd.DataFrame(hs_df, columns = ['ID', 'Label', 'MCARI', 'ARI', 'CRI550', 'PSRI'])
            df_name = "Indices/Hyperspectral_indices_" + l[19:24] + "_" + d + ".csv"
            dfs.to_csv(df_name, index=False)

main()
