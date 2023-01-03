import cv2
import numpy as np
import zipfile
from PIL import Image
import io
from matplotlib import pyplot as plt
from scipy import ndimage, misc
import math
import pandas as pd

p=0
archive = zipfile.ZipFile('Heightmap/20210521_FX10_heightmap.zip', 'r')
imgdata = archive.read('Params_SpecimFX10_IL404_20210521-001_heightmap.png')
bytes_io = io.BytesIO(imgdata)
img = Image.open(bytes_io).convert('L') 
imr = np.array(img)
rows,cols = imr.shape
corr = imr[0:rows, 0+p:cols-p]
im = cv2.resize(corr, (1000, 700))

zf = zipfile.ZipFile('/home/myrthe/Drone data/LinkLettuce/Batch1_IL401-IL591.zip') 
pixinfo = "Batch1_IL401-IL591/IL404_0521.tsv"
dfs6 = pd.read_csv(zf.open(pixinfo), sep="\t")
ul6 = dfs6['Label'].unique()
new_im6 = np.copy(im)
new_im6[:,:] = 0
ul_len = len(ul6)

for l in range(ul_len):
    y = dfs6.loc[dfs6['Label'] == ul6[l]]["Y"].values
    x = dfs6.loc[dfs6['Label'] == ul6[l]]["X"].values
    for yy, xx in zip(y,x):
        if im[yy,xx] > -1:
            #print(im[yy,xx])
            new_im6[yy, xx] = im[yy,xx]
    
plt.imshow(new_im6)
plt.savefig('visualization.png')