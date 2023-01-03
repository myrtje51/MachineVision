# MachineVision

## Description project
This GIT repository contains all the code that I've written over the course of 5 months for my major internship project where I use machine vision techniques to analyze sensory and image data. This includes IPython NoteBooks en normal Python scripts (Python 3.11). Some of the folders include data but the most important data used in most of the code is not uploaded in this directory since some of the data takes up a lot of space. 

## Description per folder
There are two main folders: 
- Height_and_SurfaceArea_data: this folder contains all the code related to how the height, surface area and growthrate is calculated
  - ipynb & py: contains most of the important code where the first folder contains IPython Notebooks and the second python code. The code: ClosestPixels.py is the code that determines the centers using Local Maxima and finding the nearest pixels. The code: ClosestPixelHeightCalculation.py calculates the height per pixel and saves it in a different file. The code: RateCalculationClosestPixels.py calculates the growth rates for the surface area of the plant and the height. The code: autoencoder_crop.py is where I on a friday afternoon tried to code a simple autoencoder that can extract features from the RGB images. 
  - GrowthRates_Radius: contains the calculated growthrates of the Radius method and the code: growthrate_sativa.py. Which is the code for calculating the growthrates using the Radius method
  - LinkLettuce/PhenotypesClosestPixels: contains all the results from the Nearest Pixel methods including the growth rates
- Hyperspectral_Data: this folder contains all the code related to the hyperspectral indices that are calculated. The code: Input_GWAS_generation.py generates data that can be used as input for our GWAS. The code: read_hyperspectral.py generates a TSV file with the different spectral indice calculations. 
