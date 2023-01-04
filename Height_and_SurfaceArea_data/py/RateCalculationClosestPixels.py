import os
import pandas as pd

# Open the directory with the height information per pixel
list_height = os.listdir("Heights/ClosestPixelHeight/")
# Create an empty list
df_list = []
# Loop through the list of files in the directory with the height information
for he in list_height:
    # Filter by only taking the files of a certain date. In this case: 06-25
    if "0625" in he:
        # Determine where the file names with the path 
        name_H = "Heights/ClosestPixelHeight/" + he
        # Read the files as csvs
        df_H = pd.read_csv(name_H, sep="\t")
        # Calculate the mean median and max height and area for each plant
        df_Ar = df_H.groupby(['Label']).size().reset_index(name='area')
        df_mean = df_H.groupby(['Label'])['HeightPixel'].mean().reset_index(name='mean')
        df_median = df_H.groupby(['Label'])['HeightPixel'].median().reset_index(name='median')
        df_max = df_H.groupby(['Label'])['HeightPixel'].max().reset_index(name='max')
        # Correct for soil height
        df_Ar["meanHeight"] = df_mean["mean"] - df_H['HeightSoil']
        df_Ar["medianHeight"] = df_median["median"] - df_H['HeightSoil']
        df_Ar["maxHeight"] = df_max["max"] - df_H['HeightSoil']
        # Add the id to the dataframe
        df_Ar["id"] = he[0:5]
        # Append dataframe to an empty list
        df_list.append(df_Ar)
        
appended_data = pd.concat(df_list)
appended_data.to_csv("20210625_ClosestPixel_PhenotypesBatch2.tsv", sep="\t")
