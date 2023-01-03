import os
import pandas as pd
import zipfile

# Open the 2 zipped folders with Batches of spectral indices 
zf_b1 = zipfile.ZipFile('Indices/Batch1_IL401-IL585_indices.zip')
zf_b2 = zipfile.ZipFile('Indices/Batch2_IL594-IL783_indices.zip')
nl_b1 = zf_b1.namelist()
nl_b2 = zf_b2.namelist()

df_list = []
for b1 in nl_b1:
    # Filter for the date
    if "0625" in b1:
        # Open the files
        df = pd.read_csv(zf_b1.open(b1))
        # Drop the NaN values
        df1 = df.dropna()
        # Calculate the means of the indices
        df_PSRI = df1.groupby(['ID'])['PSRI'].mean().reset_index(name='meanPSRI')
        df_CRI550 = df1.groupby(['ID'])['CRI550'].mean().reset_index(name='mean')
        df_MCARI = df1.groupby(['ID'])['MCARI'].mean().reset_index(name='mean')
        df_ARI = df1.groupby(['ID'])['ARI'].mean().reset_index(name='mean')
        df_PSRI["meanCRI550"] = df_CRI550["mean"]
        df_PSRI["meanMCARI"] = df_MCARI["mean"]
        df_PSRI["meanARI"] = df_ARI["mean"]
        # Add to the empty list so that they can become a full dataframe
        df_list.append(df_PSRI)
        
for b2 in nl_b2:
    # Filter for the date
    if "0625" in b2:
        # Open the files
        dfl = pd.read_csv(zf_b2.open(b2))
        # Drop the NaN values
        df2 = dfl.dropna()
        # Calculate the means of the indices
        df_PSRI2 = df2.groupby(['ID'])['PSRI'].mean().reset_index(name='meanPSRI')
        df_CRI5502 = df2.groupby(['ID'])['CRI550'].mean().reset_index(name='mean')
        df_MCARI2 = df2.groupby(['ID'])['MCARI'].mean().reset_index(name='mean')
        df_ARI2 = df2.groupby(['ID'])['ARI'].mean().reset_index(name='mean')
        df_PSRI2["meanCRI550"] = df_CRI5502["mean"]
        df_PSRI2["meanMCARI"] = df_MCARI2["mean"]
        df_PSRI2["meanARI"] = df_ARI2["mean"]
        # Add to the same list so that they can become a full dataframe
        df_list.append(df_PSRI2)

# Concat all the dataframe into one big dataframe
appended_data = pd.concat(df_list)
appended_data.to_csv("20210625_ClosestPixel_HyperspectralIndex.tsv", sep="\t", index=False)
