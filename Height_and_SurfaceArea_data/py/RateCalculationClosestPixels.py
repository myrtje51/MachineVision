import os
import pandas as pd

list_height = os.listdir("Heights/ClosestPixelHeight/")
df_list = []
for he in list_height:
    if "0625" in he:
        name_H = "Heights/ClosestPixelHeight/" + he
        df_H = pd.read_csv(name_H, sep="\t")
        df_Ar = df_H.groupby(['Label']).size().reset_index(name='area')
        df_mean = df_H.groupby(['Label'])['HeightPixel'].mean().reset_index(name='mean')
        df_median = df_H.groupby(['Label'])['HeightPixel'].median().reset_index(name='median')
        df_max = df_H.groupby(['Label'])['HeightPixel'].max().reset_index(name='max')
        df_Ar["meanHeight"] = df_mean["mean"] - df_H['HeightSoil']
        df_Ar["medianHeight"] = df_median["median"] - df_H['HeightSoil']
        df_Ar["maxHeight"] = df_max["max"] - df_H['HeightSoil']
        df_Ar["id"] = he[0:5]
        df_list.append(df_Ar)
        
appended_data = pd.concat(df_list)
appended_data.to_csv("20210625_ClosestPixel_PhenotypesBatch2.tsv", sep="\t")

#df_21 = pd.read_csv("Heights/ClosestPixelHeight/IL402_0608_height.tsv", sep="\t")
#df_25 = pd.read_csv("LinkLettuce/IL402_0608.tsv", sep="\t")
#print(df_21)
#print(df_25)