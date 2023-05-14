# parameter selection test script
import numpy as np
import pandas as pd
import sys
from parameter_gen import parameter_gen

# load the dataset
inputFile = ""
if len(sys.argv) > 1:
    inputFile = sys.argv[1]
else:
    print("Enter file to process")
    inputFile = input()

dataframe = pd.read_csv("../data/processed/"+inputFile, index_col=0)
dataframe.index = pd.to_datetime(dataframe.index)

params_10_yrs = parameter_gen(dataframe.index.min(), dataframe.index.max(), 3650)
params_5_yrs = parameter_gen(dataframe.index.min(), dataframe.index.max(), 1825)
params_2_yrs = parameter_gen(dataframe.index.min(), dataframe.index.max(), 730)
annual_params = parameter_gen(dataframe.index.min(), dataframe.index.max(), 365)
quarterly_params = parameter_gen(dataframe.index.min(), dataframe.index.max(), 121)
monthly_params = parameter_gen(dataframe.index.min(), dataframe.index.max(), 30)
weekly_params = parameter_gen(dataframe.index.min(), dataframe.index.max(), 7)
generated_params = pd.concat([params_10_yrs, params_5_yrs, params_2_yrs, annual_params, quarterly_params, monthly_params, weekly_params], axis=1, join='inner')

all_params = pd.concat([dataframe, generated_params], axis=1, join='inner')

params_corr = all_params.corr()["Price"].abs()

print(params_corr.nlargest(20))