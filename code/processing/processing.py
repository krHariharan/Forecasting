import pandas as pd 
import sys
inputFile = ""
if len(sys.argv) > 1:
    inputFile = sys.argv[1]
else:
    print("Enter file to process")
    inputFile = input()

inputData = pd.read_csv("../../data/raw/"+inputFile, thousands=",")[{"Date", "Price"}]
inputData["Date"] = pd.to_datetime(inputData["Date"])

inputData["Previous Price"] = inputData["Price"].shift(-1)
inputData["Previous Date"] = inputData["Date"].shift(-1)
inputData["Date Delta"] = inputData["Date"] - inputData["Previous Date"]

d_range = pd.date_range(inputData["Date"].min(), inputData["Date"].max())
inputData = inputData.set_index('Date').reindex(d_range)

inputData["Price"] = (inputData["Price"].fillna(method='ffill') + inputData["Price"].fillna(method='bfill'))/2

inputData = inputData["Price"]

print(inputData)
inputData.to_csv("../../data/processed/"+inputFile)