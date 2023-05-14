# formatting and extrapolating data
import pandas as pd 
import sys
inputFile = ""
if len(sys.argv) > 1:
    inputFile = sys.argv[1]
else:
    print("Enter file to process")
    inputFile = input()

# reading input
inputData = pd.read_csv("../data/raw/"+inputFile, thousands=",")[["Date", "Price"]]
inputData["Date"] = pd.to_datetime(inputData["Date"])

# limits to normalize data
minPrice = inputData["Price"].min()
priceRange = inputData["Price"].max() - inputData["Price"].min()

# Add missing dates (ie, holidays and weekends)
d_range = pd.date_range(inputData["Date"].min(), inputData["Date"].max())
inputData = inputData.set_index('Date').reindex(d_range)

# Giving missing dates, price value that is the average of the previous and next biusiness day price
inputData["Price"] = (inputData["Price"].fillna(method='ffill') + inputData["Price"].fillna(method='bfill'))/2

# Normalizing price, and then adding powers
inputDataSimple = inputData[["Price"]]
inputData = (inputData[["Price"]] - minPrice)/ priceRange
inputData["P^0.5"] = inputData["Price"] ** 0.5
inputData["P^2"] = inputData["Price"] ** 2
inputData["P^3"] = inputData["Price"] ** 3
inputData["P^4"] = inputData["Price"] ** 4

limits = pd.Series(data=[minPrice, priceRange], index=["minPrice", "priceRange"])

if __debug__:
    print(inputData)
    print(limits)

inputData.to_csv("../data/processed/"+inputFile)
limits.to_csv("../data/processed/limits_"+inputFile)
inputDataSimple.to_csv("../data/processed/simple_"+inputFile)