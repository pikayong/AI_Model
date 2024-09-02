import glob
import pandas as pd
import statistics as stat
import numpy as np
import csv


# actions = ["walk", "run", "jump", "sit"]
# features = ["Accelerometer", "Gyroscope"]
# path = "Data\\self\\dataset\\raw\\"
# ftype = ".csv"

# new_list = []
# for action in actions:
#     for feature in features:
#         for file_name in glob.glob(path + action + feature + "*" + ftype):
#             # print(f"Processing file: {file_name}")
#             try:
#                 data = pd.read_csv(file_name)
#                 if data.empty:
#                     # print("The file is empty, skipping...")
#                     print("")
#                 else: 
#                     scaler = MinMaxScaler()
#                     scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
                    
#             except pd.errors.EmptyDataError:
#                 # print(f"The file {file_name} is empty, skipping...")
#                 print("")

# # Specify the filename
# filename = path + "extractedFeatures.csv"

# # Writing to the CSV file
# with open(filename, mode='w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=new_list[0].keys())
    
#     # Write the header (column names)
#     writer.writeheader()
    
#     # Write the rows
#     writer.writerows(new_list)

# print(f"Data has been written to {filename}")

x = [1, 2, 3]
y = [-5, 0, 4]
z = [2, 8, 1]

overallMin = min([min(x), min(y), min(z)])
overallMax = max([max(x), max(y), max(z)])
overallRange = overallMax - overallMin
x = [ ((_x - overallMin) / overallRange) for _x in x]
y = [ ((_y - overallMin) / overallRange) for _y in y]
z = [ ((_z - overallMin) / overallRange) for _z in z]
print(f"Overall Min: {overallMin}")
print(f"Overall Max: {overallMax}")
print(x)
print(y)
print(z)