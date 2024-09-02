import glob
import pandas as pd
import statistics as stat
import numpy as np
import csv


actions = ["walk", "run", "jump", "sit"]
features = ["Accelerometer", "Gyroscope"] # GravitySensor, LinearAccelerationSensor
path = "Data\\self\\dataset\\raw\\"
ftype = ".csv"

new_list = []
for action in actions:
    for feature in features:
        for file_name in glob.glob(path + action + feature + "*" + ftype):
            # print(f"Processing file: {file_name}")
            try:
                data = pd.read_csv(file_name)
                if data.empty:
                    # print("The file is empty, skipping...")
                    print("")
                else: 
                    id = file_name[file_name.rfind('_') + 1:file_name.rfind('.')]
                    print('*' * 20)
                    print(f"Path: {file_name}")
                    print(f"Id: {id}")
                    row_count = data.shape[0]
                    time = list(range(1, row_count + 1))
                    x = data[data.columns[0]].to_numpy().tolist()
                    y = data[data.columns[1]].to_numpy().tolist()
                    z = data[data.columns[2]].to_numpy().tolist()


                    overallMin = min([min(x),min(y),min(z)])
                    overallMax = max([max(x),max(y),max(z)])
                    overallRange = overallMax - overallMin
                    x = [ ((_x - overallMin) / overallRange) for _x in x]
                    y = [ ((_x - overallMin) / overallRange) for _x in y]
                    z = [ ((_x - overallMin) / overallRange) for _x in z]



                    print(f"{action} : {feature} : Mean : x-{stat.mean(x)} y-{stat.mean(y)} z-{stat.mean(z)}")
                    print(f"{action} : {feature} : Standard Deviation : x-{stat.stdev(x)} y-{stat.stdev(y)} z-{stat.stdev(z)}")
                    print(f"{action} : {feature} : Median : x-{stat.median(x)} y-{stat.median(y)} z-{stat.median(z)}")
                    print(f"{action} : {feature} : Min : x-{min(x)} y-{min(y)} z-{min(z)}")
                    print(f"{action} : {feature} : Max : x-{max(x)} y-{max(y)} z-{max(z)}")
                    print(f"{action} : {feature} : Range : x-{(max(x)-min(x))} y-{(max(y)-min(y))} z-{(max(z)-min(z))}")
                    xq3, xq1 = np.percentile(x, [75 ,25])
                    yq3, yq1 = np.percentile(y, [75 ,25])
                    zq3, zq1 = np.percentile(z, [75 ,25])
                    xiqr = xq3 - xq1
                    yiqr = yq3 - yq1
                    ziqr = zq3 - zq1
                    print(f"{action} : {feature} : Interquartile Range : x-{xiqr} y-{yiqr} z-{ziqr}")
                    print('*' * 20)

                    exists = next((item for item in new_list if item["id"] == id), None)
                    if exists:
                        xq3, xq1 = np.percentile(x, [75 ,25])
                        yq3, yq1 = np.percentile(y, [75 ,25])
                        zq3, zq1 = np.percentile(z, [75 ,25])
                        exists[feature + " MeanX"] = stat.mean(x)
                        exists[feature + " MeanY"] = stat.mean(y)
                        exists[feature + " MeanZ"] = stat.mean(z)
                        exists[feature + " StandardDeviationX"] = stat.stdev(x)
                        exists[feature + " StandardDeviationY"] = stat.stdev(y)
                        exists[feature + " StandardDeviationZ"] = stat.stdev(z)
                        exists[feature + " MedianX"] = stat.median(x)
                        exists[feature + " MedianY"] = stat.median(y)
                        exists[feature + " MedianZ"] = stat.median(z)
                        exists[feature + " MinX"] = min(x)
                        exists[feature + " MinY"] = min(y)
                        exists[feature + " MinZ"] = min(z)
                        exists[feature + " MaxX"] = max(x)
                        exists[feature + " MaxY"] = max(y)
                        exists[feature + " MaxZ"] = max(z)
                        exists[feature + " RangeX"] = max(x)-min(x)
                        exists[feature + " RangeY"] = max(y)-min(y)
                        exists[feature + " RangeZ"] = max(z)-min(z)
                        exists[feature + " InterquartileRangeX"] = xq3 - xq1
                        exists[feature + " InterquartileRangeY"] = yq3 - yq1
                        exists[feature + " InterquartileRangeZ"] = zq3 - zq1
                    else: 
                        _new = {}
                        _new["id"] = id
                        _new["action"] = action
                        xq3, xq1 = np.percentile(x, [75 ,25])
                        yq3, yq1 = np.percentile(y, [75 ,25])
                        zq3, zq1 = np.percentile(z, [75 ,25])
                        _new[feature + " MeanX"] = stat.mean(x)
                        _new[feature + " MeanY"] = stat.mean(y)
                        _new[feature + " MeanZ"] = stat.mean(z)
                        _new[feature + " StandardDeviationX"] = stat.stdev(x)
                        _new[feature + " StandardDeviationY"] = stat.stdev(y)
                        _new[feature + " StandardDeviationZ"] = stat.stdev(z)
                        _new[feature + " MedianX"] = stat.median(x)
                        _new[feature + " MedianY"] = stat.median(y)
                        _new[feature + " MedianZ"] = stat.median(z)
                        _new[feature + " MinX"] = min(x)
                        _new[feature + " MinY"] = min(y)
                        _new[feature + " MinZ"] = min(z)
                        _new[feature + " MaxX"] = max(x)
                        _new[feature + " MaxY"] = max(y)
                        _new[feature + " MaxZ"] = max(z)
                        _new[feature + " RangeX"] = max(x)-min(x)
                        _new[feature + " RangeY"] = max(y)-min(y)
                        _new[feature + " RangeZ"] = max(z)-min(z)
                        _new[feature + " InterquartileRangeX"] = xq3 - xq1
                        _new[feature + " InterquartileRangeY"] = yq3 - yq1
                        _new[feature + " InterquartileRangeZ"] = zq3 - zq1
                        
                        
                        new_list.append(_new)
            except pd.errors.EmptyDataError:
                # print(f"The file {file_name} is empty, skipping...")
                print("")

# Specify the filename
filename = path + "extractedFeatures.csv"

# Writing to the CSV file
with open(filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=new_list[0].keys())
    
    # Write the header (column names)
    writer.writeheader()
    
    # Write the rows
    writer.writerows(new_list)

print(f"Data has been written to {filename}")