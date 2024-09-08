import glob
from itertools import chain
import pandas as pd
import statistics as stat
import numpy as np
import csv
import math
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split
import os.path
from sklearn.metrics import accuracy_score
import random

paper = r"https://ieeexplore.ieee.org/document/9430501"

class Classifier:
    svm = "svm"
    decisionTree = "decisionTree"


actions = ["walk", "run", "jump", "sit"]
features = ["Accelerometer", "Gyroscope"] # GravitySensor, LinearAccelerationSensor
path = "Data\\self\\dataset\\raw\\"
ftype = ".csv"

classifier = Classifier.decisionTree
FeatureFusion = False
TestingIteration = 1000

new_list = []
for action_index, action in enumerate(actions):
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
                    # print('*' * 20)
                    # print(f"Path: {file_name}")
                    # print(f"Id: {id}")
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



                    # print(f"{action} : {feature} : Mean : x-{stat.mean(x)} y-{stat.mean(y)} z-{stat.mean(z)}")
                    # print(f"{action} : {feature} : Standard Deviation : x-{stat.stdev(x)} y-{stat.stdev(y)} z-{stat.stdev(z)}")
                    # print(f"{action} : {feature} : Median : x-{stat.median(x)} y-{stat.median(y)} z-{stat.median(z)}")
                    # print(f"{action} : {feature} : Min : x-{min(x)} y-{min(y)} z-{min(z)}")
                    # print(f"{action} : {feature} : Max : x-{max(x)} y-{max(y)} z-{max(z)}")
                    # print(f"{action} : {feature} : Range : x-{(max(x)-min(x))} y-{(max(y)-min(y))} z-{(max(z)-min(z))}")
                    xq3, xq1 = np.percentile(x, [75 ,25])
                    yq3, yq1 = np.percentile(y, [75 ,25])
                    zq3, zq1 = np.percentile(z, [75 ,25])
                    xiqr = xq3 - xq1
                    yiqr = yq3 - yq1
                    ziqr = zq3 - zq1
                    # print(f"{action} : {feature} : Interquartile Range : x-{xiqr} y-{yiqr} z-{ziqr}")
                    # print('*' * 20)

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
                        exists[feature + " Mean"] =  math.sqrt(pow(stat.mean(x), 2) + pow(stat.mean(y), 2) + pow(stat.mean(z), 2))
                        exists[feature + " StandardDeviation"] =  math.sqrt(pow(stat.stdev(x), 2) + pow(stat.stdev(y), 2) + pow(stat.stdev(z), 2))
                        exists[feature + " Median"] =  math.sqrt(pow(stat.median(x), 2) + pow(stat.median(y), 2) + pow(stat.median(z), 2))
                        exists[feature + " Range"] =  math.sqrt(pow(max(x)-min(x), 2) + pow(max(y)-min(y), 2) + pow(max(z)-min(z), 2))
                        exists[feature + " InterquartileRange"] =  math.sqrt(pow(xq3 - xq1, 2) + pow(yq3 - yq1, 2) + pow(zq3 - zq1, 2))
                    else: 
                        _new = {}
                        _new["id"] = id
                        _new["action"] = action_index
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
                        _new[feature + " Mean"] =  math.sqrt(pow(stat.mean(x), 2) + pow(stat.mean(y), 2) + pow(stat.mean(z), 2))
                        _new[feature + " StandardDeviation"] =  math.sqrt(pow(stat.stdev(x), 2) + pow(stat.stdev(y), 2) + pow(stat.stdev(z), 2))
                        _new[feature + " Median"] =  math.sqrt(pow(stat.median(x), 2) + pow(stat.median(y), 2) + pow(stat.median(z), 2))
                        _new[feature + " Range"] =  math.sqrt(pow(max(x)-min(x), 2) + pow(max(y)-min(y), 2) + pow(max(z)-min(z), 2))
                        _new[feature + " InterquartileRange"] =  math.sqrt(pow(xq3 - xq1, 2) + pow(yq3 - yq1, 2) + pow(zq3 - zq1, 2))
                        
                        
                        new_list.append(_new)
            except pd.errors.EmptyDataError:
                # print(f"The file {file_name} is empty, skipping...")
                print("")

# Specify the filename
filename = path + "extractedFeatures.csv"

# Writing to the CSV file
if not os.path.isfile(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=new_list[0].keys())
        
        # Write the header (column names)
        writer.writeheader()
        
        # Write the rows
        writer.writerows(new_list)

    print(f"Data has been written to {filename}")

columns = []
xyzList = []
if FeatureFusion:
    xyzList = [""]
else:
    xyzList = ["X", "Y", "Z"]

for xyzData in xyzList:
    for feature in features:
        columns.append(feature + " Mean" + xyzData)
        columns.append(feature + " StandardDeviation" + xyzData)
        columns.append(feature + " Median" + xyzData)
        columns.append(feature + " Range" + xyzData)
        columns.append(feature + " InterquartileRange" + xyzData)

# print(columns)

# print(new_list)
# new_list = new_list * 5
# random.shuffle(new_list)
OverallAccuracy = 0
storedList = new_list
if TestingIteration < 1: 
    TestingIteration = 1
for i in range(TestingIteration):
    new_list = storedList
    x_vals = [[data[column] for column in columns] for data in new_list]
    y_vals = [[data['action']] for data in new_list]

    # print(x_vals)
    # print(y_vals)

    X_train, X_test, y_train, y_test = train_test_split(x_vals, y_vals, test_size=0.33, random_state=42)
    xy_lists = [X_train, X_test, y_train, y_test]
    multiplier = 5
    duplicatedList = [lst * multiplier for lst in xy_lists]
    X_train, X_test, y_train, y_test = duplicatedList

    print(f"Trained Sets: {len(X_train)}")
    print(f"Tested Sets: {len(X_test)}")

    y_predicted = None
    if classifier == Classifier.svm:
        clf = svm.SVC()
        clf.fit(X_train, np.array(y_train).ravel())

        y_predicted = clf.predict(X_test)

    if classifier == Classifier.decisionTree:
        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train, np.array(y_train).ravel())

        y_predicted = clf.predict(X_test)

    print(f"Actions:   {actions}")
    print(f"Predicted: {list(y_predicted)}")
    print(f"True Value:{list(chain.from_iterable(y_test))}")

    accuracy = accuracy_score(y_test, y_predicted) * 100
    OverallAccuracy += accuracy
    print(f"\nAccuracy: {accuracy}%")

OverallAccuracy = OverallAccuracy / TestingIteration
print(f"\nClassifier: {classifier}")
print(f"Testing Iteration: {TestingIteration} times")
print(f"Overall Accuracy: {OverallAccuracy}%")
