import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import glob

def generateGraph(file_path, title, xlabel, ylabel): 
    data = pd.read_csv(file_path)
    row_count = data.shape[0]
    time = list(range(1, row_count + 1))
    x = data[data.columns[0]].to_numpy().tolist()
    y = data[data.columns[1]].to_numpy().tolist()
    z = data[data.columns[2]].to_numpy().tolist()

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot data
    plt.plot(time, x, label='X axis data')
    plt.plot(time, y, label='Y axis data')
    plt.plot(time, z, label='Z axis data')

    # Label the axes
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

# generateGraph('Data/self/Accelerometer.csv', 'Linear Graph of Acceleration of X, Y, Z axis data over Time', 'Frequency', 'Acceleration (m/s2)')
# generateGraph('Data/self/GravitySensor.csv', 'Linear Graph of GravitySensor of X, Y, Z axis data over Time', 'Frequency', 'GravitySensor (m/s2)')
# generateGraph('Data/self/Gyroscope.csv', 'Linear Graph of Gyroscope of X, Y, Z axis data over Time', 'Frequency', 'Gyroscope (rad/s)')
# generateGraph('Data/self/LinearAccelerationSensor.csv', 'Linear Graph of LinearAccelerationSensor of X, Y, Z axis data over Time', 'Frequency', 'LinearAccelerationSensor (m/s2)')
    
    # Loop through all files starting with walk_a_
for file_a in glob.glob("Data\self\dataset\raw\runLinearAccelerationSensor_*.csv"):
    # print(f"Processing file: {file_a}")
    generateGraph(file_a, 'Linear Graph of LinearAccelerationSensor of X, Y, Z axis data over Time', 'Frequency', 'Acceleration (m/s2)')
    # Add your processing logic here