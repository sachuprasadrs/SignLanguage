# Importing the Libraries Required

import os
import string

# Creating the directory Structure

if not os.path.exists("dataSet"):
    os.makedirs("dataSet")
    print("Created directory: dataSet/")

if not os.path.exists("dataSet/trainingData"):
    os.makedirs("dataSet/trainingData")
    print("Created directory: dataSet/trainingData/")

if not os.path.exists("dataSet/testingData"):
    os.makedirs("dataSet/testingData")
    print("Created directory: dataSet/testingData/")

# Making folder 0 (i.e blank) in the training and testing data folders respectively
# The original code used 'range(0)' which creates no folders, but the main app uses 
# a folder named '0' for the blank sign. We should create it.

if not os.path.exists("dataSet/trainingData/0"):
    os.makedirs("dataSet/trainingData/0")
    print("Created directory: dataSet/trainingData/0 (for blank sign)")

if not os.path.exists("dataSet/testingData/0"):
    os.makedirs("dataSet/testingData/0")
    print("Created directory: dataSet/testingData/0 (for blank sign)")

# Making Folders from A to Z in the training and testing data folders respectively

for i in string.ascii_uppercase:
    if not os.path.exists("dataSet/trainingData/" + i):
        os.makedirs("dataSet/trainingData/" + i)
        print(f"Created directory: dataSet/trainingData/{i}")
    
    if not os.path.exists("dataSet/testingData/" + i):
        os.makedirs("dataSet/testingData/" + i)
        print(f"Created directory: dataSet/testingData/{i}")

print("\nDataset folder structure complete.")
print("Run TrainingDataCollection.py and TestingDataCollection.py to build your dataset.")
