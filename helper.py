
# Confusion Matrix
import pandas as pd
import seaborn as sn

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import classification_report

CLASSES = ["Benign", "Malignant"]


def getMetrics(correctLabels, predictedLabels, conf_mat, experimentNo=1, reportText=""):

    # Classification report
    report = classification_report(correctLabels, predictedLabels, target_names=CLASSES)
    print(report)

    with open("./plots/EXP{}-classificationReport.txt".format(experimentNo), "w+") as f:
        f.write(reportText)
        f.write(report)

    # Confusion matrix
    plotConfMatrix(conf_mat.value(), "./plots/EXP{}-Conf.png".format(experimentNo)) # Normal
    conf_mat.normalized = True
    plotConfMatrix(conf_mat.value(), "./plots/EXP{}-Conf_n.png".format(experimentNo)) # Normal


def plotConfMatrix(conf, outputPath):
    df_cm = pd.DataFrame(conf, CLASSES, CLASSES)
    plt.figure()
    sn.set(font_scale=1.5)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 20}, fmt='g')
    plt.savefig(outputPath)


def loadData():

    rawData = []

    # Load the data attributes
    with open("data.txt") as f:

        lines = f.read().split("\n")

        for line in lines:
            rawData.append([float(num) for num in line.split(",")])

    # Load and append the labels
    with open("labels.txt") as f:
        lines = f.read().split("\n")

        for line in lines:
            rawData.append([int(num) for num in line.split(",")])


    # Reshape data
    data = []
    numSamples = len(rawData[0])
    numAttributes = len(rawData)

    print("numSamples: {}".format(numSamples))
    print("numAttributes: {}".format(numAttributes))

    for s in range(numSamples):

        sample = []

        for a in range(numAttributes):
            sample.append(rawData[a][s])

        data.append(sample)


    # Split data
    trainData = []
    testData = []

    for s in range(numSamples):
        if s%2==0:
            trainData.append(data[s])
        else:
            testData.append(data[s])


    print("Training samples: {}".format(len(trainData)))
    print("Test samples: {}\n".format(len(testData)))
    return trainData, testData