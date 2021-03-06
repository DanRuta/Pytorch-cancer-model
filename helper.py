
import math
import numpy as np
import warnings

# Confusion Matrix
from torchnet import meter
import pandas as pd
import seaborn as sn

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from sklearn.metrics import classification_report

CLASSES = ["Benign", "Malignant"]
warnings.filterwarnings("ignore")


def ensembleVote(models, testData):

    correctLabels = []
    votes = [] # All the individual votes
    majorityVotes = [] # The maximum vote of all ensemble models

    for sample in testData:

        sampleVotes = []
        sampleLabels = sample[models[0].numAttributes:models[0].numAttributes+models[0].numClasses]
        correctLabels.append(sampleLabels.index(max(sampleLabels))) # Collect class indeces from one-hot vectors

        for m in models:
            output = m.eval(sample) # Vector
            sampleVotes.append(output.index(max(output))) # Class vote

        votes.append(sampleVotes)

    # Compute majority vote
    for v in range(len(votes)):
        sampleVotes = votes[v]
        # Create an array to hold counters for each present number in the list
        voteCounts = [0 for i in range(max(sampleVotes)+1)]
        for vote in sampleVotes:
            voteCounts[vote] += 1

        majorityVotes.append(voteCounts.index(max(voteCounts)))


    # Calculate the accuracy of the majority vote ensemble
    correct = 0

    for i in range(len(majorityVotes)):
        if majorityVotes[i]==correctLabels[i]:
            correct += 1

    return 100 * correct / len(majorityVotes), [correctLabels, majorityVotes]



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
    plotConfMatrix(conf_mat.value(), "./plots/EXP{}-Conf_n.png".format(experimentNo)) # Normalized


def plotConfMatrix(conf, outputPath):
    df_cm = pd.DataFrame(conf, CLASSES, CLASSES)
    plt.figure()
    sn.set(font_scale=1.5)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 20}, fmt='g')
    plt.savefig(outputPath)
    sn.reset_orig()


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


def plotTopologyGroup(topologies, topologyGroup, bestAccuracy, configs, path):

    topology, epochs = configs

    points3D = [[],[],[]]
    newBestFound = False
    newBestAccuracy = -math.inf
    newBestEnsemble = None
    newBestCRData = None
    newBestTopologyI = None
    newBestEpochs = None

    for tp in range(len(topologyGroup)):

        for v in range(len(topologyGroup[tp][0])):
            points3D[0].append(topologyGroup[tp][0][v]) # Ensembles
            points3D[1].append(epochs[tp]) # Epochs
            points3D[2].append(topologyGroup[tp][1][v]) # Accuracy

            if topologyGroup[tp][1][v] > bestAccuracy and topologyGroup[tp][1][v] > newBestAccuracy:
                newBestFound = True
                newBestAccuracy = topologyGroup[tp][1][v]
                newBestEnsemble = topologyGroup[tp][0][v]
                newBestCRData = topologyGroup[tp][2][v]
                newBestTopologyI = tp
                newBestEpochs = epochs[tp]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_facecolor((1.0, 1.0, 1.0))
    ax.set_xlabel("Ensembles")
    ax.set_ylabel("Epochs")
    ax.set_zlabel("Accuracy")
    plt.title("Ensemble vote accuracies for Topology: 9-{}-2".format(topologies[topology]))

    xRange = np.linspace(0, 25, 25)
    yRange = np.linspace(epochs[0], epochs[len(epochs)-1], len(epochs))

    X, Y = np.meshgrid(xRange, yRange)
    Z = griddata((points3D[0], points3D[1]), points3D[2], (X, Y), method="cubic")

    ax.plot_trisurf(points3D[0], points3D[1], np.array([bestAccuracy for i in range(len(points3D[2]))]), color="#ddaa0050")
    ax.plot_surface(X, Y, Z, lw=0.5, cmap=cm.coolwarm)

    plt.savefig(path)

    if newBestFound:
        return newBestAccuracy, newBestEnsemble, newBestCRData, newBestTopologyI, newBestEpochs


def plotEXP2Results(topologyGroups, bestAccuracy, topologies, epochs, expNo=2):

    newBestModel = None # Accuracy, Ensembles, Topology, Epochs

    for t in range(len(topologyGroups)):

        topologyGroup = topologyGroups[t]
        newBest = plotTopologyGroup(topologies, topologyGroup, bestAccuracy, [t,epochs], "./plots/EXP{}-Ensemble-Topology 9-{}-2.png".format(expNo, t))

        if newBest is not None:
            newBestModel = newBest

    # New best has been found
    if newBestModel is not None:
        printStr1 = "A new best configuration has been found, at an accuracy of {:.4f}%, over {:.4f}%, ".format(newBestModel[0], bestAccuracy)
        printStr2 = "for an ensemble of {} networks, of topology: 9-{}-2, trained for {} epochs.".format(newBestModel[1], topologies[newBestModel[3]], newBestModel[4])
        reportText = printStr1 + "\n" + printStr2 + "\n"
        print("{}{}".format(printStr1, printStr2))

        conf_mat = meter.ConfusionMeter(2)
        conf_mat.add(Variable(torch.Tensor(newBestModel[2][1])), Variable(torch.Tensor(newBestModel[2][0])))

        getMetrics(newBestModel[2][0], newBestModel[2][1], conf_mat, "{}-Ensemble".format(expNo), reportText)
