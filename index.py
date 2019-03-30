
import math

import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

# Confusion Matrix
import pandas as pd
import seaborn as sn

from sklearn.metrics import classification_report

from Model import Model
CLASSES = ["Benign", "Malignant"]


def exp1(trainData, testData, backPropAlg="SGD"):

    averageAcrossXModels = 3
    # epochs = [4,8,16,32,64]
    # epochs = [4,8,16]
    epochs = [4,8]
    # topologies = [2,8,32]
    topologies = [2,8]
    # learningRates = [0.05, 0.01, 0.005, 0.001]
    learningRates = [0.01]

    accuracyValues = []

    for e in epochs:

        print("Epoch search: {}".format(e))

        epochsAcc = []

        for t in topologies:

            print("Topology search: {}".format(t))

            topologyAcc = []

            for lr in learningRates:

                lrAccuracies = [[], [], [], [], []]
                totalAccuracy = 0

                # for it in range(30):
                for it in range(averageAcrossXModels):
                    model = Model([9, t, 2], lr, backPropAlg)
                    model.train(trainData, e)
                    accuracy = model.test(testData)
                    totalAccuracy += accuracy
                    lrAccuracies[0].append(accuracy)
                    lrAccuracies[1].append(model.trainingErrors)
                    lrAccuracies[2].append([model.correctLabels, model.predictedLabels]) # Classification report
                    lrAccuracies[3].append(model.conf_mat) # Confusion matrix
                    lrAccuracies[4].append(model.conf_mat_n) # Confusion matrix (normalized)

                topologyAcc.append(lrAccuracies)

            epochsAcc.append(topologyAcc)

        accuracyValues.append(epochsAcc)


    bestAccuracy = -math.inf
    bestAccuracyConfig = [0, 0, 0]
    bestAccuracyConfigIndeces = [0, 0, 0, 0] # Epochs, topology, learning rate, experiment #


    for lr in range(len(learningRates)):

        points3D = [[],[],[]]

        for e in range(len(accuracyValues)):
            for t in range(len(accuracyValues[e])):

                averageAccuracy = sum(accuracyValues[e][t][lr][0]) / len(accuracyValues[e][t][lr][0])

                if averageAccuracy > bestAccuracy:
                    bestAccuracy = averageAccuracy
                    bestAccuracyConfig[0] = epochs[e]
                    bestAccuracyConfigIndeces[0] = e
                    bestAccuracyConfig[1] = topologies[t]
                    bestAccuracyConfigIndeces[1] = t
                    bestAccuracyConfig[2] = learningRates[lr]
                    bestAccuracyConfigIndeces[2] = lr

                points3D[0].append(epochs[e])
                points3D[1].append(topologies[t])
                points3D[2].append(averageAccuracy)

                bestLRAccuracy = accuracyValues[e][t][lr][0][0]
                for exp in range(len(accuracyValues[e][t][lr][0])):
                    if accuracyValues[e][t][lr][0][exp] >= bestLRAccuracy:
                        bestAccuracyConfigIndeces[3] = exp


        xScale = np.linspace(epochs[0], epochs[len(epochs)-1], epochs[len(epochs)-1]-epochs[0]) # Epochs
        yScale = np.linspace(topologies[0], topologies[len(topologies)-1], topologies[len(topologies)-1]-topologies[0]) # Topologies

        fig = plt.figure()
        ax = Axes3D(fig)

        # Non-smooth graph
        # scatter = ax.plot_trisurf(points3D[0], points3D[1], points3D[2], cmap=cm.coolwarm)

        # Smooth graph
        X, Y = np.meshgrid(xScale, yScale)
        z2 = griddata((points3D[0], points3D[1]), points3D[2], (X, Y), method="cubic")
        scatter = ax.plot_surface(X, Y, z2, cmap=cm.coolwarm)

        fig.suptitle("Average accuracies for learning rate={}".format(learningRates[lr]), fontsize=16)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Number of hidden nodes")
        ax.set_zlabel("Average accuracy")
        plt.savefig("./plots/EXP1-{}.png".format(learningRates[lr]))



    # Plot the error graph
    bestEpoch = bestAccuracyConfigIndeces[0]
    bestTopology = bestAccuracyConfigIndeces[1]
    bestLR = bestAccuracyConfigIndeces[2]
    bestExpNo = bestAccuracyConfigIndeces[3]

    bestTrainingErrors = accuracyValues[bestEpoch][bestTopology][bestLR][1]
    averageErrors = [0 for er in bestTrainingErrors[0]]
    stdDeviations = [0 for er in bestTrainingErrors[0]]


    # Compute the standard deviation of the errors at this point, in every run
    for er in range(len(bestTrainingErrors[0])):
        errsAtThisPoint = []

        for run in bestTrainingErrors:
            errsAtThisPoint.append(run[er])

        # Compute the average (mean) error at this point, in every run
        averageErrors[er] = sum(errsAtThisPoint) / len(errsAtThisPoint)

        # For each error value, substract the mean, and square the result
        stdDeviations[er] = [(err-averageErrors[er])**2 for err in errsAtThisPoint]
        # Re-compute mean
        stdDeviations[er] = sum(stdDeviations[er]) / len(stdDeviations[er])
        # Square root it
        stdDeviations[er] = math.sqrt(stdDeviations[er])


    fig = plt.figure()
    xRange = np.linspace(0, len(averageErrors), len(averageErrors))
    plt.errorbar(xRange, averageErrors, yerr=stdDeviations, fmt='-o')
    plt.xlabel("Epochs")
    plt.ylabel("Training Error, with std.")
    plt.title("Training errors for {} Epochs, 9-{}-2 Topology, and {} Learning Rate".format(bestAccuracyConfig[0], bestAccuracyConfig[1], bestAccuracyConfig[2]))
    plt.savefig("./plots/EXP1-Errors.png")


    # Classification report
    classReportVals = accuracyValues[bestEpoch][bestTopology][bestLR][2][bestExpNo]
    report = classification_report(classReportVals[0], classReportVals[1], target_names=CLASSES)

    bestModelStats = "Best accuracy is: {}%\tStandard Deviation: {}".format(bestAccuracy, stdDeviations[len(stdDeviations)-1])
    bestModelConfigs = "Configs: Epochs: {}\tTopology: {}\tLearning rate: {}".format(bestAccuracyConfig[0], bestAccuracyConfig[1], bestAccuracyConfig[2])
    print(bestModelStats)
    print(bestModelConfigs)
    print(report)

    with open("./plots/EXP1-classificationReport.txt", "w+") as f:
        f.write(bestModelStats + "\n")
        f.write(bestModelConfigs + "\n\n")
        f.write(report)

    # Confusion matrix
    plotConfMatrix(accuracyValues[bestEpoch][bestTopology][bestLR][3][bestExpNo], "./plots/EXP1-Conf.png") # Normal
    plotConfMatrix(accuracyValues[bestEpoch][bestTopology][bestLR][4][bestExpNo], "./plots/EXP1-Conf_n.png") # Normalized
    # plt.show()

    return bestAccuracyConfig[0], bestAccuracyConfig[1], bestAccuracyConfig[2]



def exp2(trainData, testData, topology, epochs, lr):

    models = []
    errors = []

    for i in range(3):
        model = Model([9, topology, 1], epochs)
        model.train()
        models.append(model)

    errors.append(ensembleVote(models))

    # 3 to 25
    for i in range(11):
        model1 = Model([9, topology, 1])
        model1.train(epochs)
        models.append(model1)

        model2 = Model([9, topology, 1])
        model2.train(epochs)
        models.append(model2)

        errors.append(ensembleVote(models))


def ensembleVote(models):

    votes = []

    for m in models:
        vote = m.eval

    pass


def exp3():
    exp1("SOMETHING, NOT LM") # TODO
    exp1("rprop")


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


if __name__ == "__main__":

    print("Loading data")
    trainData, testData = loadData()

    print("\nRunning Experiment 1")
    print("====================\n")
    topology, epochs, lr = exp1(trainData, testData)


    # exp2(trainData, testData, topology, epochs, lr)


