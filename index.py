
import math

import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

from Model import Model
from helper import *

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

                lrAccuracies = [[], [], [], []]
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


    classReportVals = accuracyValues[bestEpoch][bestTopology][bestLR][2][bestExpNo]
    bestModelStats = "Best accuracy is: {:.4f}%\tStandard Deviation: {:.5f}".format(bestAccuracy, stdDeviations[len(stdDeviations)-1])
    bestModelConfigs = "Configs: Epochs: {}\tTopology: {}\tLearning rate: {}".format(bestAccuracyConfig[0], bestAccuracyConfig[1], bestAccuracyConfig[2])
    reportText = bestModelStats + "\n" + bestModelConfigs + "\n"
    print(bestModelStats)
    print(bestModelConfigs)

    getMetrics(classReportVals[0], classReportVals[1], accuracyValues[bestEpoch][bestTopology][bestLR][3][bestExpNo], 1, reportText)

    # plt.show()
    return bestAccuracyConfig[0], bestAccuracyConfig[1], bestAccuracyConfig[2], bestAccuracy



def exp2(trainData, testData, topology, epochs, lr, bestSingleAccuracy, resultsOnly=False):

    averageXTimes = 2
    runs = []

    for run in range(averageXTimes):

        sys.stdout.write("\rRun #: {}".format(run+1))
        sys.stdout.flush()

        models = [] # The collection of models to be used in ensemble
        errors = [] # Collect errors at different counts of models in ensemble

        # for i in range(3):
        for i in range(1):
            model = Model([9, topology, 2], lr)
            model.train(trainData, epochs)
            models.append(model)

        # Vote with 3 models in an ensemble
        errors.append(ensembleVote(models, testData))

        # Vote with 5 to 25 models in an ensemble
        for i in range(11):
            model1 = Model([9, topology, 2], lr)
            model1.train(trainData, epochs)
            models.append(model1)

            model2 = Model([9, topology, 2], lr)
            model2.train(trainData, epochs)
            models.append(model2)

            errors.append(ensembleVote(models, testData))

        runs.append(errors)
    print("\n")


    xRange = np.linspace(3, len(runs[0])*2+1, len(runs[0]))
    xRange = [*[0], *xRange] # Prepend an index for the base-line accuracy of no ensemble

    # Average each ensemble accuracy value, across the the runs
    yVals = [0 for e in range(len(runs[0]))] # Start each ensemble value at 0
    for run in runs:
        for e in range(len(run)):
            yVals[e] += run[e]


    # Take averages
    bestAccuracy = -math.inf
    bestAccuracyEV = 0

    for y in range(len(yVals)):
        yVals[y] /= len(runs)

        if yVals[y] > bestAccuracy:
            bestAccuracy = yVals[y]
            bestAccuracyEV = 2 + y * 2

    yVals = [*[bestSingleAccuracy], *yVals] # Prepend the base-line accuracy of no ensemble


    if resultsOnly:
        return xRange, yVals


    if bestAccuracy > bestSingleAccuracy:
        print("Best accuracy is {:.4f}, for an ensemble of {} models.".format(bestAccuracy, bestAccuracyEV))
    else:
        print("The best accuracy is that of the single model, without an ensemble, at {:.4f}".format(bestSingleAccuracy))


    plt.plot(xRange, yVals)
    plt.plot([0, 25], [bestSingleAccuracy, bestSingleAccuracy], marker = 'o')
    plt.xlabel("Ensembles")
    plt.ylabel("Accuracy")
    plt.title("Ensemble vote accuracies vs single best")
    # plt.show()
    plt.savefig("./plots/EXP2-EXP1-Ensemble.png")






def exp3():
    exp1("SOMETHING, NOT LM") # TODO
    exp1("rprop")





if __name__ == "__main__":

    print("Loading data")
    trainData, testData = loadData()



    print("\nRunning Experiment 1")
    print("====================\n")
    topology, epochs, lr, bestAccuracy = exp1(trainData, testData)



    print("\nRunning Experiment 2")
    print("====================\n")
    exp2(trainData, testData, topology, epochs, lr, bestAccuracy)

    epochs = [4,8,16,32,64]

    topologies = [2,8,32]

    newBestModel = None # Accuracy, Ensembles, Topology, Epochs

    for t in topologies:

        topologyGroup = []

        for e in epochs:
            print("Testing ensemble for Topology: {}, Epochs: {}".format(t, e))
            topologyGroup.append(exp2(trainData, testData, t, e, lr, bestAccuracy, True))

        newBest = plotTopologyGroup(topologyGroup, bestAccuracy, [t,epochs], "./plots/EXP2-Ensemble-T{}-E{}.png".format(t, e))

        if newBest is not None:
            newBestModel = newBest

    # New best has been found
    if newBestModel is not None:

        printStr1 = "A new best configuration has been found, at an accuracy of {:.4f}%, over {:.4f}%, ".format(newBestModel[0], bestAccuracy)
        printStr2 = "for an ensemble of {} networks, of topology: 9-{}-2, trained for {} epochs.".format(newBestModel[1], topologies[newBestModel[2]], newBestModel[3])
        print("{}{}".format(printStr1, printStr2))


