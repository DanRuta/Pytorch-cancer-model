
import sys
import math
import argparse

import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

from Model import Model
from helper import *


epochs = [4,8,16,32,64]
topologies = [2,8,16,32]
learningRates = [0.05, 0.01, 0.005, 0.001, 0.0005]
modelAverages = 30


def main(configRanges, debug, skip1, skip2, skip3, topology, bestEpochs, lr, bestAccuracy):

    if debug:
        print("DEBUG mode")
        epochs = [4,8]
        topologies = [2,8]
        learningRates = [0.001]
        modelAverages = 2
    else:
        [epochs, topologies, learningRates, modelAverages] = configRanges


    print("Loading data")
    trainData, testData = loadData()


    if not skip1:
        print("\nRunning Experiment 1")
        print("====================\n")
        topology, bestEpochs, lr, bestAccuracy = exp1(trainData, testData, epochs, topologies, learningRates, modelAverages)


    if not skip2:
        print("\nRunning Experiment 2")
        print("====================\n")
        exp2(trainData, testData, topology, bestEpochs, lr, modelAverages, bestAccuracy)

        topologyGroups = []
        for t in topologies:

            topologyGroup = []

            for e in epochs:
                print("Testing ensemble for Topology: {}, Epochs: {}".format(t, e))
                topologyGroup.append(exp2(trainData, testData, t, e, lr, modelAverages, bestAccuracy, True))

            topologyGroups.append(topologyGroup)

        plotEXP2Results(topologyGroups, bestAccuracy, topologies, epochs)

    if not skip3:
        print("\nRunning Experiment 3")
        print("====================\n")

        exp3(trainData, testData, topologies, epochs, learningRates, modelAverages, bestAccuracy)

    print("\nFINISHED.")



def exp1(trainData, testData, epochs, topologies, learningRates, modelAverages, backPropAlg="SGD", expNo=1):

    accuracyValues = []

    for e in epochs:

        print("Epoch search: {}".format(e))

        epochsAcc = []

        for t in topologies:

            print("Topology search: {}".format(t))

            topologyAcc = []

            for lr in learningRates:

                lrAccuracies = [[], [], [], [], []]

                for it in range(modelAverages):
                    model = Model([9, t, 2], lr, backPropAlg)
                    model.train(trainData, e, testData)
                    accuracy = model.test(testData)
                    lrAccuracies[0].append(accuracy)
                    lrAccuracies[1].append(model.trainingErrors)
                    lrAccuracies[2].append([model.correctLabels, model.predictedLabels]) # Classification report
                    lrAccuracies[3].append(model.conf_mat) # Confusion matrix
                    lrAccuracies[4].append(model.testingErrors)

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
        plt.savefig("./plots/EXP{}-{}.png".format(expNo, learningRates[lr]))



    # Plot the error graph
    bestEpoch = bestAccuracyConfigIndeces[0]
    bestTopology = bestAccuracyConfigIndeces[1]
    bestLR = bestAccuracyConfigIndeces[2]
    bestExpNo = bestAccuracyConfigIndeces[3]

    bestTrainingErrors = accuracyValues[bestEpoch][bestTopology][bestLR][1]
    averageErrors = [0 for er in bestTrainingErrors[0]]
    stdDeviations = [0 for er in bestTrainingErrors[0]]

    bestTestErrors = accuracyValues[bestEpoch][bestTopology][bestLR][4]
    averageErrorsTest = [0 for er in bestTrainingErrors[0]]
    stdDeviationsTest = [0 for er in bestTrainingErrors[0]]

    # Compute the standard deviation of the errors at this point, in every run
    for er in range(len(bestTrainingErrors[0])):

        errsAtThisPoint = []
        errsAtThisPointTest = []

        for runI in range(len(bestTrainingErrors)):
            errsAtThisPoint.append(bestTrainingErrors[runI][er])
            errsAtThisPointTest.append(bestTestErrors[runI][er])

        # Compute the average (mean) error at this point, in every run
        averageErrors[er] = sum(errsAtThisPoint) / len(errsAtThisPoint)
        averageErrorsTest[er] = sum(errsAtThisPointTest)/len(errsAtThisPointTest)

        # For each error value, substract the mean, and square the result
        stdDeviations[er] = [(err-averageErrors[er])**2 for err in errsAtThisPoint]
        stdDeviationsTest[er] = [(err-averageErrorsTest[er])**2 for err in errsAtThisPointTest]
        # Re-compute mean
        stdDeviations[er] = sum(stdDeviations[er]) / len(stdDeviations[er])
        stdDeviationsTest[er] = sum(stdDeviationsTest[er]) / len(stdDeviationsTest[er])
        # Square root it
        stdDeviations[er] = math.sqrt(stdDeviations[er])
        stdDeviationsTest[er] = math.sqrt(stdDeviationsTest[er])


    fig = plt.figure()
    xRange = np.linspace(1, len(averageErrors), len(averageErrors))
    plt.errorbar(xRange, averageErrors, yerr=stdDeviations, fmt="-o", label="Training")
    plt.errorbar(xRange, averageErrorsTest, yerr=stdDeviationsTest, fmt="-o", label="Test")
    plt.legend(loc="upper right")

    plt.xlabel("Epochs")
    plt.ylabel("Train/Test losses, with std.")
    plt.title("Losses for {} Epochs, 9-{}-2 Topology, and {} lr, using {}".format(bestAccuracyConfig[0], bestAccuracyConfig[1], bestAccuracyConfig[2], backPropAlg))
    plt.savefig("./plots/EXP{}-Errors.png".format(expNo))


    classReportVals = accuracyValues[bestEpoch][bestTopology][bestLR][2][bestExpNo]
    bestModelStats = "Best accuracy is: {:.4f}%\tStandard Deviation: {:.5f}".format(bestAccuracy, stdDeviations[len(stdDeviations)-1])
    bestModelConfigs = "Configs: Optimizer: {}\tEpochs: {}\tTopology: {}\tLearning rate: {}".format(backPropAlg, bestAccuracyConfig[0], bestAccuracyConfig[1], bestAccuracyConfig[2])
    reportText = bestModelStats + "\n" + bestModelConfigs + "\n"
    print(bestModelStats)
    print(bestModelConfigs)

    getMetrics(classReportVals[0], classReportVals[1], accuracyValues[bestEpoch][bestTopology][bestLR][3][bestExpNo], expNo, reportText)

    # plt.show()
    return bestAccuracyConfig[0], bestAccuracyConfig[1], bestAccuracyConfig[2], bestAccuracy



def exp2(trainData, testData, topology, epochs, lr, modelAverages, bestSingleAccuracy, resultsOnly=False, backPropAlg="SGD", expNo=2):

    runs = []

    for run in range(modelAverages):

        sys.stdout.write("\rRun #: {}".format(run+1))
        sys.stdout.flush()

        models = [] # The collection of models to be used in ensemble
        errors = [] # Collect errors at different counts of models in ensemble

        # for i in range(3):
        for i in range(1):
            model = Model([9, topology, 2], lr, backPropAlg)
            model.train(trainData, epochs)
            models.append(model)

        # Vote with 3 models in an ensemble
        errors.append(ensembleVote(models, testData))

        # Vote with 5 to 25 models in an ensemble
        for i in range(11):
            model1 = Model([9, topology, 2], lr, backPropAlg)
            model1.train(trainData, epochs)
            models.append(model1)

            model2 = Model([9, topology, 2], lr, backPropAlg)
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
        print("The best accuracy is {:.4f}%, for an ensemble of {} models.\n".format(bestAccuracy, bestAccuracyEV))
    else:
        print("The best accuracy is that of the single model, without an ensemble, at {:.4f}%.\n".format(bestSingleAccuracy))


    plt.plot(xRange, yVals)
    plt.plot([0, 25], [bestSingleAccuracy, bestSingleAccuracy], marker = 'o')
    plt.xlabel("Ensembles")
    plt.ylabel("Accuracy")
    plt.title("Ensemble vote accuracies vs single best")
    # plt.show()
    plt.savefig("./plots/EXP{}-EXP1-Ensemble.png".format(expNo))



def exp3(trainData, testData, topologies, epochs, learningRates, modelAverages, bestAccuracy):

    optims = ["Rprop", "Adadelta"]
    # Learning rate ranges specific to each optimizer
    optimLearningRates = [learningRates, [1.0]]

    for o in range(len(optims)):

        optim = optims[o]
        print("Running single model experiments for the {} optimizer.\n".format(optim))

        # Perform Experiment 1 with the new optimizer
        topology, bestEpochs, lr, accuracy = exp1(trainData, testData, epochs, topologies, optimLearningRates[o], modelAverages, optim, expNo="3-{}".format(optim))

        if accuracy > bestAccuracy:
            print("New best accuracy, for the {} optimizer: {:.4f}".format(optim, accuracy))
            print("Configs: Topology: 9-{}-2\tEpochs: {}\tLearning Rate: {}".format(topology, bestEpochs, lr))
            bestAccuracy = accuracy

        print("Running ensemble experiments for the {} optimizer.\n".format(optim))
        exp2(trainData, testData, topology, bestEpochs, lr, modelAverages, accuracy, False, optim, expNo="3-{}".format(optim))

        topologyGroups = []
        for t in topologies:

            topologyGroup = []

            for e in epochs:
                print("Testing ensemble for Topology: {}, Epochs: {}".format(t, e))
                topologyGroup.append(exp2(trainData, testData, t, e, lr, modelAverages, bestAccuracy, True, optim, expNo="3-{}".format(optim)))

            topologyGroups.append(topologyGroup)

        plotEXP2Results(topologyGroups, bestAccuracy, topologies, epochs, expNo="3-{}".format(optim))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--d", default=False, type=bool, help="Debug")
    parser.add_argument("--skip1", default=False, type=bool, help="Skip Experiment 1")
    parser.add_argument("--skip2", default=False, type=bool, help="Skip Experiment 2")
    parser.add_argument("--skip3", default=False, type=bool, help="Skip Experiment 3")
    parser.add_argument("--t", default=8, type=int, help="Default best topology from Experiment 1")
    parser.add_argument("--e", default=16, type=int, help="Default best epoch count from Experiment 1")
    parser.add_argument("--lr", default=0.001, type=int, help="Default best learning rate from Experiment 1")
    parser.add_argument("--ba", default=96.8386, type=int, help="Default best accuracy from Experiment 1")
    args = parser.parse_args()

    configRanges = [epochs, topologies, learningRates, modelAverages]

    main(configRanges, args.d, args.skip1, args.skip2, args.skip3, args.t, args.e, args.lr, args.ba)

