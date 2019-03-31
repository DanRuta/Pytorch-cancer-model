import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Confusion Matrix
from torchnet import meter


class Model(nn.Module):

    def __init__(self, topology, lr, optimFn="SGD"):
        super(Model, self).__init__()

        self.layers = []
        self.numAttributes = topology[0]
        self.numClasses = topology[len(topology)-1]

        for i in range(1, len(topology)):
            self.layers.append(("FC {}".format(i+1), nn.Linear(topology[i-1], topology[i])))

        self.model = nn.Sequential(collections.OrderedDict(self.layers))
        self.criterion = nn.CrossEntropyLoss()
        if optimFn=="SGD":
            self.optimizer = getattr(optim, optimFn)(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            self.optimizer = getattr(optim, optimFn)(self.model.parameters(), lr=lr)
        self.testingErrors = []


    def forward(self, x):
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        return x

    def train(self, trainData, epochs, testData=None):

        self.bestEpoch = 1
        self.trainingErrors = []

        for epoch in range(epochs):

            # print("Epoch {}/{}".format(epoch+1, epochs))

            runningLoss = 0

            for d in range(len(trainData)):
                inputs = trainData[d][0:self.numAttributes]
                labels = trainData[d][self.numAttributes:self.numAttributes+self.numClasses]

                input, label = Variable(torch.Tensor([inputs])), Variable(torch.Tensor([labels]))

                self.optimizer.zero_grad()
                output = self.model(input)
                label = torch.Tensor([0.0 if labels[0]==1 else 1.0]).long()

                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                runningLoss += loss.item()

            self.trainingErrors.append(runningLoss/len(trainData))

            # Collect would-be test accuracy/errors, for a hypothetical model trained for this long
            # to make plotting easier, for a task in EXP-1
            if testData is not None:
                self.test(testData)

            # print("Epoch error: {}".format(runningLoss/len(trainData)))

    def test(self, testData):

        runningLoss = 0
        correct = 0
        total = len(testData)

        # For sklearn classification report
        self.correctLabels = []
        self.predictedLabels = []

        # Confusion Matrix
        conf_mat = meter.ConfusionMeter(self.numClasses, normalized=False)

        with torch.no_grad():

            for d in range(len(testData)):
                inputs = testData[d][0:self.numAttributes]
                labels = testData[d][self.numAttributes:self.numAttributes+self.numClasses]

                input, label = Variable(torch.Tensor([inputs])), Variable(torch.Tensor([labels]))

                output = self.model(input)
                _, predicted = torch.max(output.data, 1)

                label = torch.Tensor([0.0 if labels[0]==1 else 1.0]).long()

                self.correctLabels.append(label.item())
                self.predictedLabels.append(predicted.item())
                correct += (predicted == label).sum().item()

                loss = self.criterion(output, label)
                runningLoss += loss.item()

                # Add to confusion matrix
                confLabel = testData[d][self.numAttributes:self.numAttributes+self.numClasses]
                confLabel = Variable(torch.Tensor(confLabel))

                # Turn one hot vector into class index
                confLabelIndex = 0
                for c in range(len(confLabel)):
                    if confLabel[c] >= max(confLabel):
                        confLabelIndex = c
                confLabel = torch.Tensor([confLabelIndex])

                conf_mat.add(output.data, confLabel)


        # print("Test accuracy: {}".format(correct/total*100))
        self.testingErrors.append(runningLoss/total)
        self.conf_mat = conf_mat
        return 100 * correct / total

    def eval(self, sample):
        inputs = sample[0:self.numAttributes]
        output = self.model(Variable(torch.Tensor([inputs])))
        return output.data.tolist()[0]

