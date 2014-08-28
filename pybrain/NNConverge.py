import csv
import numpy as np

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import CrossValidator
from pybrain.tools.shortcuts import buildNetwork
import ml_metrics as metrics
from pybrain.structure.modules   import LSTMLayer
import argparse
from pybrain.supervised import RPropMinusTrainer

def nonZeroValues(actual, predicted):

    actual = np.array(actual)
    indexes = np.where(actual == 0)[0]

    actual = [v for i,v in enumerate(actual) if i not in frozenset(indexes)]
    predicted = [v for i,v in enumerate(predicted) if i not in frozenset(indexes)] 
    
    return (actual, predicted)

def mape(actual, predicted):
    """
    Computes the mean absolute percentage error.

    This function computes the mean absolute percentage error between two lists
    of numbers. 
    BE CAREFUL: it can cause division-by-zero errors!

    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value

    Returns
    -------
    score : double
            The mean absolute percentage error between actual and predicted

    """
    actual, predicted = nonZeroValues(actual, predicted)
    return np.mean(np.divide(np.abs(np.array(actual) - np.array(predicted)), np.array(actual))) * 100


def predict(n, toPredict, actualValues):
    """
    Fetches the clients whose name starts with the parameter term.

    Args:
    n : the pyBrain neural network
    toPredict : array of the input to predict
    actualValues: array of the output values

    Returns:
    predictedA : array with the predicted values
    actualA : array with the real values

    Raises:
    ?
    """
    predictedA = []
    actualA = []
    for i in xrange(len(toPredict)):
        predicted = n.activate(toPredict[i])
        actual = actualValues[i]

        predictedA.append(predicted)
        actualA.append(actual)
    return (predictedA, actualA)

def splitWithProportion(DS, proportion = 0.5):
        """Produce two new datasets, the first one containing the fraction given
by `proportion` of the samples."""
        leftIndices = set(range(0, int(len(DS)*proportion)))
        leftDs = DS.copy()
        leftDs.clear()
        rightDs = leftDs.copy()
        index = 0
        for sp in DS:
            if index in leftIndices:
                leftDs.addSample(*sp)
            else:
                rightDs.addSample(*sp)
            index += 1
        return leftDs, rightDs

#
# Argument parser
#
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('filename', type=str, help='the filename to parse')
parser.add_argument('--features', dest='nFeatures', type=int, default=10, help='sum the integers (default: 10)')
parser.add_argument('--neurons', dest='nNeurons', type=int, default=60, help='number of neurons in the hidden layer (default: 60)')
parser.add_argument('--epochs', dest='nEpochs', type=int, default=20, help='number of epochs to train the NN (default: 20)')
args = parser.parse_args()

#
# Configuration
#
filename = args.filename
nFeatures = args.nFeatures
nOutput = 1
nNeurons = args.nNeurons
nEpochs = args.nEpochs

#
# Read dataset
#
f = open(filename, 'r')
data = csv.reader(f, delimiter=",")

DS = SupervisedDataSet(nFeatures, nOutput)
count = 0
for sample in data:
    #
    # Discard the label row
    #
    count = count +1
    if sample[0] == 'gas [m3]':
        continue

    label, x = sample[0], sample[1:]
    
    #
    # Insert the sample into the Dataset
    #
    DS.appendLinked(x, label)


#
# Divide the dataset in training set and test set
#
DS, tstdata = DS.splitWithProportion( 0.75 )
#DS, tstdata = splitWithProportion(DS, 0.75 )
print "Number of training patterns: ", len(DS)
print "Input and output dimensions: ", DS.indim, DS.outdim
print "number of units in hidden layer: ", nNeurons

#
# Build network with
#
n = buildNetwork(nFeatures, nNeurons, nOutput, bias=True)
trainer = BackpropTrainer( n, dataset=DS, verbose=True, momentum=0.01, learningrate=0.001)
#trainer = RPropMinusTrainer( n, dataset=DS, verbose=True )
#
# Training graph
#
graph = [("training", "test")]
a,b=predict(n, tstdata['input'], tstdata['target'])
bestValError = metrics.rmse(a,b)
epochsThreshold = 12
epochsCount = 0
bestweights = trainer.module.params.copy()
i = 0
while True:
    trainer.trainEpochs(1)

    predictedA, actualA = predict(n, DS['input'], DS['target'])
    trainingError = metrics.rmse(actualA, predictedA)
    predictedA, actualA = predict(n, tstdata['input'], tstdata['target'])
    validationError = metrics.rmse(actualA, predictedA)

    if validationError > bestValError:
        epochsCount = epochsCount+1
    else:
        bestValError = validationError
        bestweights = trainer.module.params.copy()
        epochsCount = 0

    if epochsCount >= epochsThreshold:
        trainer.module.params[:] = bestweights
        break

    graph.append((i, trainingError, validationError))

    i = i+1

with open('results/graphs/'+filename, 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(graph)


#
# Write the output of the final network
#
predictedA, actualA = predict(n, tstdata['input'], tstdata['target'])

print "EPOCHS: ", i-epochsThreshold
print "MAPE: ", mape(actualA, predictedA)
print "RMSE: ", metrics.rmse(actualA, predictedA)
print "MAE: ", metrics.mae(actualA, predictedA)

data = [["actual", "predicted"]]
data.extend(np.hstack([actualA, predictedA]))

with open('results/'+filename, 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(data)



