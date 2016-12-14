import csv
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import metrics
import math

def tail(list):
    return list[-1]

def plotPredictibility(x_quantiles,classProbabilities):
    plt.plot(range(0,len(x_quantiles)),classProbabilities, 'ro')
    plt.xticks(range(0,len(x_quantiles)), x_quantiles, rotation='vertical')
    plt.show()

## Previuos Count Feature
def howFrequentIsFinalElementInSequence(sequence):
    result = 0
    for element in sequence[:-1]:
        if element == mostFrequentFinalElement:
            result += 1
    return result

## Maximum Value In Sequence Feature
def maximumValueInSequence(sequence):
    if sequence[:-1]:
        return max(list(map(int, sequence[:-1])))
    else:
        return None

# Are There Negative Values
def areThereNegativeValuesInSequence(sequence):
    matching = sum(1 for number in map(int, sequence[:-1]) if number < 0)
    if matching > 0:
        return True
    else:
        return False

## Are There Any Zeroes In Sequence Feature
def areThereAnyZeroesInSequence(sequence):
    if 0 in map(int, sequence[:-1]):
        return True
    else:
        return False

# If the Most frequent final element is even are most of the elements in the sequence even?
def evenOddMatchInSequence(sequence):
    matching = sum(1 for number in map(int, sequence[:-1]) if number % 2 == int(mostFrequentFinalElement) % 2)
    unmatching = len(sequence[:-1]) - matching
    if matching > unmatching:
        return True
    else:
        return False

# Create Bucketed Version of a feature
def createBucketedVersionOfFeature(features,newFeatureName,featureName):
    features[newFeatureName] = "First Level"
    features[newFeatureName][(features[featureName] >= features[featureName].quantile(.25)) & (features[featureName]< features[featureName].quantile(.5))] = "Second Level"
    features[newFeatureName][(features[featureName] >= features[featureName].quantile(.5)) & (features[featureName]< features[featureName].quantile(.75))] = "Third Level"
    features[newFeatureName][(features[featureName] >= features[featureName].quantile(.75))] = "Fourth Level"
    features[newFeatureName] = features[newFeatureName].astype("category", categories=["First Level","Second Level", "Third Level","Fourth Level"], ordered=True)


### Load train data and split sequences string into a Pandas series
train = pd.read_csv('inputs/train.csv')
sequences = train.Sequence.str.split(',')

# Classifier with Most Frequest Final Element
finalElements = sequences.apply(tail)
mostFrequentFinalElement = finalElements.value_counts().index.tolist()[0]

random.seed(123)

#Create training and test datasets
trainingIndices = random.sample(finalElements[finalElements==mostFrequentFinalElement].index,sum(finalElements==mostFrequentFinalElement)//2) + random.sample(finalElements[finalElements!=mostFrequentFinalElement].index,sum(finalElements!=mostFrequentFinalElement)//2)
testIndices = list(set(range(1,len(sequences))) - set(trainingIndices))

### Create Classifier and Features for Training set

# Create features dataframe with all the features
features = pd.DataFrame({'previousCount':sequences[trainingIndices].apply(howFrequentIsFinalElementInSequence),
                       'negatives':sequences[trainingIndices].apply(areThereNegativeValuesInSequence),
                       'zeroes':sequences[trainingIndices].apply(areThereAnyZeroesInSequence),
                       'max':sequences[trainingIndices].apply(maximumValueInSequence),
                       'evenOddMatch' : sequences[trainingIndices].apply(evenOddMatchInSequence),
                       'featuresClass':finalElements[trainingIndices] == mostFrequentFinalElement,
                       })

# Bucketed 'max' feature
createBucketedVersionOfFeature(features,'maxBucketed','max')

# Featrues Class feature (this is the dependent variable)
features['featuresClass'] = finalElements[trainingIndices] == mostFrequentFinalElement

### Evaluate predictibility of each feature plotting the probabilities of each to predict the last element in sequence

# How predictive is previousCount feature?
def numberOfOcurrencesOfElementInSequence(count):
    return (finalElements[trainingIndices][features['previousCount']==count]==mostFrequentFinalElement).mean()

maxCountOfElement = features['previousCount'].max()
classProbabilities = pd.Series(range(0,maxCountOfElement)).apply(numberOfOcurrencesOfElementInSequence)
#plotPredictibility(range(0,maxCountOfElement),classProbabilities)

# How predictive is max feature?
# Bucketing is made for the sake of visualization
def finalElementsByBucket(elements):
    return (elements == mostFrequentFinalElement).mean()

buckets = pd.cut(map(int, features['max'].dropna()),features['max'].dropna().quantile(np.arange(0,1.1,0.1)))
classProbabilities = finalElements[trainingIndices][5:].dropna().groupby(buckets).apply(finalElementsByBucket)
#plotPredictibility(buckets.categories,classProbabilities)

# How predictive is negatives feature?
classProbabilities = [(finalElements[trainingIndices][features['negatives']]==mostFrequentFinalElement).mean(),(finalElements[trainingIndices][~features['negatives']]==mostFrequentFinalElement).mean()]
# plotPredictibility(range(0,len(classProbabilities)),classProbabilities)

# How predictive is evenOddMatch feature?
classProbabilities = [(finalElements[trainingIndices][features['evenOddMatch']]==mostFrequentFinalElement).mean(),(finalElements[trainingIndices][~features['evenOddMatch']]==mostFrequentFinalElement).mean()]
# plotPredictibility(range(0,len(classProbabilities)),classProbabilities)

# Fit the model using a logistic regression
fit = smf.glm(formula="featuresClass ~ previousCount + maxBucketed + negatives + zeroes + evenOddMatch", family=sm.families.Binomial(), data=features).fit()
print fit.summary()

## Create Classifier and Features for Test set

# Create features dataframe with all the features
features = pd.DataFrame({'previousCount':sequences[testIndices].apply(howFrequentIsFinalElementInSequence),
                       'negatives':sequences[testIndices].apply(areThereNegativeValuesInSequence),
                       'zeroes':sequences[testIndices].apply(areThereAnyZeroesInSequence),
                       'max':sequences[testIndices].apply(maximumValueInSequence),
                       'evenOddMatch' : sequences[testIndices].apply(evenOddMatchInSequence),
                       'featuresClass':finalElements[testIndices] == mostFrequentFinalElement,
                       })

# Bucketed 'max' feature
createBucketedVersionOfFeature(features,'maxBucketed','max')

# make prediction based on the fit date
predictions = fit.predict(features)

# Get AUC ROC for the prediction
prediction = (1 - metrics.roc_auc_score(features['featuresClass'],predictions))

print prediction
