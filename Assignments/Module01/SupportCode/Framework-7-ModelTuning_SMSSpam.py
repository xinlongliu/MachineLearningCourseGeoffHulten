import os

kOutputDirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualize')

import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

(xRaw, yRaw) = SMSSpamDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation

import time

# A helper function for calculating FN rate and FP rate across a range of thresholds
def TabulateModelPerformanceForROC(model, xValidate, yValidate):
   pointsToEvaluate = 100
   thresholds = [ x / float(pointsToEvaluate) for x in range(pointsToEvaluate + 1)]
   FPRs = []
   FNRs = []
   yPredicted = model.predictProbabilities(xValidate)

   try:
      for threshold in thresholds:
        yHats = [ 1 if pred > threshold else 0 for pred in yPredicted ]
        FPRs.append(EvaluateBinaryClassification.FalsePositiveRate(yValidate, yHats))
        FNRs.append(EvaluateBinaryClassification.FalseNegativeRate(yValidate, yHats))
   except NotImplementedError:
      raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

   return (FPRs, FNRs, thresholds)

import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
## This function will help you plot with error bars. Use it just like PlotSeries, but with parallel arrays of error bar sizes in the second variable
#     note that the error bar size is drawn above and below the series value. So if the series value is .8 and the confidence interval is .78 - .82, then the value to use for the error bar is .02

# Charting.PlotSeriesWithErrorBars([series1, series2], [errorBarsForSeries1, errorBarsForSeries2], ["Series1", "Series2"], xValues, chartTitle="<>", xAxisTitle="<>", yAxisTitle="<>", yBotLimit=0.8, outputDirectory=kOutputDirectory, fileName="<name>")


## This helper function should execute a single run and save the results on 'runSpecification' (which could be a dictionary for convienience)
#    for later tabulation and charting...
def ExecuteEvaluationRun(runSpecification, xTrainRaw, yTrain, numberOfFolds = 5):
    startTime = time.time()
    
    # HERE upgrade this to use crossvalidation
    validationSetAccuracy = 0
    for i in range(numberOfFolds):
        xTrainRawCV, yTrainCV, xEvaluateRawCV, yEvaluateCV = CrossValidation.CrossValidation(xTrainRaw, yTrain,
                                                                                             numberOfFolds, i)

        featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
        featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords=runSpecification['numFrequentWords'],
                                    numMutualInformationWords=runSpecification['numMutualInformationWords'])
        xTrainCV = featurizer.Featurize(xTrainRawCV)
        xValidateCV = featurizer.Featurize(xEvaluateRawCV)
        model = LogisticRegression.LogisticRegression()
        model.fit(xTrainCV, yTrainCV, convergence=runSpecification['convergence'],
                  stepSize=runSpecification['stepSize'], verbose=True)
        validationSetAccuracy += EvaluateBinaryClassification.Accuracy(yEvaluateCV, model.predict(xValidateCV))
    validationSetAccuracy /= numberOfFolds
    runSpecification['accuracy'] = validationSetAccuracy

    # HERE upgrade this to calculate and save some error bounds...
    (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(validationSetAccuracy, len(xTrainRaw), 0.5)
    runSpecification['lowerBound'] = lowerBound
    runSpecification['upperBound'] = upperBound
    
    endTime = time.time()
    runSpecification['runtime'] = endTime - startTime
    
    return runSpecification

evaluationRunSpecifications = []
for numMutualInformationWords in [20, 40, 60, 80, 100, 120]:

    runSpecification = {}
    runSpecification['optimizing'] = 'numMutualInformationWords'
    runSpecification['numMutualInformationWords'] = numMutualInformationWords
    runSpecification['stepSize'] = 1.0
    runSpecification['convergence'] = 0.005
    runSpecification['numFrequentWords'] = 0
    
    evaluationRunSpecifications.append(runSpecification)

## if you want to run in parallel you need to install joblib as described in the lecture notes and adjust the comments on the next three lines...
from joblib import Parallel, delayed
evaluations = Parallel(n_jobs=12)(delayed(ExecuteEvaluationRun)(runSpec, xTrainRaw, yTrain, 2) for runSpec in evaluationRunSpecifications)

# evaluations = [ ExecuteEvaluationRun(runSpec, xTrainRaw, yTrain) for runSpec in evaluationRunSpecifications ]

acc, err, rt, mi = [], [], [], []
accBest, rtBest, best = 0, 9999, []
for evaluation in evaluations:
    print(evaluation)
    acc.append(evaluation['accuracy'])
    err.append(evaluation['accuracy'] - evaluation['lowerBound'])
    rt.append(evaluation['runtime'])
    mi.append(evaluation['numMutualInformationWords'])
    if evaluation['accuracy'] > accBest:
        accBest = evaluation['accuracy']
        rtBest = evaluation['runtime']
        best = evaluation
    elif evaluation['accuracy'] == accBest:
        if evaluation['runtime'] < rtBest:
            accBest = evaluation['accuracy']
            rtBest = evaluation['runtime']
            best = evaluation
Charting.PlotSeriesWithErrorBars([acc], [err], ["accuracy"], mi,
                                 chartTitle=best['optimizing'] + " - accuracy",
                                 xAxisTitle=best['optimizing'], yAxisTitle="accuracy",
                                 yBotLimit=0.8, outputDirectory=kOutputDirectory,
                                 fileName=best['optimizing'] + " - accuracy")
Charting.PlotSeries([rt], ['runtime'], mi, chartTitle=best['optimizing'] + " - runtime",
                    xAxisTitle=best['optimizing'], yAxisTitle="runtime", outputDirectory=kOutputDirectory,
                    fileName=best['optimizing'] + " - runtime")


# Good luck!