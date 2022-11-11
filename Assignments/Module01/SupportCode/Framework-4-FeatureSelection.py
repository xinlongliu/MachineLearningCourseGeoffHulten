import os

kOutputDirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualize')

import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

(xRaw, yRaw) = SMSSpamDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize

findTop10Words = True
if findTop10Words:
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)

    print("Top 10 words by frequency: ", featurizer.FindMostFrequentWords(xTrainRaw, 10))
    print("Top 10 words by mutual information: ", featurizer.FindTopWordsByMutualInformation(xTrainRaw, yTrain, 10))

# set to true when your implementation of the 'FindWords' part of the assignment is working
doModeling = True
if doModeling:
    # Now get into model training
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification

    # The hyperparameters to use with logistic regression for this assignment
    stepSize = 1.0
    convergence = 0.001

    # Remeber to create a new featurizer object/vocabulary for each part of the assignment
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords=15)

    # Remember to reprocess the raw data whenever you change the featurizer
    xTrain      = featurizer.Featurize(xTrainRaw)
    xValidate   = featurizer.Featurize(xValidateRaw)
    xTest       = featurizer.Featurize(xTestRaw)

    ## Good luck!
    logisticRegressionModel = LogisticRegression.LogisticRegression()
    logisticRegressionModel.fit(xTrain, yTrain, stepSize=stepSize, convergence=convergence)
    logisticRegressionModel.visualize()
    EvaluateBinaryClassification.ExecuteAll(yValidate,
                                            logisticRegressionModel.predict(xValidate, classificationThreshold=0.5))

    import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel
    mostCommonModel = MostCommonClassModel.MostCommonClassModel()
    # go read the ModelMostCommon code to see what model.fit does
    mostCommonModel.fit(xTrain, yTrain)
    mostCommonModel.visualize()
    EvaluateBinaryClassification.ExecuteAll(yValidate, mostCommonModel.predict(xValidate))

    # Use numMutualInformationWords
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords=10)
    xTrain = featurizer.Featurize(xTrainRaw)
    xValidate = featurizer.Featurize(xValidateRaw)
    xTest = featurizer.Featurize(xTestRaw)
    logisticRegressionModel = LogisticRegression.LogisticRegression()
    logisticRegressionModel.fit(xTrain, yTrain, stepSize=stepSize, convergence=convergence)
    logisticRegressionModel.visualize()
    EvaluateBinaryClassification.ExecuteAll(yValidate,
                                            logisticRegressionModel.predict(xValidate, classificationThreshold=0.5))

    # Sweep on numFrequentWords
    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
    trainLosses, validationLosses, lossXLabels = [], [], []
    for num in [1, 10, 20, 30, 40, 50]:
        # Remeber to create a new featurizer object/vocabulary for each part of the assignment
        featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
        featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords=num)
        xTrain = featurizer.Featurize(xTrainRaw)
        xValidate = featurizer.Featurize(xValidateRaw)
        xTest = featurizer.Featurize(xTestRaw)
        logisticRegressionModel = LogisticRegression.LogisticRegression()
        logisticRegressionModel.fit(xTrain, yTrain, stepSize=stepSize, convergence=convergence)
        logisticRegressionModel.visualize()
        EvaluateBinaryClassification.ExecuteAll(yValidate,
                                                logisticRegressionModel.predict(xValidate, classificationThreshold=0.5))
        trainLosses.append(logisticRegressionModel.loss(xTrain, yTrain))
        validationLosses.append(logisticRegressionModel.loss(xValidate, yValidate))
        lossXLabels.append(num)
    Charting.PlotSeries([trainLosses, validationLosses], ['Train', 'Validate'], lossXLabels,
                        chartTitle="Logistic Regression", xAxisTitle="Gradient Descent Steps", yAxisTitle="Avg. Loss",
                        outputDirectory=kOutputDirectory, fileName="4-Sweep on numFrequentWords Train vs Validate loss")

    # Sweep on numMutualInformationWords
    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
    trainLosses, validationLosses, lossXLabels = [], [], []
    for num in [1, 10, 20, 30, 40, 50]:
        # Remeber to create a new featurizer object/vocabulary for each part of the assignment
        featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
        featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords=num)
        xTrain = featurizer.Featurize(xTrainRaw)
        xValidate = featurizer.Featurize(xValidateRaw)
        xTest = featurizer.Featurize(xTestRaw)
        logisticRegressionModel = LogisticRegression.LogisticRegression()
        logisticRegressionModel.fit(xTrain, yTrain, stepSize=stepSize, convergence=convergence)
        logisticRegressionModel.visualize()
        EvaluateBinaryClassification.ExecuteAll(yValidate,
                                                logisticRegressionModel.predict(xValidate, classificationThreshold=0.5))
        trainLosses.append(logisticRegressionModel.loss(xTrain, yTrain))
        validationLosses.append(logisticRegressionModel.loss(xValidate, yValidate))
        lossXLabels.append(num)
    Charting.PlotSeries([trainLosses, validationLosses], ['Train', 'Validate'], lossXLabels,
                        chartTitle="Logistic Regression", xAxisTitle="Gradient Descent Steps", yAxisTitle="Avg. Loss",
                        outputDirectory=kOutputDirectory, fileName="4-Sweep on numMutualInformationWords Train vs Validate loss")

    # Use numMutualInformationWords = 100
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords=100)
    xTrain = featurizer.Featurize(xTrainRaw)
    xValidate = featurizer.Featurize(xValidateRaw)
    xTest = featurizer.Featurize(xTestRaw)
    logisticRegressionModel = LogisticRegression.LogisticRegression()
    logisticRegressionModel.fit(xTrain, yTrain, stepSize=stepSize, convergence=convergence)
    logisticRegressionModel.visualize()
    EvaluateBinaryClassification.ExecuteAll(yValidate,
                                            logisticRegressionModel.predict(xValidate, classificationThreshold=0.5))
