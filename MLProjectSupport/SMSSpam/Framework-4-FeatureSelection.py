kOutputDirectory = "C:\\temp\\visualize"

import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamSupport as SMSSpamSupport

kDataPath = "MachineLearningCourse\\MLProjectSupport\\SMSSpam\\dataset\\SMSSpamCollection"

(xRaw, yRaw) = SMSSpamSupport.LoadRawData(kDataPath)
(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = SMSSpamSupport.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamFeaturize as SMSSpamFeaturize

findTop10Words = True
if findTop10Words:
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)

    print("Top 10 words by frequency: ", featurizer.FindMostFrequentWords(xTrainRaw, 10))
    print("Top 10 words by mutual information: ", featurizer.FindTopWordsByMutualInformation(xTrainRaw, yTrain, 10))

# set to true when your implementation of the 'FindWords' stuff is working
doModeling = False
if doModeling:
    # Now get into model training
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification

    # The hyper parameters to use with logistic regression for this assignment
    stepSize = 0.1
    convergence = 0.0001

    # Remeber to create a new featurizer object/vocabulary for each part of the assignment
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = 10)

    # Remember to reprocess the raw data whenever you change the featurizer
    xTrain      = featurizer.Featurize(xTrainRaw)
    xValidate   = featurizer.Featurize(xValidateRaw)
    xTest       = featurizer.Featurize(xTestRaw)

    ## Good luck!