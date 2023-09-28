import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

(xRaw, yRaw) = SMSSpamDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

doModelEvaluation = True
if doModelEvaluation:
    ######
    ### Build a model and evaluate on validation data
    stepSize = 1.0
    convergence = 0.001

    featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords = 25)

    xTrain      = featurizer.Featurize(xTrainRaw)
    xValidate   = featurizer.Featurize(xValidateRaw)
    xTest       = featurizer.Featurize(xTestRaw)

    frequentModel = LogisticRegression.LogisticRegression()
    frequentModel.fit(xTrain, yTrain, convergence=convergence, stepSize=stepSize, verbose=True)

    ######
    ### Use equation 5.1 from Mitchell to bound the validation set error and the true error
    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

    print("Logistic regression with 25 features by mutual information:")
    validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, frequentModel.predict(xValidate))
    print("Validation set accuracy: %.4f." % (validationSetAccuracy))
    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(validationSetAccuracy, len(xValidate), confidence)    
        print(" %.2f%% accuracy bound: %.4f - %.4f" % (confidence, lowerBound, upperBound))

    ### Compare to most common class model here...
    mccm = MostCommonClassModel.MostCommonClassModel()
    mccm.fit(xTrainRaw, yTrain)
    print("Most common class model:")
    validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, mccm.predict(xValidateRaw))
    print("Validation set accuracy: %.4f." % (validationSetAccuracy))
    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(validationSetAccuracy, len(xValidateRaw), confidence)
        print(" %.2f%% accuracy bound: %.4f - %.4f" % (confidence, lowerBound, upperBound))

# Set this to true when you've completed the previous steps and are ready to move on...
doCrossValidation = True
if doCrossValidation:
    import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation    
    numberOfFolds = 5

    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
    
    # Good luck!
    lrAccuracy = 0
    mcAccuracy = 0
    for i in range(numberOfFolds):
        xTrainRawCV, yTrainCV, xEvaluateRawCV, yEvaluateCV = CrossValidation.CrossValidation(xTrainRaw, yTrain, numberOfFolds, i)

        featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
        featurizer.CreateVocabulary(xTrainRawCV, yTrainCV, numMutualInformationWords=25)
        xTrainCV = featurizer.Featurize(xTrainRawCV)
        xValidateCV = featurizer.Featurize(xEvaluateRawCV)
        frequentModel = LogisticRegression.LogisticRegression()
        frequentModel.fit(xTrainCV, yTrainCV, convergence=convergence, stepSize=stepSize, verbose=True)
        lrAccuracy += EvaluateBinaryClassification.Accuracy(yEvaluateCV, frequentModel.predict(xValidateCV))

        mccm = MostCommonClassModel.MostCommonClassModel()
        mccm.fit(xTrainRawCV, yTrainCV)
        mcAccuracy += EvaluateBinaryClassification.Accuracy(yEvaluateCV, mccm.predict(xEvaluateRawCV))
    lrAccuracy /= numberOfFolds
    mcAccuracy /= numberOfFolds
    print("Validation set accuracy - logistic regression: %.4f." % (lrAccuracy))
    print("Validation set accuracy - most common: %.4f." % (mcAccuracy))
    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(lrAccuracy, len(xTrainRaw), confidence)
        print(" %.2f%% accuracy bound - logistic regression: %.4f - %.4f" % (confidence, lowerBound, upperBound))
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(mcAccuracy, len(xTrainRaw), confidence)
        print(" %.2f%% accuracy bound - most common: %.4f - %.4f" % (confidence, lowerBound, upperBound))
