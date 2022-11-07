# This file contains stubs for evaluating binary classifications. You must complete these functions as part of your assignment.
#     Each function takes in: 
#           'y':           the arrary of 0/1 true class labels; 
#           'yPredicted':  the prediction your model made for the cooresponding example.


def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value not in [0, 1]:
            valueError = True
    for value in yPredicted:
        if value not in [0, 1]:
            valueError = True

    if valueError:
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected values. Must be 0 or 1.")

def Accuracy(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct.append(1)
        else:
            correct.append(0)

    return sum(correct)/len(correct)

def Precision(y, yPredicted):
    # print("Stub precision in ", __file__)
    TP = PP = 0
    for i in range(len(y)):
        if yPredicted[i]:
            PP += 1
            if y[i]:
                TP += 1

    return TP / PP if PP > 0 else None

def Recall(y, yPredicted):
    # print("Stub Recall in ", __file__)
    TP = AP = 0
    for i in range(len(y)):
        if y[i]:
            AP += 1
            if yPredicted[i]:
                TP += 1

    return TP / AP if AP > 0 else None

def FalseNegativeRate(y, yPredicted):
    # print("Stub FalseNegativeRate in ", __file__)
    FN = AP = 0
    for i in range(len(y)):
        if y[i]:
            AP += 1
            if not yPredicted[i]:
                FN += 1

    return FN / AP if AP > 0 else None

def FalsePositiveRate(y, yPredicted):
    # print("Stub FalsePositiveRate in ", __file__)
    FP = AN = 0
    for i in range(len(y)):
        if not y[i]:
            AN += 1
            if yPredicted[i]:
                FP += 1

    return FP / AN if AN > 0 else None

def ConfusionMatrix(y, yPredicted):
    # This function should return: [[<# True Negatives>, <# False Positives>], [<# False Negatives>, <# True Positives>]]
    #  Hint: writing this function first might make the others easier...
    TN = FP = FN = TP = 0
    for i in range(len(y)):
        if y[i]:
            if yPredicted[i]:
                TP += 1
            else:
                FN += 1
        else:
            if yPredicted[i]:
                FP += 1
            else:
                TN += 1
    # print("Stub preConfusionMatrix in ", __file__)
    return [[TN, FP], [FN, TP]]

def ExecuteAll(y, yPredicted):
    print(ConfusionMatrix(y, yPredicted))
    print("Accuracy:", Accuracy(y, yPredicted))
    print("Precision:", Precision(y, yPredicted))
    print("Recall:", Recall(y, yPredicted))
    print("FPR:", FalsePositiveRate(y, yPredicted))
    print("FNR:", FalseNegativeRate(y, yPredicted))
    
