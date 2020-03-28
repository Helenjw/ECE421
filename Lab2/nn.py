import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load('notMNIST.npz') as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget
  
# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest    
  
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)

def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

"""1.1 Helper Functions"""

def relu(x):
  return np.maximum(x, 0)

def softmax(x):
  x -= np.max(x, axis=-1, keepdims=True) # axis should be last
  return np.exp(x) / np.sum( np.exp(x), axis=-1, keepdims=True) # axis should be last

def computeLayer(X, W, b):
  return X @ W + b

def CE(target, prediction):
  # print("preidiction:",prediction)
  return -np.mean(np.sum(target*np.log(prediction)), axis = -1)
   
def gradCE(target, prediction):    
  #prediction = softmax(o)
  return prediction - target #di(L)/di(o)

"""1.2 Backpropogation Derivation"""

def gradOuterWeights(target, prediction, hiddenPrediction):
  #di(L)/di(o) * di(o)/di(Wo) = di(L)/di(Wo)
  return hiddenPrediction.T @ gradCE(target, prediction)

def gradOuterBias(target, prediction):
  return np.sum(gradCE(target, prediction), axis = 0)

def gradInnerWeights (target, prediction, x, Wo, hiddenPrediction): 
  #hiddenPrediction = Whx + bh
  reluGrad = (hiddenPrediction > 0).astype(np.int32) #gradient of ReLU
  return x.T @ ( reluGrad * ((gradCE(target, prediction) @ Wo.T)) ) # switch relu product back to outside

# Assume this was actually for inner bias
def gradInnerBias(target, hiddenPrediction, prediction, Wo): 
  reluGrad = (hiddenPrediction > 0).astype(np.int32) #gradient of ReLU
  return np.sum( reluGrad * ((gradCE(target, prediction) @ Wo.T)), axis=0) # switch relu product back to outside

"""1.3 Learning"""

def TrainNN(data, target, epochs, hidden_units, alpha, gamma):
  # Reshape data
  data = data.reshape( data.shape[0], data.shape[1]**2 )
  
  # Initialize weights, bias and v for hidden and output layers
  h_weight, h_bias, h_v_w, h_v_b = initWeightBiasV(0, data.shape[1], hidden_units)
  o_weight, o_bias, o_v_w, o_v_b = initWeightBiasV(0, hidden_units, 10)

  # Accuracy and loss accounting
  train_accuracy = []
  train_loss = []

  for epoch in range(epochs):
    print("Epoch:", epoch)
    
    # Get forward prop prediction
    hiddenPrediction, trainPrediction = getForwardProp( data, h_weight, h_bias, o_weight, o_bias )

    # Get accuracy and loss
    train_accuracy.append( getAccuracy(trainPrediction, target) )
    train_loss.append( CE(target, trainPrediction) )

    # Update weights
    h_weight, h_bias, h_v_w, h_v_b = updateHiddenWeights( alpha, gamma, data, target, trainPrediction, hiddenPrediction, o_weight, h_bias, h_v_w, h_v_b, h_weight )
    o_weight, o_bias, o_v_w, o_v_b = updateOuterWeights( alpha, gamma, target, trainPrediction, hiddenPrediction, o_weight, o_bias, o_v_w, o_v_b )
  
  # report accuracies and losses
  print("Final Accuracy:", train_accuracy[epochs-1])
  print("Final Loss:", train_loss[epochs-1])
  PlotAccuracyLoss(train_loss, train_accuracy, epochs, hidden_units, alpha, gamma)

def initWeightBiasV(mean, in_size, out_size):
  ## Xavier Init weights for hidden and output layer
  # | w(0,0)          ...          w(0, output layer size -1) |
  # | w(input size, 0) ..  w(input size, output layer size -1)|
  stdev = np.sqrt( 2.0 / (in_size + out_size) )
  w = np.random.normal( mean, stdev, (in_size, out_size) )
  b = np.zeros( (out_size) ) # zeros is better
  v_w = np.full( w.shape, 1e-5 ) # too big by x10
  v_b = np.full( b.shape, 1e-5 ) # ditto
  return w, b, v_w, v_b

def getForwardProp(input, h_weight, h_bias, o_weight, o_bias):
  hiddenPrediction = relu( computeLayer(input, h_weight, h_bias) )
  NN_Output = softmax( computeLayer(hiddenPrediction, o_weight, o_bias) )
  print("Hidden Layer Output:")
  print(hiddenPrediction)
  print("NN Prediction:")
  print(NN_Output)
  return hiddenPrediction, NN_Output

def getAccuracy(prediction, target):
  # Get number of elements that are equal and return equal el / total el
  comparison = np.equal( np.argmax(prediction, axis=1), np.argmax(target, axis=1) )
  return np.mean( comparison )

def updateHiddenWeights( alpha, gamma, input, target, prediction, hiddenPrediction, o_weight, h_bias, h_v_w, h_v_b, h_weight ):
  # update weights
  grad_w = gradInnerWeights(target, prediction, input, o_weight, hiddenPrediction)
  v_w_new = gamma * h_v_w + alpha * grad_w
  h_weight -= v_w_new

  # update bias
  grad_b = gradInnerBias(target, hiddenPrediction, prediction, o_weight)
  v_b_new = gamma * h_v_b + alpha * grad_b
  h_bias -= v_b_new

  return h_weight, h_bias, v_w_new, v_b_new

def updateOuterWeights(alpha, gamma, target, prediction, hiddenPrediction, o_weight, o_bias, o_v_w, o_v_b ):
  # Update weights
  grad_w = gradOuterWeights(target, prediction, hiddenPrediction)
  v_w_new = gamma * o_v_w + alpha * grad_w
  o_weight -= v_w_new

  # Update bias
  grad_b = gradOuterBias(target, prediction)
  v_b_new = gamma * o_v_b + alpha * grad_b
  o_bias -= v_b_new

  return o_weight, o_bias, v_w_new, v_b_new

def PlotAccuracyLoss(train_loss, train_acc, epochs, hidden_units, alpha, gamma):
    x_axis = np.arange(0, epochs).tolist()

    #plot accuracies
    title = "Accuracy\nEpochs:" + str(epochs) + "  hidden units:" + str(hidden_units) + "  alpha:" + str(alpha) + "  gamma:" + str(gamma)
    plt.title(title)
    plt.plot(x_axis, train_acc, 'r', label = "Training")
    # plt.plot(x_axis,validation_acc,'b', label = "Validation")
    # plt.plot(x_axis,test_acc,'g', label = "Testing")
    plt.legend(loc="lower right")
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Epochs')
    plt.show()

    #plot losses
    title = "Loss\nEpochs:" + str(epochs) + "  hidden units:" + str(hidden_units) + "  alpha:" + str(alpha) + "  gamma:" + str(gamma)
    plt.title(title)
    plt.plot(x_axis,train_loss,'r', label = "Training")
    # plt.plot(x_axis,valid_loss,'b', label = "Validation")
    # plt.plot(x_axis,testing_loss,'g', label = "Testing")
    plt.legend(loc="upper right")
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    plt.show()
