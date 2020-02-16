import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import e

from google.colab import drive
drive.mount('/content/drive')

file_dir = '/content/drive/My Drive/Colab Notebooks/'

def loadData():
    with np.load(file_dir + 'notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Extract data
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData();

Weight = np.full((trainData.shape[1] * trainData.shape[1] ,1 ), 0)

# Reshaping data
trainData = trainData.reshape(trainData.shape[0], trainData.shape[1] * trainData.shape[1])
validData = validData.reshape(validData.shape[0], validData.shape[1] * validData.shape[1])
testData = testData.reshape(testData.shape[0], testData.shape[1] * testData.shape[1])

def TrainModel( trainData, validData, testData, trainTarget, validTarget, testTarget, reg, losstype, epochs, batchSize, b1, b2, e ):

  # Store
  training_loss = []
  valid_loss = []
  test_loss = []
  training_acc = []
  valid_acc = []
  test_acc = []

  # Initialize tensors
  W, B, X, Y_hat, Y, Reg, Loss, Optimizer = buildGraph(losstype, b1, b2, e)

  # launch session
  sess = tf.InteractiveSession()
  sess.run( tf.global_variables_initializer() )

  # Training 
  for epoch in range(epochs):

    # Shuffle data and labels, make run graph
    shuffledData, shuffledTargets = Shuffle( trainData, trainTarget, epoch )

    # Train batch
    for batch in range( int(trainData.shape[0]/batchSize) ):
      batchData, batchTarget = makeBatch(shuffledData, shuffledTargets, batch, batchSize)
      w, b, y_hat, loss, optimizer = sess.run( [W, B, Y_hat, Loss, Optimizer], feed_dict={X:batchData, Y:batchTarget, Reg:reg} )
    
    w_, b_, y_hat_test, testloss = sess.run( [W, B, Y_hat, Loss], feed_dict={X:testData, Y:testTarget, Reg:reg} )
    w_, b_, y_hat_valid, validloss = sess.run( [W, B, Y_hat, Loss], feed_dict={X:validData, Y:validTarget, Reg:reg} )

    print(epoch)
    training_loss.append( loss )
    test_loss.append( testloss )
    valid_loss.append( validloss )

    # Get accuracy
    acc = tf.compat.v1.placeholder( tf.float32, shape=(), name="acc" )
    validacc = tf.compat.v1.placeholder( tf.float32, shape=(2,1), name="validacc" )
    testacc = tf.compat.v1.placeholder( tf.float32, shape=(), name="testacc" )

    acc = sess.run( tf.math.reduce_mean(tf.cast(tf.math.equal(tf.math.greater( y_hat, tf.constant(0.5, tf.float32, shape=(y_hat.shape))), batchTarget), tf.float32)) )
    validacc = sess.run( tf.math.reduce_mean(tf.cast(tf.math.equal(tf.math.greater( y_hat_valid, tf.constant(0.5, tf.float32, shape=(y_hat_valid.shape))), validTarget), tf.float32)) )
    testacc = sess.run( tf.math.reduce_mean(tf.cast(tf.math.equal(tf.math.greater( y_hat_test, tf.constant(0.5, tf.float32, shape=(y_hat_test.shape))), testTarget), tf.float32)) )
    # print( sess.run([acc, validacc, testacc]) )

    training_acc.append( acc )
    test_acc.append( testacc )
    valid_acc.append( validacc )

  Plot( reg, training_loss, valid_loss, test_loss, epochs, batchSize, training_acc, valid_acc, test_acc )
  print( training_loss[epochs-1], valid_loss[epochs-1], test_loss[epochs-1], training_acc[epochs-1], valid_acc[epochs-1], test_acc[epochs-1] )
  
  def Shuffle( Data, Target, epoch ):
  np.random.seed(epoch)
  randIndx = np.arange( len(Data) )
  np.random.shuffle(randIndx)
  return Data[randIndx], Target[randIndx]
  
  def makeBatch( Data, Target, batchNo, batchSize ):
  lower = batchNo*batchSize
  upper = (batchNo+1)*batchSize
  return Data[lower:upper], Target[lower:upper]
  
    def Plot( reg, training_loss, valid_loss, testing_loss, epochs, batchSize, training_acc, validation_acc, test_acc ):

    x_axis = np.arange(0, epochs).tolist()

    #plot losses
    title = "Loss vs Epoch for reg = " + str(reg) + ", batch size = " + str(batchSize)
    plt.title(title)
    plt.plot(x_axis,training_loss,'r', label = "Training")
    plt.plot(x_axis,valid_loss,'b', label = "Validation")
    plt.plot(x_axis,testing_loss,'g', label = "Testing")
    plt.legend(loc="upper right")
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    plt.show()

    #plot accuracies
    title = "Accuracy vs Epoch for reg = " + str(reg) +  ", batch size = " + str(batchSize)
    plt.title(title)
    plt.plot(x_axis,training_acc,'r', label = "Training")
    plt.plot(x_axis,validation_acc,'b', label = "Validation")
    plt.plot(x_axis,test_acc,'g', label = "Testing")
    plt.legend(loc="lower right")
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Epochs')
    plt.show()
