import numpy as np
import matplotlib.pyplot as plt
import random as rnd

### QUESTION 1
# PART a)
x = np.array([[-1,-1],
              [-1,1],
              [1,1],
              [1,-1]])
y = np.transpose(np.array([1,-1,1,-1]))
print('X\n',x)
print('Y\n',y)

# Just assigning zero initially for alphas
alphas = np.array([0,0,0,0])


# PART b)
# Defining Kernel Function
def kernelf(x1, x2):
    output = (np.dot(x1, x2) + 1)**2
    return output

# Initializing Kernel Matrix
K_Matrix = np.zeros((np.shape(x)[0], np.shape(x)[0])) 

# Calculating Kernel Matrix
for i in range(np.shape(x)[0]):
    for j in range(np.shape(x)[0]):
        K_Matrix[i,j] = kernelf(x[i], x[j])
print('Kernel Matrix\n', 
      K_Matrix)

# Checking the Mercer's Condition
# 1 - Symmetry
K_Transpose = np.transpose(K_Matrix) # It is the same as K_Matrix

# 2 - Positive Definiteness
eigenvalues = np.linalg.eigvals(K_Matrix) # They all greater than zero (0)
print('Eigenvalues of Kernel Matrix\n', eigenvalues)

# PART c)
yn_ym_Matrix = np.zeros((np.shape(K_Matrix))) # Initializing matrix of yn*ym values

# Calculating matrix of yn*ym values
for i in range(np.shape(yn_ym_Matrix)[0]):
    for j in range(np.shape(yn_ym_Matrix)[0]):
        yn_ym_Matrix[i,j] = np.dot(y[i], y[j])
        
# Calculating the matrix of yn*ym*K(xn,xm) values
yn_ym_K_Matrix = np.zeros((np.shape(K_Matrix))) # Initialization

for i in range(np.shape(yn_ym_K_Matrix)[0]):
    for j in range(np.shape(yn_ym_K_Matrix)[0]):
        yn_ym_K_Matrix[i,j] = np.dot(yn_ym_Matrix[i,j], K_Matrix[i,j])
# this matrix is beneficial for writing down explicit formulation      
    
 
# alphas are calculated by hand;
alphas = np.array([1/8, 1/8, 1/8, 1/8])  
print('Alpha Values\n', alphas)
print('All alphas are greater than zero, they are all support vectors')
# all alphas are greater than zero, they are all support vectors     

Q_alpha = np.sum(alphas)
for i in range(np.shape(x)[0]):
    for j in range(np.shape(x)[0]):
        Q_alpha -= 0.5*(y[i]*y[j]*alphas[i]*alphas[j]*kernelf(x[i],x[j]))
print("Q(alpha) =", Q_alpha) 

  
# PARD d)

second_orders_Matrix = np.zeros((np.shape(x)[0],5))
# new variables = x1^2, sqrt(2)*x1*x2, x2^2, sqrt(2)*x1, sqrt(2)*x2 
print('New variables = x1^2, sqrt(2)*x1*x2, x2^2, sqrt(2)*x1, sqrt(2)*x2')
def second_order_trans(x):
    trans = np.array([x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2, 
                      np.sqrt(2)*x[0], np.sqrt(2)*x[1]])
    return trans

for i in range(np.shape(x)[0]):
    second_orders_Matrix[i] = second_order_trans(x[i])
print('Vectors of the second order polynomial non-linear transformation for the data\n', second_orders_Matrix)
    
# PART e)
# Primal Solutions

weights_sv = np.zeros((1,np.shape(second_orders_Matrix)[1]))
for i in range(np.shape(second_orders_Matrix)[0]):
    weights_sv += y[i]*alphas[i]*second_orders_Matrix[i]

    
bias = 0

#since they all support vectors, we can select one of them, let say 1st data point
chosen_SV_index = 0
for i in range(np.shape(x)[0]):
    bias += y[i]*alphas[i]*np.dot(second_orders_Matrix[i],
                                  second_orders_Matrix[chosen_SV_index])
    
bias = y[chosen_SV_index] - bias
print("Primal Solutions\n", "Weights = ", weights_sv, "Bias = ", bias) # It is too small (2.2e-16), can be accepted as zero



### QUESTION 2
def calculatePredictedValues(trainmodified, weights):
    predictions = np.ones(len(trainmodified))
    for n in range(len(trainmodified)):
        probability = 1/(1+np.exp(-np.dot(trainmodified[n], np.transpose(weights))))
        if probability>0.5:
            predictions[n] = 1
        else:
            predictions[n] = -1
    return predictions

def UpdateWeights(trainmodified, trainlabels, weights,learningRate):
    # staochastic gradient descent regarding a random instance
    selectedinstances = rnd.randint(0,len(trainmodified)-1)

    expo = 1 + np.exp(trainlabels[selectedinstances]*(np.dot(weights, np.transpose(trainmodified[selectedinstances]))))
    weights = weights + trainlabels[selectedinstances] * trainmodified[selectedinstances] * learningRate / expo

    return weights


def CalculateLossFuncValue(trainmodified, trainlabels, weights):
    loss = 0
    for n in range(len(trainmodified)):
        loss +=np.log(1 + np.exp(-trainlabels[n]*(np.dot(weights, np.transpose(trainmodified[n])))))

    loss = loss /len(trainmodified)
    return loss


#algorithm
rnd.seed(2021)
trainfeatures = np.load('train_features.npy')
trainlabels = np.load('train_labels.npy')

# add a new variable in the first index for bias for each instance
trainmodified = trainfeatures
trainmodified = np.insert(trainmodified, 0,np.ones(len(trainmodified)) , axis=1)

# 10 fold cross validation on train set, randomy select instances to each fold with replacements (some instances can
# be included in many folds, while some instances may not be chosen as test instances
K = 10
sizeofsample = int(np.round(len(trainmodified) / K, 0))  # to generate equal sized folds from train data
indices = []
for ind in range(len(trainmodified)):
    indices.append(ind)
samples = []
for s in range(K):
    list1 = rnd.sample(indices, sizeofsample)
    samples.append(list1)
maxAccuracy = 0

outF = open("results.txt", "w")
outF.close()
for k in range(K):
    # initialize initial weights and bias w[0] is the bias, and w[1], w[2] is the weights for x1, x2
    weights = np.array(np.zeros([3], dtype=float))
    learningRate = 5
    rnd.seed(100*k)
    testinstances = np.zeros((len(samples[k]),3), dtype= float)
    testInslabels = np.zeros((len(samples[k]),1), dtype= int)
    traininstances = np.zeros(((len(trainmodified)-len(samples[k])), 3), dtype=float)
    trainInslabels = np.zeros(((len(trainmodified)-len(samples[k])), 1), dtype=int)
    countTest = 0
    countTrain = 0
    for n in range(len(trainmodified)):
        if samples[k].__contains__(n):
            testinstances[countTest,] = trainmodified[n,]
            testInslabels[countTest] = trainlabels[n]
            countTest+=1
        else:
            traininstances[countTrain,] = trainmodified[n,]
            trainInslabels[countTrain] = trainlabels[n]
            countTrain += 1
    for iter in range(8):
        iter += 1
        for epoch in range(1500):
            epoch+=1
            # update weights using gradient descent
            weights = UpdateWeights(traininstances, trainInslabels, weights, 1)
            # update learning rate
            learningRate = learningRate / ((iter-1)*1500 + epoch)
        loss = CalculateLossFuncValue(traininstances,trainInslabels,weights)
        outF = open("results.txt", "a")
        outF.write(str(k) + "\t" + str(iter)+ "\t"+str(epoch)+"\t"+str(loss))
        outF.write("\n")
        outF.close()
    predictions = calculatePredictedValues(testinstances, weights)
    accuracyTestFold = 0
    misclassifiedTestFold = 0
    for n in range(len(testinstances)):
        if predictions[n] == testInslabels[n]:
            accuracyTestFold += 1
        else:
            misclassifiedTestFold += 1
    accuracyLevelTestFold= 100 * (accuracyTestFold / len(testInslabels))
    accuracyLevelTestFold = np.round(accuracyLevelTestFold, 2)
    if accuracyLevelTestFold > maxAccuracy:
        maxAccuracy = accuracyLevelTestFold
        BestWeights = weights
    outF = open("results.txt", "a")
    outF.write(str(k)  + "\t" + str(accuracyLevelTestFold))
    outF.write("\n")
    outF.close()
# results on training set
weights = BestWeights
predictions = calculatePredictedValues(trainmodified, weights)
accuracyTrain = 0
misclassifiedTrain = 0
for n in range(len(trainmodified)):
    if predictions[n] == trainlabels[n]:
        accuracyTrain +=1
    else:
        misclassifiedTrain +=1

accuracyLevelTrain = 100* (accuracyTrain / len(trainlabels))
accuracyLevelTrain = np.round(accuracyLevelTrain,2)

plt.scatter( trainmodified[:, 1], trainmodified[:, 2],
              c= 10 +(predictions))
plt.title('Predicted Classes for Train Set')
plt.figtext(0.6, .8, "Accuracy level: "+str(accuracyLevelTrain)+ "%")
plt.savefig('PredictedClassesForTrain.png')
plt.show()

plt.scatter( trainmodified[:, 1], trainmodified[:, 2],
              c= (trainlabels +10))
plt.title('True Classes for Train Set')
plt.savefig('TrueClassesForTrain.png')
plt.show()

# results on test data
testfeatures = np.load('test_features.npy')
testlabels = np.load('test_labels.npy')

testmodified = testfeatures
testmodified = np.insert(testmodified, 0,np.ones(len(testmodified)) , axis=1)
predictions = calculatePredictedValues(testmodified, weights)

accuracyTest = 0
misclassifiedTest = 0
for n in range(len(testmodified)):
    if predictions[n] == testlabels[n]:
        accuracyTest +=1
    else:
        misclassifiedTest +=1

accuracyLevelTest = 100* (accuracyTest / len(testlabels))
accuracyLevelTest = np.round(accuracyLevelTest,2)

plt.scatter( testmodified[:, 1], testmodified[:, 2],
              c= 10 + (predictions))
plt.title('Predicted Classes for Test Set')
plt.figtext(0.6, .8, "Accuracy level: "+str(accuracyLevelTest)+"%")
plt.savefig('PredictedClassesForTest.png')
plt.show()

plt.scatter( testmodified[:, 1], testmodified[:, 2],
              c= (testlabels +10))
plt.title('True Classes for Test Set')
plt.savefig('TrueClassesForTest.png')
plt.show()




